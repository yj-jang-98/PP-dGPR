"""Luo-inspired secret-sharing simulation for exact local GPs and PoE.

Two arrays represent the computing servers; an assistant supplies masks
and Beaver triples. Local predictions remain shared until PoE aggregation.
The simulation uses floating-point shares and bounded masks with fixed-point
wrapping. All parties run in one process; this is not a network MPC library."""

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import time

import numpy as np
import torch


@dataclass(frozen=True)
class FixedPointConfig:
    word_bits: int = 64
    fractional_bits: int = 26
    modulus: int = 2**40
    # Luo reports l_f=26 but not the experimental r range.  Radius 8 keeps
    # exp(-r) representable at this precision; radius 16 requires l_f>=29.
    mask_radius: float = 8.0

    @property
    def scale(self):
        return 1 << self.fractional_bits


@dataclass
class Shared:
    """Two additive shares held by logical computing servers S0 and S1."""

    s0: np.ndarray
    s1: np.ndarray

    @property
    def shape(self):
        return self.s0.shape

    @property
    def ndim(self):
        return self.s0.ndim

    @property
    def T(self):
        return Shared(self.s0.T, self.s1.T)

    def copy(self):
        return Shared(self.s0.copy(), self.s1.copy())

    def __getitem__(self, key):
        return Shared(self.s0[key], self.s1[key])

    def __setitem__(self, key, value):
        if isinstance(value, Shared):
            self.s0[key], self.s1[key] = value.s0, value.s1
        else:
            self.s0[key], self.s1[key] = value, 0.0

    def __neg__(self):
        return Shared(-self.s0, -self.s1)

    def __add__(self, other):
        if isinstance(other, Shared):
            return Shared(self.s0 + other.s0, self.s1 + other.s1)
        return Shared(self.s0 + np.asarray(other), self.s1.copy())

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, Shared):
            return Shared(self.s0 - other.s0, self.s1 - other.s1)
        return Shared(self.s0 - np.asarray(other), self.s1.copy())

    def __rsub__(self, other):
        return Shared(np.asarray(other) - self.s0, -self.s1)

    def sum(self, axis=None, keepdims=False):
        return Shared(
            self.s0.sum(axis=axis, keepdims=keepdims),
            self.s1.sum(axis=axis, keepdims=keepdims),
        )


class Assistant:
    """Logical assistant T: preprocessing only, independent of private data."""

    def __init__(self, rng, fixed_point):
        self.rng = rng
        self.fp = fixed_point

    def quantize(self, value):
        value = np.asarray(value, dtype=np.float64)
        encoded = np.rint(value * self.fp.scale).astype(np.int64)
        q = self.fp.modulus
        encoded = (encoded + q // 2) % q - q // 2
        return encoded.astype(np.float64) / self.fp.scale

    def uniform_ring(self, shape):
        """Uniform fixed-point element of Z_q, represented in centered form."""
        encoded = self.rng.integers(
            -self.fp.modulus // 2,
            self.fp.modulus // 2,
            size=shape,
            dtype=np.int64,
        )
        return encoded.astype(np.float64) / self.fp.scale

    def share(self, value):
        value = self.quantize(value)
        mask = self.quantize(
            self.rng.uniform(-self.fp.mask_radius, self.fp.mask_radius, value.shape)
        )
        return Shared(mask, value - mask)

    def public(self, value):
        value = self.quantize(value)
        return Shared(value, np.zeros_like(value))

    def zeros(self, shape):
        return self.public(np.zeros(shape, dtype=np.float64))

    def beaver_elementwise(self, shape):
        a = self.quantize(self.rng.uniform(-1.0, 1.0, shape))
        b = self.quantize(self.rng.uniform(-1.0, 1.0, shape))
        return self.share(a), self.share(b), self.share(self.quantize(a * b))

    def beaver_matmul(self, left_shape, right_shape):
        a = self.quantize(self.rng.uniform(-0.5, 0.5, left_shape))
        b = self.quantize(self.rng.uniform(-0.5, 0.5, right_shape))
        return self.share(a), self.share(b), self.share(self.quantize(a @ b))

    def truncation_mask(self, shape):
        # A fixed-point grid mask permits round(z+r)-r without opening z.
        r = self.quantize(
            self.rng.uniform(-self.fp.mask_radius, self.fp.mask_radius, shape)
        )
        return self.share(r)


class LuoPPGPR:
    """Luo Algorithms 1--5 using two shares and assistant preprocessing."""

    def __init__(
        self,
        lengthscale=0.23,
        signal_variance=0.8,
        noise_variance=0.1,
        fixed_point=None,
        seed=1,
        communication_latency=0.0002,
        communication_bandwidth=625e6,
        simulate_serialization=True,
        dtype=torch.float64,  # retained for backward-compatible notebook API
    ):
        del dtype
        self.lengthscale = float(lengthscale)
        self.signal_variance = float(signal_variance)
        self.noise_variance = float(noise_variance)
        self.fixed_point = fixed_point or FixedPointConfig()
        self.rng = np.random.default_rng(seed)
        self.assistant = Assistant(self.rng, self.fixed_point)
        self.communication_latency = float(communication_latency)
        self.communication_bandwidth = float(communication_bandwidth)
        self.simulate_serialization = bool(simulate_serialization)
        # S0 and S1 are evaluated by separate workers, as if they execute
        # concurrently on different machines.
        self._server_pool = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="luo-server"
        )
        self._server_pool_closed = False
        self.serialization_time = 0.0
        self.communication = {
            "shared_input_elements": 0,
            "opened_masked_elements": 0,
            "final_opened_elements": 0,
            "pp_exp_elements": 0,
            "pp_exp_rounds": 0,
            "pp_mi_rounds": 0,
            "distance_rounds": 0,
            "matmul_rounds": 0,
            "secure_multiply_rounds_executed": 0,
            "secure_division_calls": 0,
            "online_bytes": 0,
            "data_upload_rounds": 0,
            "executed_exchange_rounds": 0,
        }

    @property
    def online_communication_rounds(self):
        return self.communication["executed_exchange_rounds"]

    @property
    def empirical_latency_time(self):
        return self.online_communication_rounds * self.communication_latency

    @property
    def empirical_bandwidth_time(self):
        return self.communication["online_bytes"] / self.communication_bandwidth

    @property
    def empirical_communication_time(self):
        return self.empirical_latency_time + self.empirical_bandwidth_time

    @property
    def latency_adjusted_time(self):
        return self.fit_time + self.predict_time + self.empirical_communication_time

    @property
    def communication_adjusted_time(self):
        return self.latency_adjusted_time

    def _parallel_servers(self, operation0, operation1):
        """Evaluate the two computing servers concurrently and independently."""
        future0 = self._server_pool.submit(operation0)
        future1 = self._server_pool.submit(operation1)
        return future0.result(), future1.result()

    def close(self):
        if not self._server_pool_closed:
            self._server_pool.shutdown(wait=True)
            self._server_pool_closed = True

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _transmit_bidirectional(self, s0, s1):
        """Serialize one message in each direction and return received copies."""
        a0 = np.ascontiguousarray(s0, dtype=np.float64)
        a1 = np.ascontiguousarray(s1, dtype=np.float64)
        self.communication["online_bytes"] += a0.nbytes + a1.nbytes
        if not self.simulate_serialization:
            return a0.copy(), a1.copy()
        start = time.perf_counter()
        wire0 = a0.tobytes(order="C")
        wire1 = a1.tobytes(order="C")
        received_at_s1 = np.frombuffer(wire0, dtype=np.float64).reshape(a0.shape).copy()
        received_at_s0 = np.frombuffer(wire1, dtype=np.float64).reshape(a1.shape).copy()
        self.serialization_time += time.perf_counter() - start
        return received_at_s1, received_at_s0

    def _open_masked(self, value, count_round=True):
        """The two computing servers exchange and reconstruct a masked value."""
        if count_round:
            self.communication["executed_exchange_rounds"] += 1
        self.communication["opened_masked_elements"] += np.size(value.s0)
        s0_at_s1, s1_at_s0 = self._transmit_bidirectional(value.s0, value.s1)
        opened0, opened1 = self._parallel_servers(
            lambda: value.s0 + s1_at_s0,
            lambda: value.s1 + s0_at_s1,
        )
        # Both servers obtain the same public masked value.  Keeping one copy
        # avoids an artificial single-machine memory penalty after exchange.
        return self.assistant.quantize(opened0)

    def _open_masked_pair(self, first, second):
        """Open two independent Beaver differences in one batched round."""
        self.communication["executed_exchange_rounds"] += 1
        return (
            self._open_masked(first, count_round=False),
            self._open_masked(second, count_round=False),
        )

    def _open_final(self, value):
        self.communication["executed_exchange_rounds"] += 1
        self.communication["final_opened_elements"] += np.size(value.s0)
        # Both shares are sent to the authorized model user.
        self.communication["online_bytes"] += value.s0.nbytes + value.s1.nbytes
        if self.simulate_serialization:
            start = time.perf_counter()
            wire0 = np.ascontiguousarray(value.s0).tobytes()
            wire1 = np.ascontiguousarray(value.s1).tobytes()
            s0 = np.frombuffer(wire0, dtype=np.float64).reshape(value.shape).copy()
            s1 = np.frombuffer(wire1, dtype=np.float64).reshape(value.shape).copy()
            self.serialization_time += time.perf_counter() - start
        else:
            s0, s1 = value.s0.copy(), value.s1.copy()
        return self.assistant.quantize(s0 + s1)

    def _truncate(self, value):
        mask = self.assistant.truncation_mask(value.shape)
        opened = self._open_masked(value + mask)
        rounded = self.assistant.quantize(opened)
        s0, s1 = self._parallel_servers(
            lambda: rounded - mask.s0,
            lambda: -mask.s1,
        )
        return Shared(s0, s1)

    def _mul(self, x, y, count=True):
        """One-round Beaver element-wise multiplication."""
        shape = np.broadcast_shapes(x.shape, y.shape)
        xb = Shared(np.broadcast_to(x.s0, shape), np.broadcast_to(x.s1, shape))
        yb = Shared(np.broadcast_to(y.s0, shape), np.broadcast_to(y.s1, shape))
        a, b, c = self.assistant.beaver_elementwise(shape)
        e, f = self._open_masked_pair(xb - a, yb - b)
        s0, s1 = self._parallel_servers(
            lambda: c.s0 + e * b.s0 + f * a.s0 + e * f,
            lambda: c.s1 + e * b.s1 + f * a.s1,
        )
        out = Shared(s0, s1)
        if count:
            self.communication["secure_multiply_rounds_executed"] += 1
        return self._truncate(out)

    def _matmul(self, x, y, count=True):
        """One-round Beaver matrix multiplication."""
        a, b, c = self.assistant.beaver_matmul(x.shape, y.shape)
        e, f = self._open_masked_pair(x - a, y - b)
        s0, s1 = self._parallel_servers(
            lambda: c.s0 + e @ b.s0 + a.s0 @ f + e @ f,
            lambda: c.s1 + e @ b.s1 + a.s1 @ f,
        )
        out = Shared(s0, s1)
        if count:
            self.communication["secure_multiply_rounds_executed"] += 1
        return self._truncate(out)

    def _divide(self, numerator, denominator, lower_bound=None, upper_bound=None):
        """Divide using eight Newton reciprocal steps and a final product.

        Positive public bounds initialize the reciprocal. Communication counts
        include the masked exchanges and fixed-point truncations executed here."""
        if lower_bound is None:
            lower_bound = self.noise_variance
        if upper_bound is None:
            upper_bound = self.signal_variance + self.noise_variance
        if not 0 < lower_bound <= upper_bound:
            raise ValueError("PP-Div requires positive public denominator bounds")
        initial = 2.0 / (lower_bound + upper_bound)
        reciprocal = self.assistant.public(np.full(denominator.shape, initial))
        two = self.assistant.public(np.full(denominator.shape, 2.0))
        for _ in range(8):
            dr = self._mul(denominator, reciprocal)
            reciprocal = self._mul(reciprocal, two - dr)
        self.communication["secure_division_calls"] += 1
        return self._mul(numerator, reciprocal)

    def _share_input(self, value):
        value = np.asarray(value, dtype=np.float64)
        self.communication["shared_input_elements"] += value.size
        # The data owner sends one 64-bit share to each computing server.
        self.communication["online_bytes"] += 2 * value.size * 8
        # Executed conceptually by the data owner, not by assistant T.
        value = self.assistant.quantize(value)
        owner_mask = self.assistant.quantize(
            self.rng.uniform(
                -self.fixed_point.mask_radius,
                self.fixed_point.mask_radius,
                value.shape,
            )
        )
        return Shared(owner_mask, value - owner_mask)

    @staticmethod
    def _concatenate(values, axis=0):
        return Shared(
            np.concatenate([v.s0 for v in values], axis=axis),
            np.concatenate([v.s1 for v in values], axis=axis),
        )

    def _squared_distance(self, x, y):
        # These independent products form one batched communication phase.
        x2 = self._mul(x, x, count=False).sum(axis=1, keepdims=True)
        y2 = self._mul(y, y, count=False).sum(axis=1)[None, :]
        xy = self._matmul(x, y.T, count=False)
        self.communication["distance_rounds"] += 1
        self.communication["secure_multiply_rounds_executed"] += 1
        return x2 + y2 - Shared(2.0 * xy.s0, 2.0 * xy.s1)

    def _pp_exp(self, u):
        """Algorithm 2: only d=u+r is opened; u remains secret-shared."""
        r_plain = self.assistant.quantize(
            self.rng.uniform(
                -self.fixed_point.mask_radius,
                self.fixed_point.mask_radius,
                u.shape,
            )
        )
        r = self.assistant.share(r_plain)
        exp_neg_r = self.assistant.share(np.exp(-r_plain))
        d = self._open_masked(u + r)
        public_exp_d = self.assistant.quantize(np.exp(d))
        s0, s1 = self._parallel_servers(
            lambda: public_exp_d * exp_neg_r.s0,
            lambda: public_exp_d * exp_neg_r.s1,
        )
        result = Shared(s0, s1)
        self.communication["pp_exp_elements"] += np.size(d)
        self.communication["pp_exp_rounds"] += 1
        return self._truncate(result)

    def _se_kernel(self, x, y):
        distance = self._squared_distance(x, y)
        exponent = Shared(
            -distance.s0 / (2.0 * self.lengthscale**2),
            -distance.s1 / (2.0 * self.lengthscale**2),
        )
        result = self._pp_exp(exponent)
        return Shared(
            self.signal_variance * result.s0,
            self.signal_variance * result.s1,
        )

    def _pp_matrix_inverse(self, matrix):
        """Algorithms 3--5, with every nonlinear operation on shares."""
        n = matrix.shape[0]
        lower = self.assistant.public(np.eye(n))
        diagonal = self.assistant.zeros((n,))

        diagonal[0] = matrix[0, 0]
        lower[:, 0] = self._divide(matrix[:, 0], diagonal[0])

        for k in range(1, n):
            row = lower[k, :k]
            squared = self._mul(row, row)
            terms = self._mul(squared, diagonal[:k])
            diagonal[k] = matrix[k, k] - terms.sum()
            if k + 1 < n:
                weighted_row = self._mul(row, diagonal[:k])
                correction = self._mul(
                    lower[k + 1 :, :k],
                    Shared(weighted_row.s0[None, :], weighted_row.s1[None, :]),
                ).sum(axis=1)
                numerator = matrix[k + 1 :, k] - correction
                lower[k + 1 :, k] = self._divide(numerator, diagonal[k])

        # Algorithm 4: secure forward substitution, one batched PP-MM per row.
        forward = self.assistant.public(np.eye(n))
        for k in range(1, n):
            products = self._mul(
                forward[:k, :k],
                Shared(lower[k, :k].s0[:, None], lower[k, :k].s1[:, None]),
            )
            forward[k, :k] = -products.sum(axis=0)

        # Algorithm 5: one vector-wise PP-Div and one PP-MM.
        scaled = self._divide(forward, diagonal[:, None])
        inverse = self._matmul(forward.T, scaled)
        self.communication["pp_mi_rounds"] += 22 * n - 6
        return inverse

    def fit(self, owner_x_parts, owner_y_parts):
        start = time.perf_counter()
        self.train_x = self._concatenate([self._share_input(x) for x in owner_x_parts])
        owner_y_arrays = [np.asarray(y) for y in owner_y_parts]
        self._single_output = all(y.ndim == 1 for y in owner_y_arrays)
        self.train_y = self._concatenate(
            [
                self._share_input(y.reshape(-1, 1) if y.ndim == 1 else y)
                for y in owner_y_arrays
            ]
        )
        # All owners upload concurrently.
        self.communication["data_upload_rounds"] += 1
        self.communication["executed_exchange_rounds"] += 1
        kernel = self._se_kernel(self.train_x, self.train_x)
        covariance = kernel + self.noise_variance * np.eye(kernel.shape[0])
        self.inverse = self._pp_matrix_inverse(covariance)
        self.alpha = self._matmul(self.inverse, self.train_y)
        self.communication["matmul_rounds"] += 1
        self.fit_time = time.perf_counter() - start
        return self

    def predict(self, test_x, batch_size=128):
        mean, variance = self.predict_shared(test_x, batch_size=batch_size)
        opened_mean = self._open_final(mean)
        opened_variance = np.maximum(self._open_final(variance), 0.0)
        self.close()
        if self._single_output:
            opened_mean = opened_mean.reshape(-1)
        return opened_mean, opened_variance

    def predict_shared(self, test_x, batch_size=128, include_observation_noise=False):
        """Return secret-shared predictions without opening local expert values."""
        start = time.perf_counter()
        shared_test_x = self._share_input(test_x)
        self.communication["data_upload_rounds"] += 1
        self.communication["executed_exchange_rounds"] += 1
        means, variances = [], []
        for begin in range(0, shared_test_x.shape[0], batch_size):
            batch = shared_test_x[begin : begin + batch_size]
            cross = self._se_kernel(batch, self.train_x)
            mean = self._matmul(cross, self.alpha)
            projected = self._matmul(cross, self.inverse)
            variance_term = self._mul(projected, cross).sum(axis=1)
            prior_variance = self.signal_variance
            if include_observation_noise:
                prior_variance += self.noise_variance
            variance = prior_variance - variance_term
            means.append(mean)
            variances.append(variance)
            self.communication["matmul_rounds"] += 2
        self.predict_time = time.perf_counter() - start
        return self._concatenate(means), self._concatenate(variances)


class LuoSSPoE:
    """Luo-inspired server-side PoE from shares of owner data only.

    Every local expert is evaluated with PP-Exp and PP-MI. Local predictive
    values remain shared; only the final PoE mean and variance are opened.
    Independent owner experts execute concurrently, while all element-wise
    PoE divisions are vectorized into two secure division phases.
    """

    def __init__(self, *, owner_workers=4, **model_kwargs):
        self.owner_workers = int(owner_workers)
        if self.owner_workers < 1:
            raise ValueError("owner_workers must be positive")
        self.model_kwargs = dict(model_kwargs)

    def fit_predict(
        self,
        owner_x_parts,
        owner_y_parts,
        test_x,
        batch_size=128,
        owner_hyperparameters=None,
    ):
        if len(owner_x_parts) != len(owner_y_parts) or not owner_x_parts:
            raise ValueError("owner partitions must be non-empty and aligned")
        for y in owner_y_parts:
            if np.asarray(y).ndim != 1:
                raise ValueError("LuoSSPoE currently expects one scalar output")
        if owner_hyperparameters is None:
            owner_hyperparameters = [None] * len(owner_x_parts)
        if len(owner_hyperparameters) != len(owner_x_parts):
            raise ValueError("owner_hyperparameters must align with owners")

        start = time.perf_counter()

        def local_expert(index):
            kwargs = {k: v for k, v in self.model_kwargs.items() if k != "seed"}
            if owner_hyperparameters[index] is not None:
                kwargs.update(owner_hyperparameters[index])
            model = LuoPPGPR(
                seed=self.model_kwargs.get("seed", 1) + index,
                **kwargs,
            )
            model.fit([owner_x_parts[index]], [owner_y_parts[index]])
            mean, variance = model.predict_shared(
                test_x,
                batch_size=batch_size,
                include_observation_noise=True,
            )
            mean = Shared(mean.s0.reshape(-1), mean.s1.reshape(-1))
            return model, mean, variance

        with ThreadPoolExecutor(max_workers=self.owner_workers) as pool:
            results = list(pool.map(local_expert, range(len(owner_x_parts))))
        models = [result[0] for result in results]
        local_means = Shared(
            np.stack([result[1].s0 for result in results]),
            np.stack([result[1].s1 for result in results]),
        )
        local_variances = Shared(
            np.stack([result[2].s0 for result in results]),
            np.stack([result[2].s1 for result in results]),
        )

        # Local protocols run concurrently, so latency follows the slowest
        # owner, while communication volume is the sum across all owners.
        local_rounds = max(model.online_communication_rounds for model in models)
        local_bytes = sum(model.communication["online_bytes"] for model in models)
        aggregator = models[0]
        before_rounds = aggregator.online_communication_rounds
        before_bytes = aggregator.communication["online_bytes"]
        signal_variances = np.asarray([model.signal_variance for model in models])
        noise_variances = np.asarray([model.noise_variance for model in models])
        variance_lower = float(noise_variances.min())
        variance_upper = float((signal_variances + noise_variances).max())
        ones = aggregator.assistant.public(np.ones(local_variances.shape))
        precisions = aggregator._divide(
            ones,
            local_variances,
            lower_bound=variance_lower,
            upper_bound=variance_upper,
        )
        weighted_means = aggregator._mul(local_means, precisions)
        precision_sum = precisions.sum(axis=0)
        weighted_sum = weighted_means.sum(axis=0)
        final_numerators = Shared(
            np.stack((weighted_sum.s0, np.ones_like(weighted_sum.s0))),
            np.stack((weighted_sum.s1, np.zeros_like(weighted_sum.s1))),
        )
        final_denominators = Shared(
            np.stack((precision_sum.s0, precision_sum.s0)),
            np.stack((precision_sum.s1, precision_sum.s1)),
        )
        precision_lower = float(np.sum(1.0 / (signal_variances + noise_variances)))
        precision_upper = float(np.sum(1.0 / noise_variances))
        final = aggregator._divide(
            final_numerators,
            final_denominators,
            lower_bound=precision_lower,
            upper_bound=precision_upper,
        )
        poe_mean = aggregator._open_final(final[0])
        poe_variance = np.maximum(aggregator._open_final(final[1]), 0.0)
        aggregation_rounds = aggregator.online_communication_rounds - before_rounds
        aggregation_bytes = aggregator.communication["online_bytes"] - before_bytes

        self.raw_time = time.perf_counter() - start
        self.online_communication_rounds = local_rounds + aggregation_rounds
        self.online_bytes = local_bytes + aggregation_bytes
        latency = self.model_kwargs.get("communication_latency", 0.0002)
        bandwidth = self.model_kwargs.get("communication_bandwidth", 625e6)
        self.empirical_latency_time = self.online_communication_rounds * latency
        self.empirical_bandwidth_time = self.online_bytes / bandwidth
        self.latency_adjusted_time = (
            self.raw_time + self.empirical_latency_time + self.empirical_bandwidth_time
        )
        self.local_predictions_opened = False
        self.final_opened_elements = 2 * np.size(poe_mean)
        for model in models:
            model.close()
        return poe_mean, poe_variance
