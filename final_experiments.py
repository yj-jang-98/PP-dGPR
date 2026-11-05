"""Prediction, consensus, and timing helpers for ``table1.ipynb``."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import time

import gpytorch
import networkx as nx
import numpy as np
import torch
from tqdm.auto import tqdm

from nawaz_sdsgp import lattigo_secure_sum
from share import privacy_preserving_avg_consensus


def build_graph_and_weights(m: int, neighbors: int, lw: float, seed: int):
    """Build a connected regular graph satisfying Assumption 2 and integer weights."""
    if not 1 <= neighbors < m:
        raise ValueError("neighbors must satisfy 1 <= neighbors < M")
    if (m * neighbors) % 2:
        raise ValueError("M * neighbors must be even for a regular graph")
    if neighbors == m - 1:
        graph = nx.complete_graph(m)
    elif neighbors >= 4 and neighbors % 2 == 0:
        graph = nx.power(nx.cycle_graph(m), neighbors // 2)
    else:
        rng = np.random.default_rng(seed)
        for _ in range(10_000):
            candidate = nx.random_regular_graph(
                neighbors, m, seed=int(rng.integers(0, 2**32 - 1))
            )
            violations = [
                (i, j)
                for i, j in candidate.edges()
                if not set(candidate.neighbors(i)).intersection(candidate.neighbors(j))
            ]
            if nx.is_connected(candidate) and not violations:
                graph = candidate
                break
        else:
            raise RuntimeError("Could not construct a graph satisfying Assumption 2")

    violations = [
        (i, j)
        for i, j in graph.edges()
        if not set(graph.neighbors(i)).intersection(graph.neighbors(j))
    ]
    if not nx.is_connected(graph) or violations:
        raise AssertionError(f"Assumption 2 violations: {violations}")
    if any(degree != neighbors for _, degree in graph.degree()):
        raise AssertionError("Graph is not Ni-regular")

    weights = np.zeros((m, m), dtype=np.float64)
    for i in range(m):
        for j in graph.neighbors(i):
            weights[i, j] = 1.0 / (2.0 * (1 + max(graph.degree[i], graph.degree[j])))
        weights[i, i] = 1.0 - weights[i].sum()
    scaled = weights / lw
    rounded = np.rint(scaled)
    if not np.allclose(scaled, rounded, atol=1e-12):
        raise ValueError("Scaled Metropolis weights are not integers")
    return graph, rounded.astype(np.int64)


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_sarcos_local_prediction(
    models, likelihoods, x_test, repeats: int, device, workers: int = 4
):
    """Time local GP predictions; return means and variances shaped (M, N, k)."""
    m, outputs = len(models), len(models[0])
    means = [[None] * outputs for _ in range(m)]
    variances = [[None] * outputs for _ in range(m)]
    times = []

    def predict_one(agent, output):
        model = models[agent][output]
        likelihood = likelihoods[agent][output]
        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            prediction = likelihood(model(x_test))
            return agent, output, prediction.mean, prediction.variance

    tasks = [(agent, output) for agent in range(m) for output in range(outputs)]
    with ThreadPoolExecutor(max_workers=min(workers, len(tasks))) as executor:
        for _ in tqdm(range(repeats), desc="SARCOS local prediction", unit="run"):
            _sync(device)
            start = time.perf_counter()
            for agent, output, mean, variance in executor.map(
                lambda item: predict_one(*item), tasks
            ):
                means[agent][output] = mean
                variances[agent][output] = variance
            _sync(device)
            times.append(time.perf_counter() - start)
    means_tensor = torch.stack([torch.stack(means[agent], dim=1) for agent in range(m)])
    vars_tensor = torch.stack(
        [torch.stack(variances[agent], dim=1) for agent in range(m)]
    )
    return means_tensor, vars_tensor, np.asarray(times)


def benchmark_diabetes_local_prediction(
    models, likelihoods, x_test, repeats: int, device, workers: int = 4
):
    """Time local GP predictions; return means and variances shaped (M, N)."""
    m = len(models)
    means, variances = [None] * m, [None] * m
    times = []

    def predict_one(agent):
        model, likelihood = models[agent], likelihoods[agent]
        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            prediction = likelihood(model(x_test))
            return agent, prediction.mean, prediction.variance

    with ThreadPoolExecutor(max_workers=min(workers, m)) as executor:
        for _ in tqdm(range(repeats), desc="Diabetes local prediction", unit="run"):
            _sync(device)
            start = time.perf_counter()
            for agent, mean, variance in executor.map(predict_one, range(m)):
                means[agent] = mean
                variances[agent] = variance
            _sync(device)
            times.append(time.perf_counter() - start)
    return torch.stack(means), torch.stack(variances), np.asarray(times)


def benchmark_poe(means: torch.Tensor, variances: torch.Tensor, repeats: int, desc):
    """Time plaintext PoE aggregation over the leading agent dimension."""
    times = []
    for _ in tqdm(range(repeats), desc=desc, unit="run"):
        start = time.perf_counter()
        precisions = variances.reciprocal()
        poe_variance = precisions.sum(dim=0).reciprocal()
        poe_mean = poe_variance * (precisions * means).sum(dim=0)
        if means.is_cuda:
            torch.cuda.synchronize(means.device)
        times.append(time.perf_counter() - start)
    return poe_mean, poe_variance, np.asarray(times)


def benchmark_private_consensus(
    means: torch.Tensor,
    variances: torch.Tensor,
    graph,
    config: dict,
    lz_values,
    repeats: int,
    device,
    desc: str,
):
    """Time batched consensus for each reciprocal scale in ``lz_values``.

    Return per-run times and the final states from the last repeat."""
    m = means.shape[0]
    weighted_means = m * means / variances
    precisions = m / variances
    combined = torch.stack((weighted_means, precisions), dim=-1)
    # Keep the packed state on its current device.  Moving CUDA predictions to
    # NumPy here would force a GPU -> CPU -> GPU round trip before consensus.
    packed = combined.reshape(m, -1).detach()
    times = np.zeros((repeats, len(lz_values)), dtype=np.float64)
    final_values = [None] * len(lz_values)

    for repeat in tqdm(range(repeats), desc=desc, unit="run"):
        for index, lz in enumerate(lz_values):
            run_config = dict(config)
            run_config["Lz"] = 1.0 / lz
            _sync(device)
            start = time.perf_counter()
            final_state = privacy_preserving_avg_consensus(
                packed,
                graph,
                run_config,
                run_config["T"],
                device,
                comm_delay=run_config["communication_latency"],
                return_history=False,
            )
            _sync(device)
            times[repeat, index] = time.perf_counter() - start
            final_values[index] = final_state.reshape(*means.shape, 2)
    return times, final_values


def consensus_mean_variance(final_value: torch.Tensor):
    """Recover mean and variance from weighted-mean and precision states."""
    denominator = final_value[..., 1]
    return final_value[..., 0] / denominator, denominator.reciprocal()


def proposed_owner_hyperparameters(models, likelihoods):
    """Extract each local GP's physical RBF parameters, grouped by output."""
    multi_output = isinstance(models[0], (list, tuple))
    output_count = len(models[0]) if multi_output else 1
    by_output = []
    for output in range(output_count):
        owner_values = []
        for owner in range(len(models)):
            model = models[owner][output] if multi_output else models[owner]
            likelihood = (
                likelihoods[owner][output] if multi_output else likelihoods[owner]
            )
            owner_values.append(
                {
                    "lengthscale": float(
                        model.covar_module.base_kernel.lengthscale.detach().cpu().item()
                    ),
                    "signal_variance": float(
                        model.covar_module.outputscale.detach().cpu().item()
                    ),
                    "noise_variance": float(likelihood.noise.detach().cpu().item()),
                }
            )
        by_output.append(owner_values)
    return by_output


def benchmark_he_poe(
    means: torch.Tensor,
    variances: torch.Tensor,
    repeats: int,
    communication_latency: float,
    workers: int = 4,
    communication_phases: int = 6,
    variance_floor: float = 1e-8,
    backend=None,
    desc: str = "Multiparty HE-PoE",
):
    """Aggregate exact local PoE statistics with Lattigo multiparty CKKS.

    Protocol time includes the backend call, final division, and modeled
    communication delay. Local GP prediction is timed by the caller."""
    means_np = means.detach().cpu().numpy().astype(np.float64)
    variances_np = variances.detach().cpu().numpy().astype(np.float64)
    if means_np.shape != variances_np.shape or means_np.ndim not in (2, 3):
        raise ValueError("means and variances must have matching (M,N[,K]) shapes")
    if variance_floor <= 0:
        raise ValueError("variance_floor must be positive")
    variances_np = np.maximum(variances_np, variance_floor)
    precisions = 1.0 / variances_np
    weighted_means = means_np * precisions
    parties = means_np.shape[0]
    statistic_shape = means_np.shape[1:]
    statistic_size = int(np.prod(statistic_shape))
    packed = np.concatenate(
        (weighted_means.reshape(parties, -1), precisions.reshape(parties, -1)),
        axis=1,
    )
    plain_sum = packed.sum(axis=0)
    plain_numerator = plain_sum[:statistic_size].reshape(statistic_shape)
    plain_precision = plain_sum[statistic_size:].reshape(statistic_shape)
    plaintext_mean = plain_numerator / plain_precision
    plaintext_variance = 1.0 / plain_precision

    if backend is None:
        configured_backend = os.environ.get("NAWAZ_LATTIGO_BACKEND")
        if configured_backend:
            backend = Path(configured_backend)
        else:
            name = "nawaz_lattigo.exe" if os.name == "nt" else "nawaz_lattigo"
            backend = Path(__file__).parent / "nawaz_sdsgp" / "lattigo_backend" / name
    latency_time = communication_phases * communication_latency
    protocol_times, he_times, division_times = [], [], []
    decrypted = metadata = secure_mean = secure_variance = None
    for _ in tqdm(range(repeats), desc=desc, unit="run"):
        start = time.perf_counter()
        decrypted, metadata = lattigo_secure_sum(packed, backend, workers=workers)
        he_time = time.perf_counter() - start
        start = time.perf_counter()
        numerator = decrypted[:statistic_size].reshape(statistic_shape)
        precision = decrypted[statistic_size:].reshape(statistic_shape)
        secure_mean = numerator / precision
        secure_variance = 1.0 / precision
        division_time = time.perf_counter() - start
        he_times.append(he_time)
        division_times.append(division_time)
        protocol_times.append(he_time + division_time + latency_time)

    expected_ciphertexts = int(np.ceil(packed.shape[1] / 2**15))
    if metadata.get("ciphertexts") != expected_ciphertexts:
        raise RuntimeError("Lattigo ciphertext-chunk count mismatch")
    # The collectively decrypted aggregate is delivered to every authorized
    # agent; each performs the same final PoE division locally.
    agent_predictions = np.repeat(secure_mean[None, ...], parties, axis=0)
    agent_variances = np.repeat(secure_variance[None, ...], parties, axis=0)
    if not np.allclose(agent_predictions, agent_predictions[0:1], atol=0, rtol=0):
        raise AssertionError("Authorized agents obtained different PoE results")
    return {
        "prediction": secure_mean,
        "variance": secure_variance,
        "agent_predictions": agent_predictions,
        "agent_variances": agent_variances,
        "plaintext_prediction": plaintext_mean,
        "plaintext_variance": plaintext_variance,
        "protocol_times": np.asarray(protocol_times),
        "he_times": np.asarray(he_times),
        "division_times": np.asarray(division_times),
        "empirical_latency_time": latency_time,
        "packed_values_per_agent": packed.shape[1],
        "ciphertexts_per_agent": expected_ciphertexts,
        "maximum_secure_sum_error": float(np.max(np.abs(decrypted - plain_sum))),
        "metadata": metadata,
    }
