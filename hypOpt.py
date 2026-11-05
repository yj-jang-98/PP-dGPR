"""Local exact GP models and consensus-based hyperparameter optimization."""

import torch
import gpytorch
from share import privacy_preserving_avg_consensus, prepare_graph_cache
import time
import html
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from IPython import get_ipython
    from IPython.display import HTML, display

    _in_notebook = (
        get_ipython() is not None
        and get_ipython().__class__.__name__ == "ZMQInteractiveShell"
    )
except ImportError:
    _in_notebook = False


if _in_notebook:

    class _NotebookProgress:
        """A dependency-free, in-place notebook progress display."""

        def __init__(
            self,
            iterable=None,
            total=None,
            desc="",
            unit="it",
            disable=False,
            **kwargs,
        ):
            self.iterable = iterable
            self.total = total if total is not None else len(iterable)
            self.desc = desc
            self.unit = unit
            self.disable = disable
            self.n = 0
            self.postfix = {}
            self.start_time = time.perf_counter()
            self.lock = threading.Lock()
            self.handle = None
            if not disable:
                self.handle = display(HTML(self._html()), display_id=True)

        def _html(self):
            elapsed = max(time.perf_counter() - self.start_time, 1e-9)
            percent = 100 * self.n / self.total if self.total else 0
            rate = self.n / elapsed
            postfix = ", ".join(f"{k}={v}" for k, v in self.postfix.items())
            suffix = f", {postfix}" if postfix else ""
            return (
                '<div style="font-family:monospace; margin:2px 0">'
                f'<span style="display:inline-block; min-width:180px">'
                f"{html.escape(self.desc)}:</span> "
                f'<progress value="{self.n}" max="{self.total}" '
                'style="width:320px"></progress> '
                f"{percent:3.0f}% | {self.n}/{self.total} "
                f"[{elapsed:0.0f}s, {rate:0.2f} {html.escape(self.unit)}/s]"
                f"{html.escape(suffix)}</div>"
            )

        def refresh(self):
            if not self.disable and self.handle is not None:
                self.handle.update(HTML(self._html()))

        def update(self, amount=1):
            with self.lock:
                self.n += amount
                self.refresh()

        def set_postfix(self, refresh=True, **kwargs):
            with self.lock:
                self.postfix = kwargs
                if refresh:
                    self.refresh()

        def set_postfix_str(self, text, refresh=True):
            with self.lock:
                self.postfix = {"status": text}
                if refresh:
                    self.refresh()

        def close(self):
            self.refresh()

        def __iter__(self):
            for value in self.iterable:
                yield value
                self.update(1)

    tqdm = _NotebookProgress
else:
    try:
        from tqdm import tqdm
    except ImportError:

        def tqdm(iterable=None, **kwargs):
            return iterable


def constrained_parameter_gradient(raw_parameter, constraint):
    """Convert a raw GPyTorch gradient to the physical parameter gradient."""
    raw_value = raw_parameter.detach().clone().requires_grad_(True)
    physical_value = constraint.transform(raw_value)
    transform_derivative = torch.autograd.grad(
        physical_value,
        raw_value,
        grad_outputs=torch.ones_like(physical_value),
    )[0]
    return raw_parameter.grad / transform_derivative


class GPModel(gpytorch.models.ExactGP):
    """Exact GP with a constant mean and a scaled isotropic RBF kernel."""

    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def hyperparameter_optimization(
    X_parts,
    Y_parts,
    G,
    config,
    scales,
    device,
    show_progress=True,
    num_workers=4,
):
    """Optimize one local GP per agent and output dimension.

    Return models and likelihoods indexed by [agent][output], and parameter
    histories and summed log marginal likelihoods indexed by [output].
    Independent outputs can be processed by separate workers."""

    M = config["M"]

    k = config["k"]

    models_all = [[None] * k for _ in range(M)]
    likelihoods_all = [[None] * k for _ in range(M)]
    history_all = [None] * k
    loss_sum_all = [None] * k

    # Create every widget on the main notebook thread. The worker threads
    # update these existing widgets instead of printing new terminal lines.
    output_progress_bars = (
        [
            tqdm(
                total=config["Thyp"],
                desc=f"SARCOS output {d + 1}/{k}",
                unit="iteration",
                leave=True,
                position=d,
            )
            for d in range(k)
        ]
        if show_progress
        else [None] * k
    )

    def optimize_output(d):
        # Extract the d-th output dimension for each agent
        Y_d_parts = [Y_parts[i][:, d].unsqueeze(-1) for i in range(M)]

        result = hyperparameter_optimization_1D(
            X_parts,
            Y_d_parts,
            G,
            config,
            scales,
            device,
            progress_desc=f"SARCOS output {d + 1}/{k}",
            show_progress=show_progress,
            progress_position=d,
            external_progress_bar=output_progress_bars[d],
        )
        return d, result

    workers = max(1, min(int(num_workers), k))
    previous_num_threads = torch.get_num_threads()

    # Build the read-only topology cache before worker threads access it.
    prepare_graph_cache(G)

    try:
        if workers > 1:
            # Prevent each output worker from starting its own large pool of
            # intra-op threads and oversubscribing the CPU.
            torch.set_num_threads(1)
            with ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix="sarcos-output",
            ) as executor:
                futures = [executor.submit(optimize_output, d) for d in range(k)]
                completed_results = [
                    future.result() for future in as_completed(futures)
                ]
        else:
            completed_results = [optimize_output(d) for d in range(k)]
    finally:
        torch.set_num_threads(previous_num_threads)
        for progress_bar in output_progress_bars:
            if progress_bar is not None:
                progress_bar.close()

    # Futures complete out of order; store each result by output index.
    for d, result in completed_results:
        models_d, likelihoods_d, history_d, loss_sum_d, elapsed = result
        print(f"[Output dimension {d + 1}] elapsed time: {elapsed:.2f}s")
        for i in range(M):
            models_all[i][d] = models_d[i]
            likelihoods_all[i][d] = likelihoods_d[i]
        history_all[d] = history_d
        loss_sum_all[d] = loss_sum_d

    return models_all, likelihoods_all, history_all, loss_sum_all


def hyperparameter_optimization_1D(
    X,
    Y,
    G,
    config,
    scales,
    device,
    progress_desc="Hyperparameter optimization",
    show_progress=True,
    progress_position=0,
    external_progress_bar=None,
):
    """Alternate one secure consensus step with local gradient ascent.

    ``scales`` gives the initialization centers for lengthscale, signal
    variance, and noise variance; each initial value varies by at most 5.
    ``X`` and ``Y`` contain one training partition per agent.

    Return local models, likelihoods, parameter histories, summed log marginal
    likelihoods, and elapsed training time."""

    M = config["M"]

    Thyp = config["Thyp"]

    eta = config["eta"]

    decay = config["decay"]

    X = [x.to(device) for x in X]
    Y = [y.to(device) for y in Y]
    Y_flat = [y.view(-1) for y in Y]

    # Initialize lengthscale, signal variance, and noise variance.
    range_half = 5.0
    scales_tensor = torch.tensor(scales, dtype=torch.float32, device=device)
    theta_estimates = scales_tensor.unsqueeze(0).repeat(M, 1)
    theta_estimates += torch.empty((M, len(scales)), device=device).uniform_(
        -range_half, range_half
    )

    history = []
    loss_sum = []

    models = []
    likelihoods = []
    mlls = []

    # Create GP Models with the initial hyperparameters
    for i in range(M):
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
        model = GPModel(X[i], Y_flat[i], likelihood).to(device)
        models.append(model)
        likelihoods.append(likelihood)
        mlls.append(gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model))

    start = time.perf_counter()

    if external_progress_bar is None:
        iteration_bar = tqdm(
            range(Thyp),
            desc=progress_desc,
            unit="iteration",
            leave=True,
            disable=not show_progress,
            position=progress_position,
        )
        iteration_range = iteration_bar
        update_manually = False
    else:
        iteration_bar = external_progress_bar
        iteration_range = range(Thyp)
        update_manually = True

    for _ in iteration_range:
        loss_sum_tmp = torch.zeros((), device=device)

        # Equation (25a): one secure consensus step.

        consensus_result = privacy_preserving_avg_consensus(
            theta_estimates.detach(),
            G,
            config,
            1,
            device,
            return_history=False,
        ).to(dtype=torch.float32)

        # Equation (25b): local gradient ascent.

        updated_estimates = torch.empty_like(theta_estimates)
        for i in range(M):

            model = models[i]
            likelihood = likelihoods[i]

            model.train()
            likelihood.train()

            with torch.no_grad():
                model.covar_module.base_kernel.raw_lengthscale.copy_(
                    model.covar_module.base_kernel.raw_lengthscale_constraint.inverse_transform(
                        consensus_result[i][0].detach()
                    )
                )
                model.covar_module.raw_outputscale.copy_(
                    model.covar_module.raw_outputscale_constraint.inverse_transform(
                        consensus_result[i][1].detach()
                    )
                )
                likelihood.noise_covar.raw_noise.copy_(
                    likelihood.noise_covar.raw_noise_constraint.inverse_transform(
                        consensus_result[i][2].detach()
                    )
                )

            model.zero_grad()
            likelihood.zero_grad()

            # Compute local log marginal likelihood
            output = model(X[i])
            loss = mlls[i](output, Y_flat[i])
            loss_sum_tmp += loss.detach()
            loss.backward()

            # Convert gradients with respect to GPyTorch's unconstrained raw
            # parameters into gradients with respect to the physical positive
            # hyperparameters used in theta_estimates.
            grad_l = constrained_parameter_gradient(
                model.covar_module.base_kernel.raw_lengthscale,
                model.covar_module.base_kernel.raw_lengthscale_constraint,
            ).reshape(())
            grad_f = constrained_parameter_gradient(
                model.covar_module.raw_outputscale,
                model.covar_module.raw_outputscale_constraint,
            ).reshape(())
            grad_n = constrained_parameter_gradient(
                likelihood.noise_covar.raw_noise,
                likelihood.noise_covar.raw_noise_constraint,
            ).reshape(())

            grad = torch.stack([grad_l, grad_f, grad_n])
            updated_estimates[i] = consensus_result[i] + eta * grad

        theta_estimates = updated_estimates

        history.append(theta_estimates.detach().cpu().numpy())
        # Transfer one scalar per iteration instead of synchronizing the GPU
        # separately for every agent and hyperparameter.
        loss_sum.append(loss_sum_tmp.item())
        if hasattr(iteration_bar, "set_postfix"):
            iteration_bar.set_postfix(
                eta=f"{eta:.3e}",
                log_marginal_likelihood=f"{loss_sum[-1]:.3f}",
                refresh=False,
            )
        if update_manually:
            iteration_bar.update(1)

        # Apply decay after the update so the first step uses eta = 0.1.
        eta *= decay

    end = time.perf_counter()
    elapsed = end - start

    models = [model.eval() for model in models]
    likelihoods = [likelihood.eval() for likelihood in likelihoods]

    return models, likelihoods, history, loss_sum, elapsed
