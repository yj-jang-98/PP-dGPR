import torch
import gpytorch
from share import privacy_preserving_avg_consensus
import numpy as np
import time


# --- Define GP model
class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# --- Section 4.2 Hyperparameter optimization
def hyperparameter_optimization(X_parts, Y_parts, G, config, scales, device):
    """
    Runs gradient based consensus algorithm for hyperparameter optimization for each output dimension in parallel.
    
    Parameters
    ----------
        X_parts : Training input data partitions for M agents
        Y_parts : Training output data partitions for M agents
        G       : Graph
        config  : Config parameters
            M: num of agents
            k: output dimension
        
    Returns
    -------
        models_all      : Each element containing k GP models
        likelihoods_all : Each element containing k likelihoods 
        history_all     : Each element containing k history of hyperparameters obtained during optimization 
        loss_sum_all    : Each element containing k LMM
    """

    # --- Parameters
    # Num of agents
    M = config["M"]
    # Output dimension 
    k = config["k"] 

    models_all = [[None]*k for _ in range(M)]
    likelihoods_all = [[None]*k for _ in range(M)]
    history_all = [None] * k 
    loss_sum_all = [None] * k 

    
    for d in range(k):
        # Extract the d-th output dimension for each agent
        Y_d_parts = [Y_parts[i][:, d].unsqueeze(-1) for i in range(M)]

        # Run 1D optimization for output dimension d
        models_d, likelihoods_d, history_d, loss_sum_d, elapsed = hyperparameter_optimization_1D(
            X_parts, Y_d_parts, G, config, scales, device
        )

        print(f"[Output dimension {d+1}] elapsed time: {elapsed:.2f}s")

        # Store results
        for i in range(M):
            models_all[i][d] = models_d[i]
            likelihoods_all[i][d] = likelihoods_d[i]
        history_all[d] = history_d
        loss_sum_all[d] = loss_sum_d

    return models_all, likelihoods_all, history_all, loss_sum_all


def hyperparameter_optimization_1D(X, Y, G, config, scales, device):
    """
    Runs Section 4.2

    Parameters
    ----------
        X      : Training input data partitions for M agents
        Y      : Training output data partitions for M agents - must be 1D   
        G      : Graph
        config : Config parameters
            M: num of agents
            k: output dimension
    
    Returns
    -------
        models      : GP model for each agent
        likelihoods : Likelihood for each agent
        history     : History of hyperparameters obtained during optimization for each agent
        loss_sum    : LMM for each agent
        elapsed     : Elapsed total time
    """
    # --- Parameters
    # Num of agents
    M = config["M"]
    # Num of iterations
    Thyp = config["Thyp"]
    # Initial learning rate
    eta = config["eta"]
    # Decay rate
    decay = config["decay"]

    X = [x.to(device) for x in X]
    Y = [y.to(device) for y in Y]
    Y_flat = [y.view(-1) for y in Y]

    # --- Initialize hyperparameters = [length scale, signal variance, noise varaince]
    range_half = 5.0
    scales_tensor = torch.tensor(scales, dtype=torch.float32, device=device)
    theta_estimates = scales_tensor.unsqueeze(0).repeat(M, 1)
    theta_estimates += torch.empty((M, len(scales)), device=device).uniform_(-range_half, range_half)

    history = []
    loss_sum = []

    models = []
    likelihoods = []
    mlls = []


    # --- Create GP Models with the initial hyperparameters
    for i in range(M):
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
        model = GPModel(X[i], Y_flat[i], likelihood).to(device)
        models.append(model)
        likelihoods.append(likelihood)
        mlls.append(gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model))
  
    start = time.perf_counter()

    # --- Main loop
    for _ in range(Thyp):
        # Decay learning rate
        eta *= decay 
        intermediates = torch.empty_like(theta_estimates)
        loss_sum_tmp = 0

        # --- Eq. (24a)
        # Take one step of local gradient ascent
        for i in range(M):
            
            model = models[i]
            likelihood = likelihoods[i]

            model.train()
            likelihood.train()

            # Current hyperparameter estimate
            with torch.no_grad():
                model.covar_module.base_kernel.raw_lengthscale.copy_(
                    model.covar_module.base_kernel.raw_lengthscale_constraint.inverse_transform(theta_estimates[i][0].detach())
                )
                model.covar_module.raw_outputscale.copy_(
                    model.covar_module.raw_outputscale_constraint.inverse_transform(theta_estimates[i][1].detach())
                )
                likelihood.noise_covar.raw_noise.copy_(
                    likelihood.noise_covar.raw_noise_constraint.inverse_transform(theta_estimates[i][2].detach())
                )

            model.zero_grad()
            likelihood.zero_grad()

            # Compute local log marginal likelihood 
            output = model(X[i])
            loss = mlls[i](output, Y_flat[i])
            loss_sum_tmp += loss.item()
            loss.backward()

            # Compute local gradients for each hyperparameter 
            grad_l = model.covar_module.base_kernel.raw_lengthscale.grad.item()
            grad_f = model.covar_module.raw_outputscale.grad.item()
            grad_n = likelihood.raw_noise.grad.item()
 
            # Take a local gradient step
            grad = torch.tensor([grad_l, grad_f, grad_n], device=device)
            intermediates[i] = theta_estimates[i] + eta * grad


        # --- Eq. (24b)
        # Take one step of secure distributed average consensus
        z_ini = intermediates.detach().cpu().numpy()
        z_history = privacy_preserving_avg_consensus(z_ini, G, config, 1, device) 
        consensus_result = torch.tensor(z_history[-1], dtype=torch.float32, device=device)

        # Update hyperparameter estimates
        theta_estimates = consensus_result

        history.append(theta_estimates.detach().cpu().numpy())
        loss_sum.append(loss_sum_tmp)

    end = time.perf_counter()   
    elapsed = (end - start) 

    
    models = [model.eval() for model in models]
    likelihoods = [likelihood.eval() for likelihood in likelihoods]

    return models, likelihoods, history, loss_sum, elapsed
