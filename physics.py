"""
Physics module: PDE residual, causal PINN loss, and training data generation.

Burgers equation:
    u_t + u * u_x = nu * u_xx
    x in [-1, 1],  t in [0, 1]
    IC: u(x, 0) = -sin(pi * x)
    BC: u(-1, t) = u(1, t) = 0

Causal weighting (Wang et al. 2022, "Respecting causality is all you need"):
    Instead of weighting all collocation points equally, we weight each
    point by how well the PDE is satisfied at all earlier times.
    Points at large t get a small weight until earlier times are solved.

    w(t_k) = exp( -epsilon * sum_{t_j < t_k} L_j )

    where L_j is the mean squared residual in the j-th time bin.
    epsilon controls how strictly causality is enforced:
      - large epsilon => very strict (nearly sequential)
      - small epsilon => closer to standard PINN
"""

import torch
import torch.nn as nn
from torch import Tensor
from config import Config


# -----------------------------------------------------------------------
# Automatic differentiation helper
# -----------------------------------------------------------------------

def grad(output: Tensor, input_: Tensor) -> Tensor:
    """
    Compute d(output)/d(input_) via autograd.
    Both output and input_ must have requires_grad=True in the graph.
    """
    return torch.autograd.grad(
        output,
        input_,
        grad_outputs=torch.ones_like(output),
        create_graph=True,
        retain_graph=True,
    )[0]


# -----------------------------------------------------------------------
# Causal weights
# -----------------------------------------------------------------------

def causal_weights(
    t_f: Tensor,
    residual_sq: Tensor,
    n_bins: int,
    epsilon: float,
    t_min: float,
    t_max: float,
) -> Tensor:
    """
    Compute per-point causal weights w(t_i) in [0, 1].

    The domain [t_min, t_max] is divided into n_bins equal time bins.
    For each bin k, the weight is:

        w_k = exp( -epsilon * cumsum_{j<k}( mean_residual_j ) )

    so early bins get weight ~1 and late bins get weight ~0 until
    the early residuals are small.

    Parameters
    ----------
    t_f         : (N, 1) collocation times
    residual_sq : (N, 1) squared PDE residuals, detached
    n_bins      : number of time bins
    epsilon     : causality strength (typical: 1.0 to 100.0)
    t_min, t_max: domain bounds

    Returns
    -------
    weights : (N, 1) tensor in (0, 1], same device as t_f
    """
    device = t_f.device
    dt     = (t_max - t_min) / n_bins

    # bin index for each point: 0 .. n_bins-1
    bin_idx = ((t_f - t_min) / dt).long().clamp(0, n_bins - 1)  # (N, 1)

    # mean residual per bin
    bin_losses = torch.zeros(n_bins, device=device)
    bin_counts = torch.zeros(n_bins, device=device)
    flat_idx   = bin_idx.squeeze(1)
    flat_res   = residual_sq.squeeze(1).detach()

    bin_losses.scatter_add_(0, flat_idx, flat_res)
    bin_counts.scatter_add_(0, flat_idx, torch.ones_like(flat_res))
    bin_counts = bin_counts.clamp(min=1.0)
    bin_losses = bin_losses / bin_counts          # mean per bin

    # cumulative sum shifted by 1 (exclude current bin)
    # w_k = exp(-epsilon * sum_{j=0}^{k-1} L_j)
    cum_losses = torch.cumsum(bin_losses, dim=0)
    cum_losses = torch.cat([torch.zeros(1, device=device), cum_losses[:-1]])
    bin_weights = torch.exp(-epsilon * cum_losses)  # (n_bins,)

    # assign per-point weights
    weights = bin_weights[flat_idx].unsqueeze(1)  # (N, 1)
    return weights


# -----------------------------------------------------------------------
# PINN loss with causal weighting
# -----------------------------------------------------------------------

def pinn_loss(
    model: nn.Module,
    batch: tuple,
    cfg: Config,
) -> tuple:
    """
    Compute the three-term PINN loss with causal weighting on L_pde.

    The PDE residual is weighted by w(t) = exp(-epsilon * cumulative_loss)
    so that the network is forced to satisfy early times before late times.

    Parameters
    ----------
    model : PINN
    batch : (x_f, t_f, x_b, t_b, x_i, t_i)
    cfg   : Config

    Returns
    -------
    total_loss, loss_pde, loss_ic, loss_bc, mean_weight
    """
    x_f, t_f, x_b, t_b, x_i, t_i = batch

    # --- PDE residual ------------------------------------------------------
    x_f = x_f.requires_grad_(True)
    t_f = t_f.requires_grad_(True)

    u    = model(x_f, t_f)
    u_t  = grad(u, t_f)
    u_x  = grad(u, x_f)
    u_xx = grad(u_x, x_f)

    residual    = u_t + u * u_x - cfg.nu * u_xx
    residual_sq = residual ** 2                    # (N, 1)

    # --- Causal weighting --------------------------------------------------
    if cfg.causal_epsilon > 0.0:
        w = causal_weights(
            t_f.detach(), residual_sq,
            n_bins=cfg.causal_bins,
            epsilon=cfg.causal_epsilon,
            t_min=cfg.t_min,
            t_max=cfg.t_max,
        )
        loss_pde   = torch.mean(w * residual_sq)
        mean_weight = w.mean().item()
    else:
        loss_pde   = torch.mean(residual_sq)
        mean_weight = 1.0

    # --- Initial condition  u(x, 0) = -sin(pi * x) -------------------------
    u_ic      = model(x_i, t_i)
    u_ic_true = -torch.sin(torch.pi * x_i)
    loss_ic   = torch.mean((u_ic - u_ic_true) ** 2)

    # --- Boundary conditions  u(+/-1, t) = 0 --------------------------------
    loss_bc = torch.mean(model(x_b, t_b) ** 2)

    # --- Weighted sum --------------------------------------------------------
    total = (
        cfg.lambda_pde * loss_pde
        + cfg.lambda_ic  * loss_ic
        + cfg.lambda_bc  * loss_bc
    )

    return total, loss_pde, loss_ic, loss_bc, mean_weight


# -----------------------------------------------------------------------
# Data generation
# -----------------------------------------------------------------------

def generate_data(cfg: Config, device: torch.device) -> tuple:
    """
    Sample collocation, boundary, and initial-condition points.

    Collocation points are a mix of:
      - uniform random draws over the full domain  (75%)
      - adaptive points concentrated near the shock (25%)

    Returns
    -------
    (x_f, t_f, x_b, t_b, x_i, t_i) -- all on device, shape (N, 1)
    """
    # -- Collocation points : uniform background ---------------------------
    N_uniform = int(cfg.N_f * 0.75)
    x_uniform = torch.rand(N_uniform, 1) * (cfg.x_max - cfg.x_min) + cfg.x_min
    t_uniform = torch.rand(N_uniform, 1) * (cfg.t_max - cfg.t_min) + cfg.t_min

    # -- Collocation points : adaptive near the shock ----------------------
    N_shock = cfg.N_f - N_uniform
    x_shock = torch.randn(N_shock, 1) * 0.15
    x_shock = x_shock.clamp(cfg.x_min, cfg.x_max)
    t_shock = torch.rand(N_shock, 1) * 0.6 + 0.4

    x_f = torch.cat([x_uniform, x_shock], dim=0)
    t_f = torch.cat([t_uniform, t_shock], dim=0)

    # -- Boundary points  x = +/-1 -----------------------------------------
    t_b = torch.rand(cfg.N_b, 1) * (cfg.t_max - cfg.t_min) + cfg.t_min
    x_b = torch.cat([
        torch.full((cfg.N_b // 2, 1), cfg.x_min),
        torch.full((cfg.N_b // 2, 1), cfg.x_max),
    ], dim=0)

    # -- Initial condition points  t = 0 -----------------------------------
    x_i = torch.rand(cfg.N_i, 1) * (cfg.x_max - cfg.x_min) + cfg.x_min
    t_i = torch.zeros(cfg.N_i, 1)

    tensors = (x_f, t_f, x_b, t_b, x_i, t_i)
    return tuple(t.to(device) for t in tensors)
