"""
Physics module: PDE residual, PINN loss, and training data generation.

Burgers equation:
    u_t + u * u_x = nu * u_xx
    x in [-1, 1],  t in [0, 1]
    IC: u(x, 0) = -sin(pi * x)
    BC: u(-1, t) = u(1, t) = 0
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
    Both `output` and `input_` must have requires_grad=True in the graph.
    """
    return torch.autograd.grad(
        output,
        input_,
        grad_outputs=torch.ones_like(output),
        create_graph=True,
        retain_graph=True,
    )[0]


# -----------------------------------------------------------------------
# PINN loss
# -----------------------------------------------------------------------

def pinn_loss(
    model: nn.Module,
    batch: tuple,
    cfg: Config,
) -> tuple:
    """
    Compute the three-term PINN loss.

    Parameters
    ----------
    model : PINN
    batch : (x_f, t_f, x_b, t_b, x_i, t_i)
        Collocation, boundary, and initial-condition points.
        All tensors must already be on the correct device.
    cfg   : Config

    Returns
    -------
    total_loss, loss_pde, loss_ic, loss_bc — all scalar Tensors
    """
    x_f, t_f, x_b, t_b, x_i, t_i = batch

    # --- PDE residual ------------------------------------------------------
    # requires_grad is set HERE (not in generate_data) to keep data generation
    # device-agnostic and avoid accidentally leaking graphs between epochs.
    x_f = x_f.requires_grad_(True)
    t_f = t_f.requires_grad_(True)

    u      = model(x_f, t_f)
    u_t    = grad(u, t_f)
    u_x    = grad(u, x_f)
    u_xx   = grad(u_x, x_f)

    residual = u_t + u * u_x - cfg.nu * u_xx
    loss_pde = torch.mean(residual ** 2)

    # --- Initial condition  u(x, 0) = -sin(pi * x) -------------------------
    u_ic      = model(x_i, t_i)
    u_ic_true = -torch.sin(torch.pi * x_i)
    loss_ic   = torch.mean((u_ic - u_ic_true) ** 2)

    # --- Boundary conditions  u(+/-1, t) = 0 ----------------------------------
    loss_bc = torch.mean(model(x_b, t_b) ** 2)

    # --- Weighted sum --------------------------------------------------------
    total = (
        cfg.lambda_pde * loss_pde
        + cfg.lambda_ic  * loss_ic
        + cfg.lambda_bc  * loss_bc
    )

    return total, loss_pde, loss_ic, loss_bc


# -----------------------------------------------------------------------
# Data generation
# -----------------------------------------------------------------------

def generate_data(cfg: Config, device: torch.device) -> tuple:
    """
    Sample collocation, boundary, and initial-condition points.

    Collocation points are a mix of:
      - uniform random draws over the full domain  (75%)
      - adaptive points concentrated near the shock (25%)
        The Burgers shock forms around x~0, t in [0.5, 1] for this IC.
        Oversampling this region helps the PDE residual drop there.

    Returns
    -------
    (x_f, t_f, x_b, t_b, x_i, t_i) -- all on device, shape (N, 1)
    """
    # -- Collocation points : uniform background ---------------------------
    N_uniform = int(cfg.N_f * 0.75)
    x_uniform = torch.rand(N_uniform, 1) * (cfg.x_max - cfg.x_min) + cfg.x_min
    t_uniform = torch.rand(N_uniform, 1) * (cfg.t_max - cfg.t_min) + cfg.t_min

    # -- Collocation points : adaptive near the shock ----------------------
    # Shock location: x~0 (Gaussian spread sigma=0.15), t in [0.4, 1.0]
    N_shock = cfg.N_f - N_uniform
    x_shock = torch.randn(N_shock, 1) * 0.15          # centered on x=0
    x_shock = x_shock.clamp(cfg.x_min, cfg.x_max)
    t_shock = torch.rand(N_shock, 1) * 0.6 + 0.4      # t in [0.4, 1.0]

    x_f = torch.cat([x_uniform, x_shock], dim=0)
    t_f = torch.cat([t_uniform, t_shock], dim=0)

    # -- Boundary points  x = +/-1 -------------------------------------------
    t_b = torch.rand(cfg.N_b, 1) * (cfg.t_max - cfg.t_min) + cfg.t_min
    x_b = torch.cat([
        torch.full((cfg.N_b // 2, 1), cfg.x_min),
        torch.full((cfg.N_b // 2, 1), cfg.x_max),
    ], dim=0)

    # -- Initial condition points  t = 0 ------------------------------------
    x_i = torch.rand(cfg.N_i, 1) * (cfg.x_max - cfg.x_min) + cfg.x_min
    t_i = torch.zeros(cfg.N_i, 1)

    tensors = (x_f, t_f, x_b, t_b, x_i, t_i)
    return tuple(t.to(device) for t in tensors)
