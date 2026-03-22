"""
Evaluation module.

* Cole-Hopf exact solution for the Burgers equation
* L2 / Linf relative error computation
* Visualisation: heatmap comparison + time slices
"""

import os

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.integrate import quad

from config import Config


# -----------------------------------------------------------------------
# Exact solution via Cole-Hopf transform  (numerically stable)
# -----------------------------------------------------------------------
#
# For u(x,0) = -sin(pi*x), the Cole-Hopf solution is:
#
#   u(x,t) = -2*nu * phi_x / phi
#
# where  phi(x,t) = ∫ exp[ F(x,xi,t) ] dxi
#        F = cos(pi*xi)/(2*nu*pi)  -  (x-xi)²/(4*nu*t)
#
# The exponent can reach ~50 for small nu, causing exp() overflow.
# Fix: subtract the maximum of F before exponentiating (log-sum-exp trick),
# then the ratio phi_x/phi cancels the offset automatically.
# -----------------------------------------------------------------------

def _log_integrand_max(x: float, t: float, nu: float) -> float:
    """Peak of F(x, xi, t) w.r.t. xi — used for numerical stabilisation."""
    # F = cos(pi*xi)/(2*nu*pi) - (x-xi)^2/(4*nu*t)
    # No closed form, so we sample densely and take the max.
    xi = np.linspace(x - 4, x + 4, 2000)
    F  = np.cos(np.pi * xi) / (2.0 * nu * np.pi) - (x - xi) ** 2 / (4.0 * nu * t)
    return F.max()


def _stable_integrals(x: float, t: float, nu: float):
    """
    Return (phi, phi_x) using a log-space stabilised quadrature.

    phi   = ∫ exp(F - C) dxi   (C = max of F, cancels in ratio)
    phi_x = ∫ -(x-xi)/(2*nu*t) * exp(F - C) dxi
    """
    C = _log_integrand_max(x, t, nu)

    def f_phi(xi):
        F = np.cos(np.pi * xi) / (2.0 * nu * np.pi) - (x - xi) ** 2 / (4.0 * nu * t)
        return np.exp(F - C)

    def f_phi_x(xi):
        F = np.cos(np.pi * xi) / (2.0 * nu * np.pi) - (x - xi) ** 2 / (4.0 * nu * t)
        return -(x - xi) / (2.0 * nu * t) * np.exp(F - C)

    phi,   _ = quad(f_phi,   -np.inf, np.inf, limit=200, epsabs=1e-10, epsrel=1e-10)
    phi_x, _ = quad(f_phi_x, -np.inf, np.inf, limit=200, epsabs=1e-10, epsrel=1e-10)
    return phi, phi_x


def exact_solution(
    x_vec: np.ndarray,
    t_vec: np.ndarray,
    nu: float,
) -> np.ndarray:
    """
    Numerically stable exact Burgers solution via Cole-Hopf.

    Parameters
    ----------
    x_vec : 1-D array, shape (Nx,)
    t_vec : 1-D array, shape (Nt,)  — t=0 handled separately
    nu    : float

    Returns
    -------
    U_exact : 2-D array, shape (Nx, Nt)
    """
    Nx, Nt = len(x_vec), len(t_vec)
    U = np.zeros((Nx, Nt))

    total = Nx * sum(1 for t in t_vec if t > 0.0)
    done  = 0

    for j, t in enumerate(t_vec):
        if t == 0.0:
            U[:, j] = -np.sin(np.pi * x_vec)
        else:
            for i, x in enumerate(x_vec):
                phi, phi_x = _stable_integrals(x, t, nu)
                U[i, j]    = 2.0 * nu * phi_x / phi
                done += 1
                if done % 500 == 0:
                    print(f"  Cole-Hopf: {done}/{total} points", end="\r", flush=True)

    print(f"  Cole-Hopf: {total}/{total} points — done.          ")
    return U


# -----------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------

def relative_l2(u_pred: np.ndarray, u_true: np.ndarray) -> float:
    """Relative L2 error: ||u_pred - u_true||_2 / ||u_true||_2"""
    return np.linalg.norm(u_pred - u_true) / np.linalg.norm(u_true)


def relative_linf(u_pred: np.ndarray, u_true: np.ndarray) -> float:
    """Relative L-infinity error: max|u_pred - u_true| / max|u_true|"""
    return np.max(np.abs(u_pred - u_true)) / np.max(np.abs(u_true))


# -----------------------------------------------------------------------
# Prediction on a grid
# -----------------------------------------------------------------------

def predict_on_grid(
    model: nn.Module,
    x_vec: np.ndarray,
    t_vec: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """
    Evaluate the PINN on a (Nx x Nt) grid.

    Returns
    -------
    U_pred : shape (Nx, Nt)
    """
    X, T = np.meshgrid(x_vec, t_vec, indexing="ij")   # (Nx, Nt)
    x_flat = torch.tensor(X.ravel()[:, None], dtype=torch.float32, device=device)
    t_flat = torch.tensor(T.ravel()[:, None], dtype=torch.float32, device=device)

    model.eval()
    with torch.no_grad():
        u_flat = model(x_flat, t_flat).cpu().numpy().ravel()

    return u_flat.reshape(len(x_vec), len(t_vec))


# -----------------------------------------------------------------------
# Visualisation
# -----------------------------------------------------------------------

def plot_results(
    U_pred: np.ndarray,
    U_exact: np.ndarray,
    x_vec: np.ndarray,
    t_vec: np.ndarray,
    cfg: Config,
    save: bool = True,
) -> None:
    """
    Three-panel figure:
      (a) PINN prediction heatmap
      (b) Exact solution heatmap
      (c) Absolute error heatmap
    plus one figure with three time-slice comparisons.
    """
    os.makedirs(cfg.save_path, exist_ok=True)
    error = np.abs(U_pred - U_exact)

    # ---- Heatmaps --------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    extent = [t_vec[0], t_vec[-1], x_vec[0], x_vec[-1]]
    kw = dict(aspect="auto", origin="lower", extent=extent)

    vmin = min(U_pred.min(), U_exact.min())
    vmax = max(U_pred.max(), U_exact.max())

    im0 = axes[0].imshow(U_pred,  cmap="RdBu_r", vmin=vmin, vmax=vmax, **kw)
    axes[0].set_title("PINN prediction")

    im1 = axes[1].imshow(U_exact, cmap="RdBu_r", vmin=vmin, vmax=vmax, **kw)
    axes[1].set_title("Exact (Cole-Hopf)")

    im2 = axes[2].imshow(error,   cmap="hot_r", **kw)
    axes[2].set_title("Absolute error")

    for ax in axes:
        ax.set_xlabel("t")
        ax.set_ylabel("x")

    plt.colorbar(im0, ax=axes[0])
    plt.colorbar(im1, ax=axes[1])
    plt.colorbar(im2, ax=axes[2])

    plt.suptitle(
        f"Burgers PINN   ν={cfg.nu:.4f}   "
        f"Rel. L2={relative_l2(U_pred, U_exact):.2e}",
        fontsize=12,
    )
    plt.tight_layout()

    if save:
        path = os.path.join(cfg.save_path, "heatmaps.png")
        plt.savefig(path, dpi=150)
        print(f"Figure saved → {path}")
    plt.show()

    # ---- Time slices -------------------------------------------------------
    t_slices = [0.25, 0.50, 0.75]
    fig2, axes2 = plt.subplots(1, 3, figsize=(13, 4), sharey=True)

    for ax, t_target in zip(axes2, t_slices):
        j = np.argmin(np.abs(t_vec - t_target))
        ax.plot(x_vec, U_exact[:, j], "k-",  lw=2,   label="Exact")
        ax.plot(x_vec, U_pred[:, j],  "r--", lw=1.5, label="PINN")
        ax.set_title(f"t = {t_vec[j]:.2f}")
        ax.set_xlabel("x")
        ax.grid(alpha=0.3)
        ax.legend()

    axes2[0].set_ylabel("u(x, t)")
    plt.suptitle("Time slice comparison", fontsize=12)
    plt.tight_layout()

    if save:
        path = os.path.join(cfg.save_path, "slices.png")
        plt.savefig(path, dpi=150)
        print(f"Figure saved → {path}")
    plt.show()


def plot_training_history(history: list[dict], cfg: Config, save: bool = True) -> None:
    """
    Plot total loss and each component over training epochs.
    """
    import pandas as pd

    df = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Total loss
    for phase, grp in df.groupby("phase"):
        axes[0].semilogy(grp["epoch"], grp["loss"], label=phase)
    axes[0].set_title("Total loss")
    axes[0].set_xlabel("Epoch / iteration")
    axes[0].legend()
    axes[0].grid(which="both", alpha=0.3)

    # Components
    for col, label in [("loss_pde", "PDE"), ("loss_ic", "IC"), ("loss_bc", "BC")]:
        axes[1].semilogy(df["epoch"], df[col], label=label)
    axes[1].set_title("Loss components")
    axes[1].set_xlabel("Epoch / iteration")
    axes[1].legend()
    axes[1].grid(which="both", alpha=0.3)

    plt.tight_layout()

    if save:
        os.makedirs(cfg.save_path, exist_ok=True)
        path = os.path.join(cfg.save_path, "loss_history.png")
        plt.savefig(path, dpi=150)
        print(f"Figure saved → {path}")
    plt.show()
