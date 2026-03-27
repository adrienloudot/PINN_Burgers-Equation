"""
Module d'évaluation.

- Solution exacte de Cole-Hopf pour l'équation de Burgers
- Calcul de l'erreur relative L2 / Linf
- Visualisation : comparaison sous forme de carte thermique + tranches temporelles

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
# Solution exacte via Cole-Hopf (numériquement stable)
# -----------------------------------------------------------------------
#
# Pour u(x,0) = -sin(π*x), la solution de Cole-Hopf est :
#
#   u(x,t) = -2*nu * φ_x / φ
#
# où  φ(x,t) = ∫ exp[ F(x,xi,t) ] dxi
#        F = cos(pi*xi)/(2*nu*pi)  -  (x-xi)²/(4*nu*t)
#
# L'exposant peut atteindre environ 50 pour de petites valeurs de nu, provoquant un débordement de la fonction exp().
# Solution : soustraire la valeur maximale de F avant l'exponentiation (astuce log-somme-exp),
# puis le rapport phi_x/phi annule automatiquement le décalage.
#
# -----------------------------------------------------------------------

def _log_integrand_max(x: float, t: float, nu: float) -> float:
    # Maximum de F(x, xi, t) par rapport à xi — utilisé pour la stabilisation numérique. 
    # F = cos(π*xi)/(2*ν*π) - (x-xi)²/(4*ν*t)
    # Pas de formule exacte, on procède donc à un échantillonnage dense et on prend la valeur maximale.
    xi = np.linspace(x - 4, x + 4, 2000)
    F  = np.cos(np.pi * xi) / (2.0 * nu * np.pi) - (x - xi) ** 2 / (4.0 * nu * t)
    return F.max()


def _stable_integrals(x: float, t: float, nu: float):
    """
    Calcule (phi, phi_x) à l'aide d'une quadrature stabilisée en espace logarithmique.

    phi   = ∫ exp(F - C) dxi   (C = max de F, s'annule dans le rapport)
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
    Solution exacte et numériquement stable de Burgers calculée via Cole-Hopf.

    Paramètres
    ----------
    x_vec : tableau 1D, forme (Nx,)
    t_vec : tableau 1D, forme (Nt,)  — le cas t = 0 est traité séparément
    nu    : float

    Retourne
    -------
    U_exact : tableau 2D, forme (Nx, Nt)
    
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
# Mesures de sortie
# -----------------------------------------------------------------------

def relative_l2(u_pred: np.ndarray, u_true: np.ndarray) -> float:
    """Relative L2 error: ||u_pred - u_true||_2 / ||u_true||_2"""
    return np.linalg.norm(u_pred - u_true) / np.linalg.norm(u_true)


def relative_linf(u_pred: np.ndarray, u_true: np.ndarray) -> float:
    """Relative L-infinity error: max|u_pred - u_true| / max|u_true|"""
    return np.max(np.abs(u_pred - u_true)) / np.max(np.abs(u_true))


# -----------------------------------------------------------------------
# Prédictions sur la grille
# -----------------------------------------------------------------------

def predict_on_grid(
    model:  nn.Module,
    x_vec:  np.ndarray,
    t_vec:  np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """
    Evalue la solution approchée grâce au PINN sur une grille (Nx,Nt)

    Returns
    -------
    U_pred : taille (Nx, Nt)

    """
    X, T   = np.meshgrid(x_vec, t_vec, indexing="ij")   # (Nx, Nt)
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
    U_pred:  np.ndarray,
    U_exact: np.ndarray,
    x_vec:   np.ndarray,
    t_vec:   np.ndarray,
    cfg:     Config,
    save:    bool = True,
) -> None:
    """
    Figure en trois volets :
      (a) Carte thermique des prévisions PINN
      (b) Carte thermique de la solution exacte
      (c) Carte thermique de l'erreur absolue
    ainsi qu'une figure présentant trois comparaisons sur des tranches temporelles.
    
    """
    os.makedirs(cfg.save_path, exist_ok=True)
    error = np.abs(U_pred - U_exact)

    # -- Heatmaps --
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

    # -- Time slices --
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
    Affiche au cours de l'entraînement : 
        - loss totale
        - loss_PDE
        - loss_CI
        - loss_BC

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
