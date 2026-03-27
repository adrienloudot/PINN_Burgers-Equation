"""
Module de physique : résidu d'équation différentielle partielle, perte PINN causale et génération de données d'apprentissage.

Équation de Burgers :
    u_t + u * u_x = nu * u_xx
    x ∈ [-1, 1],  t ∈ [0, 1]
    Condition aux limites : u(x, 0) = -sin(π * x)
    Condition aux limites : u(-1, t) = u(1, t) = 0

Pondération causale (Wang et al. 2022, « Respecting causality is all you need ») :
    Au lieu de pondérer tous les points de collocation de manière égale, nous pondérons chaque
    point en fonction du degré de satisfaction de l'équation différentielle partielle (EDP) à tous les instants antérieurs.
    Les points situés à des instants t élevés reçoivent une faible pondération jusqu'à ce que les instants antérieurs soient résolus.

    w(t_k) = exp( -epsilon * sum_{t_j < t_k} L_j )

    où L_j est la moyenne quadratique des résidus dans le j-ième intervalle de temps.
    epsilon contrôle le degré de rigueur avec lequel la causalité est appliquée :
      - epsilon élevé => très strict (presque séquentiel)
      - epsilon faible => plus proche du PINN standard

"""

import torch
import torch.nn as nn
from torch import Tensor
from config import Config


# -----------------------------------------------------------------------
# Outil de différenciation automatique
# -----------------------------------------------------------------------

def grad(output: Tensor, input_: Tensor) -> Tensor:
    """
    Calcule de d(sortie)/d(entrée) à l'aide d'autograd.
    La sortie et l'entrée doivent toutes deux avoir requires_grad = True dans le graphe.
    
    """
    return torch.autograd.grad(
        output,
        input_,
        grad_outputs = torch.ones_like(output),
        create_graph = True,
        retain_graph = True,
    )[0]


# -----------------------------------------------------------------------
# Causal weights
# -----------------------------------------------------------------------

def causal_weights(
    t_f:         Tensor,
    residual_sq: Tensor,
    n_bins:      int,
    epsilon:     float,
    t_min:       float,
    t_max:       float,
) -> Tensor:
    """
    Calculer les poids causaux par point w(t_i) dans l'intervalle [0, 1].

    L'intervalle [t_min, t_max] est divisé en n_bins tranches de temps égales.
    Pour chaque tranche k, le poids est :

        w_k = exp( -epsilon * cumsum_{j<k}( mean_residual_j ) )

    ainsi, les intervalles précoces reçoivent un poids ~1 et les intervalles tardifs un poids ~0 jusqu’à ce que
    les résidus précoces soient faibles.

    Paramètres
    ----------
    t_f          : (N, 1) temps de colocalisation
    residual_sq  : (N, 1) résidus PDE au carré, détachés
    n_bins       : nombre d’intervalles de temps
    epsilon      : force de causalité (typiquement : 1,0 à 100,0)
    t_min, t_max : limites du domaine

    Retourne
    -------
    weights : (N, 1) tenseur dans (0, 1], même dispositif que t_f

    """
    device = t_f.device
    dt     = (t_max - t_min) / n_bins

    # -- bin index pour chaque point: 0 .. n_bins-1 --
    bin_idx = ((t_f - t_min) / dt).long().clamp(0, n_bins - 1)  # (N, 1)

    # -- résidu moyen par bin --
    bin_losses = torch.zeros(n_bins, device=device)
    bin_counts = torch.zeros(n_bins, device=device)
    flat_idx   = bin_idx.squeeze(1)
    flat_res   = residual_sq.squeeze(1).detach()

    bin_losses.scatter_add_(0, flat_idx, flat_res)
    bin_counts.scatter_add_(0, flat_idx, torch.ones_like(flat_res))
    bin_counts = bin_counts.clamp(min=1.0)
    bin_losses = bin_losses / bin_counts          # moyenne par bin

    # Somme cumulative décalée de 1 (en excluant le classeur actuel)
    # w_k = exp(-epsilon * ∑_{j=0}^{k-1} L_j)
    cum_losses = torch.cumsum(bin_losses, dim=0)
    cum_losses = torch.cat([torch.zeros(1, device=device), cum_losses[:-1]])
    bin_weights = torch.exp(-epsilon * cum_losses)  # (n_bins,)

    # -- assigne des poids par point --
    weights = bin_weights[flat_idx].unsqueeze(1)  # (N, 1)
    return weights


# -----------------------------------------------------------------------
# PINN loss avec causal weighting
# -----------------------------------------------------------------------

def pinn_loss(
    model: nn.Module,
    batch: tuple,
    cfg:   Config,
) -> tuple:
    """
    Calcule la perte PINN à trois termes avec pondération causale sur L_pde.

    Le résidu de l'équation différentielle partielle est pondéré par w(t) = exp(-epsilon * cumulative_loss)
    afin que le réseau soit contraint de satisfaire les conditions aux temps précoces avant celles aux temps tardifs.

    Paramètres
    ----------
    model : PINN
    batch : (x_f, t_f, x_b, t_b, x_i, t_i)
    cfg   : Config

    Retourne
    -------
    total_loss, loss_pde, loss_ic, loss_bc, mean_weight
    
    """
    x_f, t_f, x_b, t_b, x_i, t_i = batch

    # -- Résidu PDE --
    x_f  = x_f.requires_grad_(True)
    t_f  = t_f.requires_grad_(True)

    u    = model(x_f, t_f)
    u_t  = grad(u, t_f)
    u_x  = grad(u, x_f)
    u_xx = grad(u_x, x_f)

    residual    = u_t + u * u_x - cfg.nu * u_xx
    residual_sq = residual ** 2                    # (N, 1)

    # -- Causal weighting --
    if cfg.causal_epsilon > 0.0:
        w = causal_weights(
            t_f.detach(), residual_sq,
            n_bins  = cfg.causal_bins,
            epsilon = cfg.causal_epsilon,
            t_min   = cfg.t_min,
            t_max   = cfg.t_max,
        )
        loss_pde    = torch.mean(w * residual_sq)
        mean_weight = w.mean().item()
    else:
        loss_pde    = torch.mean(residual_sq)
        mean_weight = 1.0

    # -- CI  u(x, 0) = -sin(pi * x) --
    u_ic      = model(x_i, t_i)
    u_ic_true = -torch.sin(torch.pi * x_i)
    loss_ic   = torch.mean((u_ic - u_ic_true) ** 2)

    # -- CB  u(+/-1, t) = 0 --
    loss_bc = torch.mean(model(x_b, t_b) ** 2)

    # -- Weighted sum --
    total = (
        cfg.lambda_pde * loss_pde
        + cfg.lambda_ic  * loss_ic
        + cfg.lambda_bc  * loss_bc
    )

    return total, loss_pde, loss_ic, loss_bc, mean_weight


# -----------------------------------------------------------------------
# Génération de données d'entraînement
# -----------------------------------------------------------------------

def generate_data(cfg: Config, device: torch.device) -> tuple:
    """
    Échantillon de points de collocation, de frontière et de conditions initiales.

    Les points de collocation sont un mélange de :
      - tirages aléatoires uniformes sur l'ensemble du domaine  (75 %)
      - points adaptatifs concentrés près du front d'onde de choc (25 %)

    Renvoie
    -------
    (x_f, t_f, x_b, t_b, x_i, t_i) -- tous sur le périphérique, forme (N, 1)
    
    """
    # -- Points de collocation : background uniforme --
    N_uniform = int(cfg.N_f * 0.75)
    x_uniform = torch.rand(N_uniform, 1) * (cfg.x_max - cfg.x_min) + cfg.x_min
    t_uniform = torch.rand(N_uniform, 1) * (cfg.t_max - cfg.t_min) + cfg.t_min

    # -- Points de collocation : adaptation prêt du choc --
    N_shock = cfg.N_f - N_uniform
    x_shock = torch.randn(N_shock, 1) * 0.15
    x_shock = x_shock.clamp(cfg.x_min, cfg.x_max)
    t_shock = torch.rand(N_shock, 1) * 0.6 + 0.4

    x_f     = torch.cat([x_uniform, x_shock], dim=0)
    t_f     = torch.cat([t_uniform, t_shock], dim=0)

    # -- Points à la frontière  x = +/-1 --
    t_b = torch.rand(cfg.N_b, 1) * (cfg.t_max - cfg.t_min) + cfg.t_min
    x_b = torch.cat([
        torch.full((cfg.N_b // 2, 1), cfg.x_min),
        torch.full((cfg.N_b // 2, 1), cfg.x_max),
    ], dim = 0)

    # -- Condition initiale des points à t = 0 --
    x_i = torch.rand(cfg.N_i, 1) * (cfg.x_max - cfg.x_min) + cfg.x_min
    t_i = torch.zeros(cfg.N_i, 1)

    tensors = (x_f, t_f, x_b, t_b, x_i, t_i)
    return tuple(t.to(device) for t in tensors)
