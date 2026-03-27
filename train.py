"""
Boucle d'entraînement : curriculum progressif utilisant Adam + L-BFGS.

Strategie
--------
Au lieu d'entraîner directement sur tout le domaine [0,1], on étend progressivement la fenêtre d'entraînement :

    Etape 1 : t dans [0, T1]        — pas encore de choc
    Etape 2 : t dans [0, T2]        — le choc se forme
    Etape 3 : t dans [0, 1.0]       — domaine entier avec le choc entièrement formé

Au sein de chaque étape : entraînement avec Adam puis rafinement avec L-BFGS.
Les poids du modèle sont conservés d'une étape à l'autre. Cela évite en effet que le réseau ne se stabilise sur une solution stationnaire erronée
avant d'avoir « détecté » la zone de choc
"""

import os
import csv
import time

import torch
import torch.nn as nn

from config import Config
from model import PINN
from physics import pinn_loss, generate_data


# ----------------------------------------------------------------------- #
# Entraînement sur une étape (Adam + L-BFGS)                              #
# ----------------------------------------------------------------------- #

def _train_stage(
    model:        nn.Module,
    batch:        tuple,
    cfg:          Config,
    epochs_adam:  int,
    lbfgs_iter:   int,
    stage_label:  str,
    epoch_offset: int,
    t0:           float,
) -> list[dict]:
    """Run one curriculum stage: Adam then L-BFGS."""
    history: list[dict] = []

    # -- Adam --
    optimizer_adam = torch.optim.Adam(model.parameters(), lr=cfg.lr_adam)
    print(f"  Adam ({epochs_adam} epochs) …")

    for epoch in range(1, epochs_adam + 1):
        optimizer_adam.zero_grad()
        loss, l_pde, l_ic, l_bc, w_mean = pinn_loss(model, batch, cfg)
        loss.backward()
        optimizer_adam.step()

        history.append({
            "epoch":    epoch_offset + epoch,
            "stage":    stage_label,
            "phase":    "adam",
            "loss":     loss.item(),
            "loss_pde": l_pde.item(),
            "loss_ic":  l_ic.item(),
            "loss_bc":  l_bc.item(),
            "w_mean":   w_mean,
            "elapsed":  time.perf_counter() - t0,
        })

        if epoch % cfg.log_every == 0 or epoch == 1:
            print(
                f"  [{epoch:>5d}]  loss={loss.item():.3e}"
                f"  pde={l_pde.item():.3e}"
                f"  ic={l_ic.item():.3e}"
                f"  bc={l_bc.item():.3e}"
                f"  w={w_mean:.3f}"
            )

    # -- L-BFGS --
    optimizer_lbfgs = torch.optim.LBFGS(
        model.parameters(),
        max_iter         = lbfgs_iter,
        history_size     = cfg.lbfgs_history_size,
        tolerance_grad   = cfg.lbfgs_tolerance_grad,
        tolerance_change = cfg.lbfgs_tolerance_change,
        line_search_fn   = "strong_wolfe",
    )

    iter_count = [0]

    def closure():
        optimizer_lbfgs.zero_grad()
        loss, l_pde, l_ic, l_bc, w_mean = pinn_loss(model, batch, cfg)
        loss.backward()
        iter_count[0] += 1

        history.append({
            "epoch":    epoch_offset + epochs_adam + iter_count[0],
            "stage":    stage_label,
            "phase":    "lbfgs",
            "loss":     loss.item(),
            "loss_pde": l_pde.item(),
            "loss_ic":  l_ic.item(),
            "loss_bc":  l_bc.item(),
            "w_mean":   w_mean,
            "elapsed":  time.perf_counter() - t0,
        })

        if iter_count[0] % cfg.log_every == 0 or iter_count[0] == 1:
            print(
                f"    [L-BFGS {iter_count[0]:>4d}]  loss={loss.item():.3e}"
                f"  pde={l_pde.item():.3e}"
                f"  ic={l_ic.item():.3e}"
                f"  bc={l_bc.item():.3e}"
                f"  w={w_mean:.3f}"
            )
        return loss

    print(f"  L-BFGS (max {lbfgs_iter} iterations) …")
    optimizer_lbfgs.step(closure)

    return history


# ---------------------------------------------------------------------- #
# Entraînement progressif                                                #
# ---------------------------------------------------------------------- #

def train(
    model: nn.Module,
    batch: tuple,           # ignoré — nous régénérons à chaque étape
    cfg: Config,
    device: torch.device,
) -> list[dict]:
    """
    La fenêtre temporelle s'étend sur les étapes définies dans `cfg.curriculum_stages`.
    Chaque étape dispose de son propre ensemble de points de collocation, échantillonnés uniquement
    dans l'intervalle [0, t_end] pour cette étape.

    Paramètres
    ----------
    model   : PINN 
    batch   : unused (gardé pour la compatibilité API avec main.py)
    cfg     : Config
    device  : torch.device

    Returns
    -------
    history : liste des dictionnaires enregistrés à chaque époque Adam et à chaque étape L-BFGS
    """
    t0                  = time.perf_counter()
    history: list[dict] = []
    epoch_offset        = 0
    stages              = cfg.curriculum_stages  # liste de (t_end, epochs_adam, lbfgs_iter)

    for idx, (t_end, epochs_adam, lbfgs_iter) in enumerate(stages):
        label = f"stage{idx+1}_t{t_end}"
        print()
        print("=" * 60)
        print(f"  Curriculum stage {idx+1}/{len(stages)} — t ∈ [0, {t_end}]")
        print("=" * 60)

        # Créer un nouveau lot limité à cette plage de temps
        stage_cfg = Config(
            nu = cfg.nu,
            x_min = cfg.x_min, x_max = cfg.x_max,
            t_min = cfg.t_min, t_max = t_end,          # <-- shrunk window
            N_f = cfg.N_f, N_b = cfg.N_b, N_i = cfg.N_i,
            layers = cfg.layers,
            lambda_pde = cfg.lambda_pde,
            lambda_ic = cfg.lambda_ic,
            lambda_bc = cfg.lambda_bc,
            lr_adam = cfg.lr_adam,
            epochs_adam = epochs_adam,
            lbfgs_max_iter = lbfgs_iter,
            lbfgs_history_size = cfg.lbfgs_history_size,
            lbfgs_tolerance_grad = cfg.lbfgs_tolerance_grad,
            lbfgs_tolerance_change = cfg.lbfgs_tolerance_change,
            log_every = cfg.log_every,
            save_path = cfg.save_path,
            curriculum_stages = cfg.curriculum_stages,
        )
        stage_batch = generate_data(stage_cfg, device)

        stage_hist = _train_stage(
            model, stage_batch, stage_cfg,
            epochs_adam = epochs_adam,
            lbfgs_iter = lbfgs_iter,
            stage_label = label,
            epoch_offset = epoch_offset,
            t0=t0,
        )
        history.extend(stage_hist)
        epoch_offset += epochs_adam + lbfgs_iter

        elapsed = time.perf_counter() - t0
        final_loss = stage_hist[-1]["loss"]
        print(f"  Stage {idx+1} done — loss={final_loss:.3e}  ({elapsed:.0f}s elapsed)")

    total_time = time.perf_counter() - t0
    print()
    print(f"Training complete — {total_time:.1f}s total")
    return history


# ---------------------------------------------------------------------- #
# Enregistrement l'historique dans un fichier CSV                        #
# ---------------------------------------------------------------------- #

def save_history(history: list[dict], cfg: Config) -> None:
    os.makedirs(cfg.save_path, exist_ok=True)
    path = os.path.join(cfg.save_path, "training_history.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    print(f"History saved → {path}")


# ---------------------------------------------------------------------- #
# Point d'entrée (lorsqu'il est appelé directement)                      #
# ---------------------------------------------------------------------- #

if __name__ == "__main__":
    cfg    = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    model = PINN(cfg.layers).to(device)
    batch = generate_data(cfg, device)

    history = train(model, batch, cfg, device)
    save_history(history, cfg)

    os.makedirs(cfg.save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(cfg.save_path, "model.pt"))
    print(f"Model saved → {cfg.save_path}model.pt")
