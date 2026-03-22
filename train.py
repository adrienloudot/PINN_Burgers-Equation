"""
Training loop: curriculum time-marching + Adam + L-BFGS.

Strategy
--------
Instead of training on the full domain t in [0, 1] from the start,
we progressively extend the time window:

    Stage 1 : t in [0, T1]        — easy, no shock yet
    Stage 2 : t in [0, T2]        — shock forming
    Stage 3 : t in [0, 1.0]       — full domain with sharp shock

Within each stage: Adam warm-up then L-BFGS refinement.
The model weights carry over between stages.

This avoids the network settling into a spurious stationary solution
before it has "seen" the shock region.
"""

import os
import csv
import time

import torch
import torch.nn as nn

from config import Config
from model import PINN
from physics import pinn_loss, generate_data


# -----------------------------------------------------------------------
# Single-stage training (Adam + L-BFGS)
# -----------------------------------------------------------------------

def _train_stage(
    model: nn.Module,
    batch: tuple,
    cfg: Config,
    epochs_adam: int,
    lbfgs_iter: int,
    stage_label: str,
    epoch_offset: int,
    t0: float,
) -> list[dict]:
    """Run one curriculum stage: Adam then L-BFGS."""
    history: list[dict] = []

    # ---- Adam ------------------------------------------------------------
    optimizer_adam = torch.optim.Adam(model.parameters(), lr=cfg.lr_adam)
    print(f"  Adam ({epochs_adam} epochs) …")

    for epoch in range(1, epochs_adam + 1):
        optimizer_adam.zero_grad()
        loss, l_pde, l_ic, l_bc = pinn_loss(model, batch, cfg)
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
            "elapsed":  time.perf_counter() - t0,
        })

        if epoch % cfg.log_every == 0 or epoch == 1:
            print(
                f"    [{epoch:>5d}]  loss={loss.item():.3e}"
                f"  pde={l_pde.item():.3e}"
                f"  ic={l_ic.item():.3e}"
                f"  bc={l_bc.item():.3e}"
            )

    # ---- L-BFGS ----------------------------------------------------------
    optimizer_lbfgs = torch.optim.LBFGS(
        model.parameters(),
        max_iter=lbfgs_iter,
        history_size=cfg.lbfgs_history_size,
        tolerance_grad=cfg.lbfgs_tolerance_grad,
        tolerance_change=cfg.lbfgs_tolerance_change,
        line_search_fn="strong_wolfe",
    )

    iter_count = [0]

    def closure():
        optimizer_lbfgs.zero_grad()
        loss, l_pde, l_ic, l_bc = pinn_loss(model, batch, cfg)
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
            "elapsed":  time.perf_counter() - t0,
        })

        if iter_count[0] % cfg.log_every == 0 or iter_count[0] == 1:
            print(
                f"    [L-BFGS {iter_count[0]:>4d}]  loss={loss.item():.3e}"
                f"  pde={l_pde.item():.3e}"
                f"  ic={l_ic.item():.3e}"
                f"  bc={l_bc.item():.3e}"
            )
        return loss

    print(f"  L-BFGS (max {lbfgs_iter} iterations) …")
    optimizer_lbfgs.step(closure)

    return history


# -----------------------------------------------------------------------
# Curriculum training
# -----------------------------------------------------------------------

def train(
    model: nn.Module,
    batch: tuple,           # ignored — we regenerate per stage
    cfg: Config,
    device: torch.device,
) -> list[dict]:
    """
    Curriculum time-marching training.

    The time window grows across stages defined in cfg.curriculum_stages.
    Each stage gets its own batch of collocation points sampled only
    within [0, t_end] for that stage.

    Parameters
    ----------
    model   : PINN (already on device)
    batch   : unused (kept for API compatibility with main.py)
    cfg     : Config
    device  : torch.device

    Returns
    -------
    history : list of dicts logged at every Adam epoch and L-BFGS step
    """
    t0 = time.perf_counter()
    history: list[dict] = []
    epoch_offset = 0

    stages = cfg.curriculum_stages  # list of (t_end, epochs_adam, lbfgs_iter)

    for idx, (t_end, epochs_adam, lbfgs_iter) in enumerate(stages):
        label = f"stage{idx+1}_t{t_end}"
        print()
        print("=" * 60)
        print(f"  Curriculum stage {idx+1}/{len(stages)} — t ∈ [0, {t_end}]")
        print("=" * 60)

        # Build a fresh batch restricted to this time window
        stage_cfg = Config(
            nu=cfg.nu,
            x_min=cfg.x_min, x_max=cfg.x_max,
            t_min=cfg.t_min, t_max=t_end,          # <-- shrunk window
            N_f=cfg.N_f, N_b=cfg.N_b, N_i=cfg.N_i,
            layers=cfg.layers,
            lambda_pde=cfg.lambda_pde,
            lambda_ic=cfg.lambda_ic,
            lambda_bc=cfg.lambda_bc,
            lr_adam=cfg.lr_adam,
            epochs_adam=epochs_adam,
            lbfgs_max_iter=lbfgs_iter,
            lbfgs_history_size=cfg.lbfgs_history_size,
            lbfgs_tolerance_grad=cfg.lbfgs_tolerance_grad,
            lbfgs_tolerance_change=cfg.lbfgs_tolerance_change,
            log_every=cfg.log_every,
            save_path=cfg.save_path,
            curriculum_stages=cfg.curriculum_stages,
        )
        stage_batch = generate_data(stage_cfg, device)

        stage_hist = _train_stage(
            model, stage_batch, stage_cfg,
            epochs_adam=epochs_adam,
            lbfgs_iter=lbfgs_iter,
            stage_label=label,
            epoch_offset=epoch_offset,
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


# -----------------------------------------------------------------------
# Persist history to CSV
# -----------------------------------------------------------------------

def save_history(history: list[dict], cfg: Config) -> None:
    os.makedirs(cfg.save_path, exist_ok=True)
    path = os.path.join(cfg.save_path, "training_history.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    print(f"History saved → {path}")


# -----------------------------------------------------------------------
# Entry-point (when called directly)
# -----------------------------------------------------------------------

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
