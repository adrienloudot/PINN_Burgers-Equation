"""
main.py — entry point for the Burgers PINN experiment.

Run (full pipeline):
    python main.py

Run (evaluation only, reuse existing model.pt):
    set skip_training = True in config.py
"""

import os
import csv
import numpy as np
import torch

from config import Config
from model import PINN
from physics import generate_data
from train import train, save_history
from evaluate import (
    predict_on_grid,
    exact_solution,
    relative_l2,
    relative_linf,
    plot_results,
    plot_training_history,
)


def main() -> None:
    # ------------------------------------------------------------------ #
    # Setup                                                                #
    # ------------------------------------------------------------------ #
    cfg    = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.save_path, exist_ok=True)
    weights_path = os.path.join(cfg.save_path, "model.pt")

    print("=" * 60)
    print("  Burgers PINN")
    print(f"  nu = {cfg.nu:.6f}   device = {device}")
    print(f"  architecture: {cfg.layers}")
    print(f"  skip_training = {cfg.skip_training}")
    print("=" * 60)
    print()

    # ------------------------------------------------------------------ #
    # Model                                                                #
    # ------------------------------------------------------------------ #
    model = PINN(cfg.layers).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Network parameters: {n_params:,}\n")

    # ------------------------------------------------------------------ #
    # Training  — or load existing weights                                 #
    # ------------------------------------------------------------------ #
    if cfg.skip_training:
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"skip_training=True but no weights found at {weights_path}\n"
                "Run once with skip_training=False first."
            )
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()
        print(f"Loaded existing weights from {weights_path}")
        print("Skipping training.\n")
        history = None
    else:
        batch   = generate_data(cfg, device)
        history = train(model, batch, cfg, device)
        save_history(history, cfg)
        torch.save(model.state_dict(), weights_path)
        print(f"Weights saved -> {weights_path}\n")

    # ------------------------------------------------------------------ #
    # Evaluation                                                           #
    # ------------------------------------------------------------------ #
    Nx, Nt = 100, 50
    x_vec = np.linspace(cfg.x_min, cfg.x_max, Nx)
    t_vec = np.linspace(cfg.t_min, cfg.t_max, Nt)

    print("Computing PINN prediction on grid ...")
    U_pred = predict_on_grid(model, x_vec, t_vec, device)

    # Cache exact solution — invalidate automatically if grid size changed
    exact_cache    = os.path.join(cfg.save_path, "U_exact.npy")
    expected_shape = (Nx, Nt)

    if os.path.exists(exact_cache):
        U_exact = np.load(exact_cache)
        if U_exact.shape != expected_shape:
            print(f"Cache shape mismatch {U_exact.shape} vs {expected_shape} — recomputing ...")
            os.remove(exact_cache)
            U_exact = exact_solution(x_vec, t_vec, cfg.nu)
            np.save(exact_cache, U_exact)
            print(f"Exact solution cached -> {exact_cache}")
        else:
            print(f"Loaded cached exact solution — shape {U_exact.shape} ok")
    else:
        print("Computing Cole-Hopf exact solution (this takes a few minutes) ...")
        U_exact = exact_solution(x_vec, t_vec, cfg.nu)
        np.save(exact_cache, U_exact)
        print(f"Exact solution cached -> {exact_cache}")

    print(f"U_exact  min={U_exact.min():.3f}  max={U_exact.max():.3f}")
    print(f"U_pred   min={U_pred.min():.3f}  max={U_pred.max():.3f}")

    l2   = relative_l2(U_pred, U_exact)
    linf = relative_linf(U_pred, U_exact)
    print(f"\n  Relative L2   error : {l2:.4e}")
    print(f"  Relative Linf error : {linf:.4e}\n")

    # Save metrics
    metrics_path = os.path.join(cfg.save_path, "metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["relative_l2",   l2])
        writer.writerow(["relative_linf", linf])
    print(f"Metrics saved -> {metrics_path}")

    # ------------------------------------------------------------------ #
    # Plots                                                                #
    # ------------------------------------------------------------------ #
    if history is not None:
        plot_training_history(history, cfg)

    plot_results(U_pred, U_exact, x_vec, t_vec, cfg)


if __name__ == "__main__":
    main()
