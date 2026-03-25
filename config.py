"""
Centralized configuration for the Burgers PINN experiment.
All hyperparameters live here — never hardcoded elsewhere.
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class Config:
    # --- PDE ---
    nu: float = 0.01 / np.pi          # viscosity (standard Burgers benchmark)

    # --- Domain ---
    x_min: float = -1.0
    x_max: float = 1.0
    t_min: float = 0.0
    t_max: float = 1.0

    # --- Sampling ---
    N_f: int = 20_000   # collocation points (PDE residual)
    #N_f: int = 5_000
    N_b: int = 200      # boundary condition points
    N_i: int = 1_000    # initial condition points    
    #N_i: int = 200

    # --- Network ---
    layers: list = field(default_factory=lambda: [2, 128, 128, 128, 128, 1])
    #layers: list = field(default_factory=lambda: [2, 64, 64, 64, 1])

    # --- Loss weights ---
    lambda_pde: float = 1.0
    lambda_ic:  float = 20.0   # strong IC enforcement
    lambda_bc:  float = 10.0

    
    # --- Causal weighting (Wang et al. 2022) ---
    # epsilon > 0 activates causal weighting on the PDE loss.
    # The time domain is split into causal_bins bins.
    # Each bin k gets weight exp(-epsilon * sum_{j<k} L_j).
    # Larger epsilon = stricter causality enforcement.
    # Set epsilon = 0.0 to disable (standard PINN).

    #causal_epsilon: float = 5.0    # good starting value: 1.0 to 10.0
    #causal_bins:    int   = 100    # number of time bins

    causal_epsilon: float = 1.0    # était 5.0 — moins strict, laisse les poids monter
    causal_bins:    int   = 50     # était 100 — bins plus larges, plus stables

    # --- Optimizers ---
    lr_adam: float    = 1e-3
    epochs_adam: int  = 20_000  # used only if curriculum is disabled

    lbfgs_max_iter: int         = 1_000
    lbfgs_history_size: int     = 50
    lbfgs_tolerance_grad: float = 1e-9
    lbfgs_tolerance_change: float = 1e-12

    # --- Curriculum time-marching ---
    # Each tuple: (t_end, adam_epochs, lbfgs_max_iter)
    # The time window grows progressively so the network learns the
    # smooth early dynamics before tackling the sharp shock at t~1.

    curriculum_stages: list = field(default_factory=lambda: [
        (0.25,  3_000, 200),   # Stage 1: smooth sine, no shock
        (0.50,  5_000, 300),   # Stage 2: shock starting to form
        (0.75,  5_000, 300),   # Stage 3: sharp shock region
        (1.00, 10_000, 500),   # Stage 4: full domain
    ])

    # curriculum_stages: list = field(default_factory=lambda: [
    # (0.25, 1_000,  100),
    # (0.50, 2_000,  200),
    # (0.75, 2_000,  200),
    # (1.00, 5_000,  300),
    # ])


    
    # --- Evaluation ---
    skip_training: bool = False  # True = load model.pt and skip training

    # --- Logging ---
    log_every: int = 500          # print loss every N epochs
    save_path: str = "outputs/"   # directory for saved model + figures
