"""
Configuration centralisée pour l'expérience Burgers PINN.
Tous les hyperparamètres se trouvent ici — ils ne sont jamais codés en dur ailleurs.

"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class Config:
    # -- PDE --
    nu: float = 0.01 / np.pi          # viscosité (benchmark standard pour l'équation de Burgers)

    # -- Domaine --
    x_min: float = -1.0
    x_max: float = 1.0
    t_min: float = 0.0
    t_max: float = 1.0

    # -- Sampling --
    N_f: int = 20_000   # points de collocation (résidus PDE)
    #N_f: int = 5_000   # run plus court
    N_b: int = 200      # boundary condition points
    N_i: int = 1_000    # initial condition points    
    #N_i: int = 200     # run plus court

    # -- Réseau utilisé --
    layers: list = field(default_factory=lambda: [2, 128, 128, 128, 128, 1]) # (~50k paramètres)
    #layers: list = field(default_factory=lambda: [2, 64, 64, 64, 1]) # Plus rapide, moins profond (~8k paramètres)

    # -- Loss weights --
    lambda_pde: float = 1.0
    lambda_ic:  float = 20.0   # beaucoup de poids est mis sur la loss liée aux conditions limites
    lambda_bc:  float = 10.0
    
    # --- Causal weighting (Wang et al. 2022) ---
    # Une valeur d'epsilon > 0 active la pondération causale sur la perte PDE.
    # Le domaine temporel est divisé en causal_bins intervalles.
    # Chaque intervalle k reçoit un poids égal à exp(-epsilon * sum_{j<k} L_j).
    # Plus la valeur d'epsilon est grande, plus l'application de la causalité est stricte.
    # Définissez epsilon = 0,0 pour désactiver cette fonctionnalité (PINN standard). 

    causal_epsilon: float = 1.0    
    causal_bins:    int   = 50     

    # -- Optimiseurs --
    lr_adam: float    = 1e-3
    epochs_adam: int  = 20_000  # utilisé seulement si le curriculum temporel n'est pas utilisé

    lbfgs_max_iter: int           = 1_000
    lbfgs_history_size: int       = 50
    lbfgs_tolerance_grad: float   = 1e-9
    lbfgs_tolerance_change: float = 1e-12

    # -- Curriculum time-marching --
    # Chaque tuple: (t_end, adam_epochs, lbfgs_max_iter)
    # La fenêtre temporelle s'élargit progressivement afin que le réseau apprenne la
    # dynamique initiale régulière avant d'aborder le choc brusque à t~1.

    curriculum_stages: list = field(default_factory=lambda: [
        (0.25,  3_000, 200),   # étape 1: pas de choc
        (0.50,  5_000, 300),   # étape 2: le choc commence à se former
        (0.75,  5_000, 300),   # étape 3: choc brutal
        (1.00, 10_000, 500),   # étape 4: domaine entier
    ])

    # version plus rapide :
    # curriculum_stages: list = field(default_factory=lambda: [
    # (0.25, 1_000,  100),
    # (0.50, 2_000,  200),
    # (0.75, 2_000,  200),
    # (1.00, 5_000,  300),
    # ])
    
    # -- Evaluation --
    skip_training: bool = False  # True = charger model.pt et sauter l'entraînement des poids

    # -- Logging --
    log_every: int = 500         # print la loss tous les N epochs
    save_path: str = "outputs/"  # dossier pour enregistrer les outputs et le modèle
