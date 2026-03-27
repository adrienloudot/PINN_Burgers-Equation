"""
Architecture PINN pour l'équation de Burgers.

Un MLP simple avec des fonctions d'activation tanh et une initialisation des poids selon la méthode de Xavier.
Entrée : (x, t) — coordonnées spatiales et temporelles
Sortie : u(x, t) — champ de vitesse prédit

"""

import torch
import torch.nn as nn
from torch import Tensor


class PINN(nn.Module):
    """
    Perceptron multicouche utilisé comme réseau neuronal informé par la physique.

    Paramètres
    ----------
    layers : liste[entier]
        Largeurs de chaque couche, y compris les couches d'entrée et de sortie.
        Exemple : [2, 64, 64, 64, 1]

    """

    def __init__(self, layers: list[int], sigma=1.0) -> None:
        super().__init__()
        self.net = nn.ModuleList([
            nn.Linear(layers[i], layers[i + 1])
            for i in range(len(layers) - 1)
        ])
        self._initialize_weights()

    # -- Initialisation de Xavier --
    def _initialize_weights(self) -> None:
        """Initialisation normale de Xavier pour les poids, zéros pour les biais."""
        for layer in self.net:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    # -- Calcul forward --
    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Paramètres
        ----------
        x : Tenseur de la forme (N, 1)
        t : Tenseur de la forme (N, 1)

        Résultats
        -------
        u : Tenseur de la forme (N, 1)

        """
        z = torch.cat([x, t], dim=1)
        for layer in self.net[:-1]:
          z = torch.tanh(layer(z))
        return self.net[-1](z)
