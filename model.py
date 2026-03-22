"""
PINN architecture for the Burgers equation.

A simple MLP with tanh activations and Xavier weight initialisation.
Input:  (x, t)  — spatial and temporal coordinates
Output: u(x, t) — predicted velocity field
"""

import torch
import torch.nn as nn
from torch import Tensor


class PINN(nn.Module):
    """
    Multi-layer perceptron used as a physics-informed neural network.

    Parameters
    ----------
    layers : list[int]
        Widths of each layer including input and output.
        Example: [2, 64, 64, 64, 1]
    """

    def __init__(self, layers: list[int], sigma=1.0) -> None:
        super().__init__()
        self.net = nn.ModuleList([
            nn.Linear(layers[i], layers[i + 1])
            for i in range(len(layers) - 1)
        ])
        self._initialize_weights()

    # ------------------------------------------------------------------
    def _initialize_weights(self) -> None:
        """Xavier normal init for weights, zeros for biases."""
        for layer in self.net:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    # ------------------------------------------------------------------
    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape (N, 1)
        t : Tensor of shape (N, 1)

        Returns
        -------
        u : Tensor of shape (N, 1)
        """
        z = torch.cat([x, t], dim=1)
        for layer in self.net[:-1]:
          z = torch.tanh(layer(z))
        return self.net[-1](z)
