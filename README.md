# Burgers PINN

Physics-Informed Neural Network (PINN) for the viscous Burgers equation, implemented in PyTorch.

## Equation

$$u_t + u\, u_x = \nu\, u_{xx}, \quad x \in [-1,1],\; t \in [0,1]$$

with:
- **Initial condition:** $u(x, 0) = -\sin(\pi x)$
- **Boundary conditions:** $u(-1, t) = u(1, t) = 0$
- **Viscosity:** $\nu = 0.01/\pi$

## Method

A fully-connected MLP approximates $u(x, t)$. The loss combines three terms:

$$\mathcal{L} = \lambda_\text{pde}\,\mathcal{L}_\text{pde} + \lambda_\text{ic}\,\mathcal{L}_\text{ic} + \lambda_\text{bc}\,\mathcal{L}_\text{bc}$$

where $\mathcal{L}_\text{pde}$ is the mean squared PDE residual computed via automatic differentiation.

Training uses **Adam** (warm-up) followed by **L-BFGS** (refinement), consistent with Raissi et al. (2019).

The prediction is quantitatively validated against the **exact Cole-Hopf solution**.

## Architecture

```
Input (x, t) → Linear(2→64) → tanh → ... × 4 → Linear(64→1) → u
```

Xavier weight initialisation. All hyperparameters in `config.py`.

## Project structure

```
pinn_burgers/
├── config.py        # all hyperparameters
├── model.py         # PINN network
├── physics.py       # PDE residual, loss, data generation
├── train.py         # Adam + L-BFGS training loop
├── evaluate.py      # Cole-Hopf exact solution, metrics, figures
├── main.py          # entry point
└── outputs/         # saved weights, metrics, figures (git-ignored)
```

## Quick start

```bash
pip install -r requirements.txt
python main.py
```

Outputs are written to `outputs/`:
- `model.pt` — trained weights
- `training_history.csv` — loss per epoch
- `metrics.csv` — final L2 / L∞ errors
- `heatmaps.png` — prediction vs exact vs error
- `slices.png` — time slice comparisons
- `loss_history.png` — training curves

## Results

| Metric | Value |
|---|---|
| Relative L2 error | ~O(1e-3) |
| Relative L∞ error | ~O(1e-3) |

*(Exact values depend on training duration and hardware.)*

## References

- Raissi, M., Perdikaris, P., Lagaris, I.E. (2019). *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.* Journal of Computational Physics, 378, 686–707.
