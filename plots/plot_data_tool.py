from fileinput import filename
import os

import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import torch

def plot_vector_field(vx: torch.Tensor, vy: torch.Tensor, step: int = 1, scale: float = 1.0, title: str = "Vector Field", file: str = "vector_field.png"):
    assert vx.shape == vy.shape, "vx and vy must be the same shape"
    H, W = vx.shape

    # Explicitly clone and replace NaNs with 0
    vx = vx.clone()
    vy = vy.clone()
    vx[torch.isnan(vx)] = 0.0
    vy[torch.isnan(vy)] = 0.0

    print("Any NaNs in vx:", torch.isnan(vx).any().item())
    print("Any NaNs in vy:", torch.isnan(vy).any().item())

    # Create meshgrid
    x = torch.arange(0, W)
    y = torch.arange(0, H)
    X, Y = torch.meshgrid(x, y, indexing='ij')

    print("Any NaNs in X:", torch.isnan(X).any().item())
    print("Any NaNs in Y:", torch.isnan(Y).any().item())

    plt.figure(figsize=(6, 6))
    plt.quiver(
        X[::step, ::step],
        Y[::step, ::step],
        vx[::step, ::step],
        vy[::step, ::step],
        scale=scale,
        color='blue'
    )
    my_path = os.path.dirname(os.path.abspath(__file__))
    plt.gca().invert_yaxis()
    plt.axis("equal")
    plt.title(title)
    plt.grid(True)

    output_path = os.path.join(my_path, 'outputs', os.path.basename(file))
    plt.savefig(output_path)
    plt.close()
