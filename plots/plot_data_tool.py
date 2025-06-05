from fileinput import filename

import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import torch

def plot_vector_field(vx: torch.Tensor, vy: torch.Tensor, step: int = 1, scale: float = 1.0, title: str = "Vector Field", file: str = "vector_field.png"):
    """
    Plots a 2D quiver plot from vx and vy tensors.

    Args:
        vx (torch.Tensor): X-component of the vector field (2D).
        vy (torch.Tensor): Y-component of the vector field (2D).
        step (int): Step size for downsampling vectors (for clarity).
        scale (float): Scale factor for arrows.
        title (str): Title of the plot.
        file (str): File path to save the plot.
    """
    assert vx.shape == vy.shape, "vx and vy must be the same shape"
    H, W = vx.shape

    # Create meshgrid
    x = torch.arange(0, W)
    y = torch.arange(0, H)
    X, Y = torch.meshgrid(x, y, indexing='ij')

    # Plot using matplotlib
    plt.figure(figsize=(6, 6))
    plt.quiver(
        X[::step, ::step],
        Y[::step, ::step],
        vx[::step, ::step],
        vy[::step, ::step],
        scale=scale,
        color='blue'
    )
    plt.gca().invert_yaxis()
    plt.axis("equal")
    plt.title(title)
    plt.grid(True)
    plt.savefig(file)

    plt.close()

