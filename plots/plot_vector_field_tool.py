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

    x = torch.arange(0, W)
    y = torch.arange(0, H)
    X, Y = torch.meshgrid(y, x, indexing='ij')  # (H, W) shapes to match vx, vy

    # Downsample for plotting clarity
    Xs = X[::step, ::step]
    Ys = Y[::step, ::step]
    vxs = vx[::step, ::step]
    vys = vy[::step, ::step]

    valid_mask = (~torch.isnan(vxs)) & (~torch.isnan(vys))

    Xs = Xs[valid_mask]
    Ys = Ys[valid_mask]
    vxs = vxs[valid_mask]
    vys = vys[valid_mask]

    plt.figure(figsize=(6, 6))
    plt.quiver(
        Xs.cpu(),
        Ys.cpu(),
        vxs.cpu(),
        vys.cpu(),
        scale=scale,
        color='blue'
    )
    #my_path = os.path.dirname(os.path.abspath(__file__))

    #plt.gca().invert_yaxis() THIS LINE WILL CONVERT ROTATIONAL FIELDS INTO RADIAL ONES AND VICE VERSA
    plt.axis("equal")
    plt.title(title)
    plt.grid(True)

    #output_path = os.path.join(my_path, 'outputs', os.path.basename(file))
    #plt.savefig(output_path)
    plt.savefig(file)
    plt.close()

