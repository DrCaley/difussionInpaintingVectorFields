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


def make_heatmap(tensor_2d, title=None, save_path='heat_map.png', show=True, cmap='viridis'):
    """
    Create a heatmap from a 2D torch tensor.

    Args:
        tensor_2d (torch.Tensor): A 2D tensor of shape (H, W)
        title (str, optional): Title for the plot.
        save_path (str, optional): Path to save the image.
        show (bool, optional): Whether to display the heatmap.
        cmap (str, optional): Matplotlib colormap to use.

    Returns:
        matplotlib.figure.Figure: The figure object for further use if needed.
    """
    if not isinstance(tensor_2d, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor")
    if tensor_2d.ndim != 2:
        raise ValueError("Input tensor must be 2D")

    tensor_np = tensor_2d.detach().cpu().numpy()

    fig, ax = plt.subplots()
    heatmap = ax.imshow(tensor_np, cmap=cmap)
    plt.colorbar(heatmap)

    if title:
        ax.set_title(title)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    return fig
