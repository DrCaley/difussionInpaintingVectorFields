from fileinput import filename
import os

import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import torch



def plot_vector_field(
    vx: torch.Tensor,
    vy: torch.Tensor,
    step: int = 1,
    scale: float = 1.0,
    title: str = "Vector Field",
    file: str = "vector_field.png",
    land_mask: torch.Tensor = None,
    land_color: str = "forestgreen",
    flip_x_axis: bool = False,
    crop_top_right_zero_pad: bool = False,
    zero_eps: float = 1e-8,
    auto_rescale_for_display: bool = True,
    target_median_arrow_len: float = 0.15,
    lon_bounds: tuple[float, float] | None = None,
    lat_bounds: tuple[float, float] | None = None,
    missing_mask: torch.Tensor = None,
    missing_color: str = "red",
    missing_alpha: float = 0.25,
):
    assert vx.shape == vy.shape, "vx and vy must be the same shape"

    H0, W0 = vx.shape
    if lon_bounds is not None and lat_bounds is not None:
        lon_vec = torch.linspace(float(lon_bounds[0]), float(lon_bounds[1]), W0)
        lat_vec = torch.linspace(float(lat_bounds[1]), float(lat_bounds[0]), H0)
    else:
        lon_vec = torch.arange(0, W0, dtype=torch.float32)
        lat_vec = torch.arange(0, H0, dtype=torch.float32)

    if flip_x_axis:
        vx = torch.flip(vx, dims=[0])
        vy = torch.flip(vy, dims=[0])
        lat_vec = torch.flip(lat_vec, dims=[0])
        if land_mask is not None:
            land_mask = torch.flip(land_mask, dims=[0])

    if missing_mask is not None:
        if missing_mask.dim() == 4:
            missing_mask = missing_mask.squeeze(0)[0]  # (H,W)
        elif missing_mask.dim() == 3:
            missing_mask = missing_mask[0]  # (H,W)
        if flip_x_axis:
            missing_mask = torch.flip(missing_mask, dims=[0])

    if crop_top_right_zero_pad:
        if land_mask is not None:
            nonzero_mask = ~land_mask.bool()
        else:
            nonzero_mask = (vx.abs() > zero_eps) | (vy.abs() > zero_eps)

        rows_with_signal = nonzero_mask.any(dim=1)
        cols_with_signal = nonzero_mask.any(dim=0)

        if rows_with_signal.any() and cols_with_signal.any():
            top_idx = int(torch.nonzero(rows_with_signal, as_tuple=False)[0].item())
            right_idx = int(torch.nonzero(cols_with_signal, as_tuple=False)[-1].item())
            vx = vx[top_idx:, : right_idx + 1]
            vy = vy[top_idx:, : right_idx + 1]
            lat_vec = lat_vec[top_idx:]
            lon_vec = lon_vec[: right_idx + 1]
            if land_mask is not None:
                land_mask = land_mask[top_idx:, : right_idx + 1]
            if missing_mask is not None:
                missing_mask = missing_mask[top_idx:, : right_idx + 1]

    H, W = vx.shape

    X, Y = torch.meshgrid(lon_vec, lat_vec, indexing='xy')

    # Downsample for plotting clarity
    Xs = X[::step, ::step]
    Ys = Y[::step, ::step]
    vxs = vx[::step, ::step]
    vys = vy[::step, ::step]

    valid_mask = (~torch.isnan(vxs)) & (~torch.isnan(vys))
    if land_mask is not None:
        land_ds = land_mask[::step, ::step]
        valid_mask = valid_mask & (~land_ds.bool())

    Xs = Xs[valid_mask]
    Ys = Ys[valid_mask]
    vxs = vxs[valid_mask]
    vys = vys[valid_mask]

    if auto_rescale_for_display and vxs.numel() > 0:
        effective_target_len = target_median_arrow_len
        if lon_bounds is not None and lat_bounds is not None and W > 1 and H > 1:
            lon_step = torch.abs(lon_vec[1] - lon_vec[0]) * step
            lat_step = torch.abs(lat_vec[1] - lat_vec[0]) * step
            effective_target_len = 1.35 * float(torch.minimum(lon_step, lat_step))

        magnitudes = torch.sqrt(vxs ** 2 + vys ** 2)
        nonzero = magnitudes[magnitudes > zero_eps]
        if nonzero.numel() > 0:
            median_mag = torch.quantile(nonzero, 0.5)
            gain = effective_target_len / (median_mag + 1e-12)
            vxs = vxs * gain
            vys = vys * gain

    max_dim = 8
    figsize = (max_dim, max_dim * H / W) if W > H else (max_dim * W / H, max_dim)
    plt.figure(figsize=figsize)

    if land_mask is not None:
        if land_mask.shape != (H, W):
            raise ValueError("land_mask must have the same (H, W) shape as vx/vy")

        land_np = land_mask.detach().cpu().numpy().astype(float)
        color_rgba = to_rgba(land_color, alpha=0.8)
        land_rgba = torch.zeros((H, W, 4), dtype=torch.float32)
        land_rgba[..., 0] = color_rgba[0]
        land_rgba[..., 1] = color_rgba[1]
        land_rgba[..., 2] = color_rgba[2]
        land_rgba[..., 3] = torch.from_numpy(land_np) * color_rgba[3]  # alpha on land only
        plt.imshow(
            land_rgba.numpy(),
            origin='upper',
            extent=[float(lon_vec[0]), float(lon_vec[-1]), float(lat_vec[-1]), float(lat_vec[0])],
            zorder=1,
        )

    # Semi-transparent overlay on missing (masked) region
    if missing_mask is not None:
        if missing_mask.shape != (H, W):
            raise ValueError("missing_mask must have the same (H, W) shape as vx/vy")
        miss_np = missing_mask.detach().cpu().numpy().astype(float)
        miss_rgba_color = to_rgba(missing_color, alpha=missing_alpha)
        miss_rgba = torch.zeros((H, W, 4), dtype=torch.float32)
        miss_rgba[..., 0] = miss_rgba_color[0]
        miss_rgba[..., 1] = miss_rgba_color[1]
        miss_rgba[..., 2] = miss_rgba_color[2]
        miss_rgba[..., 3] = torch.from_numpy(miss_np) * miss_rgba_color[3]
        plt.imshow(
            miss_rgba.numpy(),
            origin='upper',
            extent=[float(lon_vec[0]), float(lon_vec[-1]), float(lat_vec[-1]), float(lat_vec[0])],
            zorder=1.5,
        )

    plt.quiver(
        Xs.cpu(),
        Ys.cpu(),
        vxs.cpu(),
        vys.cpu(),
        angles='xy',
        scale_units='xy',
        scale=scale,
        width=0.002,
        headwidth=2.5,
        headlength=3.0,
        headaxislength=2.5,
        color='blue',
        zorder=2,
    )
    #my_path = os.path.dirname(os.path.abspath(__file__))

    #plt.gca().invert_yaxis() THIS LINE WILL CONVERT ROTATIONAL FIELDS INTO RADIAL ONES AND VICE VERSA
    if lon_bounds is not None and lat_bounds is not None:
        plt.gca().set_aspect('auto')
    else:
        plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim(float(lon_vec[0]), float(lon_vec[-1]))
    plt.ylim(float(lat_vec[-1]), float(lat_vec[0]))
    if lon_bounds is not None and lat_bounds is not None:
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
    plt.title(title)
    plt.grid(False)

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
