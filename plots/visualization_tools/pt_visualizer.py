import matplotlib
matplotlib.use('Agg')

import torch
import matplotlib.pyplot as plt
import numpy as np
import os

from data_prep.data_initializer import DDInitializer
from plots.visualization_tools.error_visualization import save_mse_heatmap, save_angular_error_heatmap
from ddpm.utils.inpainting_utils import calculate_mse
dd = DDInitializer()

def visualize_tensor(
    tensor,
    title="tensor",
    save_dir="tensor_images",
    height_range=None,
    width_range=None,
    vector_scale=1.0
):
    """
    Visualizes and saves 2D/3D/4D tensors (e.g. masks, vector fields, grayscale images).
    Optionally selects a subregion and adjusts vector arrow scale.
    """
    tensor = tensor.squeeze()
    os.makedirs(save_dir, exist_ok=True)

    def crop(t):
        if height_range is not None:
            t = t[..., height_range[0]:height_range[1], :]
        if width_range is not None:
            t = t[..., :, width_range[0]:width_range[1]]
        return t

    if tensor.ndim == 2:
        # Single 2D array (e.g., mask)
        tensor = crop(tensor)
        plt.imshow(tensor.cpu(), cmap='gray')
        plt.title(title)
        plt.colorbar()
        plt.savefig(os.path.join(save_dir, f"{title}.png"))
        plt.close()

    elif tensor.ndim == 3:
        if tensor.shape[0] == 2:
            # Vector field: tensor shape (2, H, W)
            tensor = crop(tensor)
            u, v = tensor[0], tensor[1]
            H, W = u.shape
            x, y = np.meshgrid(np.arange(W), np.arange(H))

            # Dynamically set figure size preserving aspect ratio, max dimension ~8
            max_dim = 8
            if W > H:
                figsize = (max_dim, max_dim * H / W)
            else:
                figsize = (max_dim * W / H, max_dim)

            plt.figure(figsize=figsize)
            plt.quiver(x, y, u.cpu(), v.cpu(), scale=1.0/vector_scale)
            plt.title(title + " (Vector Field)")
            plt.gca().invert_yaxis()
            plt.gca().set_aspect('equal', adjustable='box')  # preserve aspect ratio
            plt.savefig(os.path.join(save_dir, f"{title}_vector_field.png"))
            plt.close()
        else:
            # Grayscale channels
            tensor = crop(tensor)
            for i in range(tensor.shape[0]):
                plt.imshow(tensor[i].cpu(), cmap='gray')
                plt.title(f"{title} - Channel {i}")
                plt.colorbar()
                plt.savefig(os.path.join(save_dir, f"{title}_channel_{i}.png"))
                plt.close()

    elif tensor.ndim == 4:
        print(f"Tensor has shape {tensor.shape}, saving first item in batch.")
        visualize_tensor(
            tensor[0],
            title=title + "_sample_0",
            save_dir=save_dir,
            height_range=height_range,
            width_range=width_range,
            vector_scale=vector_scale
        )

    else:
        print("Unsupported tensor shape:", tensor.shape)

def load_and_visualize_pt(file_path, title="loaded_tensor", save_dir="pt_visualizer_images", **kwargs):
    """
    Loads a .pt file and visualizes (saves) the tensor it contains.
    Passes optional args to the visualizer.
    """
    tensor = torch.load(file_path, map_location='cpu', weights_only=False)
    visualize_tensor(tensor, title=title, save_dir=f"{save_dir}", **kwargs)

load_and_visualize_pt(
    '../../ddpm/testing/results/img0_GaussianNoiseBinaryMaskGenerator(threshold=-1, mean=0.0, std=1.0)_resample5_num_lines_0.pt',
    'img0_GaussianNoiseBinaryMaskGenerator',
    vector_scale=0.15
)
load_and_visualize_pt(
    '../../ddpm/testing/results/initial0_GaussianNoiseBinaryMaskGenerator(threshold=-1, mean=0.0, std=1.0)_resample5_num_lines_0.pt',
    'initial0_GaussianNoiseBinaryMaskGenerator',
    vector_scale=0.15
)
load_and_visualize_pt(
    '../../ddpm/testing/results/interpolated0_GaussianNoiseBinaryMaskGenerator(threshold=-1, mean=0.0, std=1.0)_resample5_num_lines_0.pt',
    'interpolated0_GaussianNoiseBinaryMaskGenerator',
    vector_scale=0.15
)
tensor = torch.load('../../ddpm/testing/results/mask0_GaussianNoiseBinaryMaskGenerator(threshold=-1, mean=0.0, std=1.0)_resample5_num_lines_0.pt',
                    map_location='cpu')
visualize_tensor(tensor[0,0], 'mask0_GaussianNoiseBinaryMaskGenerator', save_dir='pt_visualizer_images')

actual = torch.load("../../ddpm/testing/results/img0_GaussianNoiseBinaryMaskGenerator(threshold=-1, mean=0.0, std=1.0)_resample5_num_lines_0.pt", map_location='cpu', weights_only=False)
interpolation = torch.load("../../ddpm/testing/results/interpolated0_GaussianNoiseBinaryMaskGenerator(threshold=-1, mean=0.0, std=1.0)_resample5_num_lines_0.pt", map_location='cpu', weights_only=False).unsqueeze(0)
guess = torch.load("../../ddpm/testing/results/initial0_GaussianNoiseBinaryMaskGenerator(threshold=-1, mean=0.0, std=1.0)_resample5_num_lines_0.pt", map_location='cpu', weights_only=False)
mask = torch.load("../../ddpm/testing/results/mask0_GaussianNoiseBinaryMaskGenerator(threshold=-1, mean=0.0, std=1.0)_resample5_num_lines_0.pt", map_location='cpu', weights_only=False)

print(f"Average MSE per pixel over masked area in crop: {calculate_mse(actual,guess,mask):.6f}")
save_mse_heatmap(actual,guess,mask)
save_angular_error_heatmap(guess, actual, mask)