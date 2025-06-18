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

# ======================== USER INPUT ========================
noise_type = "BetterRobotPath"  # e.g. "RobotPath", "NoisyField", etc.
sample_num = 83           # Which numbered sample to visualize
vector_scale = 0.15       # Adjust for better vector field visibility
num_lines = 0
resamples = 5
# ============================================================

base_path = f"../../ddpm/testing/results"
save_dir = "pt_visualizer_images"
prefixes = ['ddpm', 'initial', 'mask', 'gp_field']

def build_filename(prefix):
    return f"{prefix}{sample_num}_{noise_type}_resample{resamples}_num_lines_{num_lines}.pt"

def visualize_tensor(
    tensor,
    title="tensor",
    save_dir="tensor_images",
    height_range=None,
    width_range=None,
    vector_scale=1.0
):
    tensor = tensor.squeeze()
    os.makedirs(save_dir, exist_ok=True)

    def crop(t):
        if height_range is not None:
            t = t[..., height_range[0]:height_range[1], :]
        if width_range is not None:
            t = t[..., :, width_range[0]:width_range[1]]
        return t

    if tensor.ndim == 2:
        tensor = crop(tensor)
        plt.imshow(tensor.cpu(), cmap='gray')
        plt.title(title)
        plt.colorbar()
        plt.savefig(os.path.join(save_dir, f"{title}.png"))
        plt.close()

    elif tensor.ndim == 3:
        if tensor.shape[0] == 2:
            tensor = crop(tensor)
            u, v = tensor[0], tensor[1]
            H, W = u.shape
            x, y = np.meshgrid(np.arange(W), np.arange(H))

            max_dim = 8
            figsize = (max_dim, max_dim * H / W) if W > H else (max_dim * W / H, max_dim)

            plt.figure(figsize=figsize)
            plt.quiver(x, y, u.cpu(), v.cpu(), scale=1.0/vector_scale)
            plt.title(title + " (Vector Field)")
            plt.gca().invert_yaxis()
            plt.gca().set_aspect('equal', adjustable='box')
            plt.savefig(os.path.join(save_dir, f"{title}_vector_field.png"))
            plt.close()
        else:
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
    tensor = torch.load(file_path, map_location='cpu', weights_only=False)
    visualize_tensor(tensor, title=title, save_dir=save_dir, **kwargs)
    return tensor

# ========================== MAIN ============================

# Load and visualize each type
data = {}
for prefix in prefixes:
    file_path = os.path.join(base_path, build_filename(prefix))
    title = build_filename(prefix).replace('.pt', '')
    try:
        if prefix == 'mask':
            tensor = torch.load(file_path, map_location='cpu', weights_only=False)
            visualize_tensor(tensor[0, 0], title, save_dir=save_dir)
        else:
            tensor = load_and_visualize_pt(file_path, title=title, save_dir=save_dir, vector_scale=vector_scale)
        data[prefix] = tensor
    except Exception as e:
        print(f"Failed to load or visualize {title}: {e}")

# Compute MSEs and angular errors comparing everything to 'initial'
mask_tensor = data.get("mask", None)
initial_tensor = data.get("initial", None)

if mask_tensor is not None and initial_tensor is not None:
    for key in data:
        if key in ("initial", "mask"):
            continue
        try:
            tensor = data[key]
            mse = calculate_mse(tensor, initial_tensor, mask_tensor)
            print(f"MSE between {key} and initial: {mse:.6f}")
            save_mse_heatmap(tensor, initial_tensor, mask_tensor, title=f"{key}_vs_initial",save_path=f"mse_{key}_vs_initial.png")
            save_angular_error_heatmap(initial_tensor, tensor, mask_tensor, title=f"{key}_vs_initial",save_path=f"angular_{key}_vs_initial.png")
        except Exception as e:
            print(f"Failed to compute/save errors between {key} and initial: {e}")
else:
    print("Missing 'initial' or 'mask' tensor — skipping error comparisons.")
