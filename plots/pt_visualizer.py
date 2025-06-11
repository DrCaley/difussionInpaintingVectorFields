import matplotlib
matplotlib.use('Agg')

import torch
import matplotlib.pyplot as plt
import numpy as np
import os


def visualize_tensor(tensor, title="tensor", save_dir="tensor_images"):
    """
    Visualizes and saves 2D/3D/4D tensors (e.g. masks, vector fields, grayscale images).
    """
    tensor = tensor.squeeze()
    os.makedirs(save_dir, exist_ok=True)

    if tensor.ndim == 2:
        # Single 2D array (e.g., mask)
        plt.imshow(tensor.cpu(), cmap='gray')
        plt.title(title)
        plt.colorbar()
        plt.savefig(os.path.join(save_dir, f"{title}.png"))
        plt.close()

    elif tensor.ndim == 3:
        if tensor.shape[0] == 2:
            # Vector field: tensor shape (2, H, W)
            u, v = tensor[0], tensor[1]
            H, W = u.shape
            x, y = np.meshgrid(np.arange(W), np.arange(H))
            plt.figure(figsize=(6, 6))
            plt.quiver(x, y, u.cpu(), v.cpu())
            plt.title(title + " (Vector Field)")
            plt.gca().invert_yaxis()
            plt.savefig(os.path.join(save_dir, f"{title}_vector_field.png"))
            plt.close()
        else:
            # Grayscale channels: save each as separate image
            for i in range(tensor.shape[0]):
                plt.imshow(tensor[i].cpu(), cmap='gray')
                plt.title(f"{title} - Channel {i}")
                plt.colorbar()
                plt.savefig(os.path.join(save_dir, f"{title}_channel_{i}.png"))
                plt.close()

    elif tensor.ndim == 4:
        print(f"Tensor has shape {tensor.shape}, saving first item in batch.")
        visualize_tensor(tensor[0], title=title + "_sample_0", save_dir=save_dir)

    else:
        print("Unsupported tensor shape:", tensor.shape)

def load_and_visualize_pt(file_path, title="loaded_tensor", save_dir="tensor_images"):
    """
    Loads a .pt file and visualizes (saves) the tensor it contains.
    """
    tensor = torch.load(file_path, map_location='cpu')
    print("Loaded tensor shape:", tensor.shape)
    visualize_tensor(tensor, title=title, save_dir=save_dir)

# Use here:
load_and_visualize_pt('../ddpm/testing/results/predicted/RandomPath_10.pt',
                      title="RandomPath_10")
