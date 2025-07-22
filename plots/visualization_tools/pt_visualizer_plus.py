import matplotlib

matplotlib.use('Agg')

import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from data_prep.data_initializer import DDInitializer
from plots.visualization_tools.error_visualization import save_mse_heatmap, save_angular_error_heatmap, \
    save_scaled_error_vectors_scalar_field, save_percent_heatmap
from ddpm.utils.inpainting_utils import calculate_mse

class PTVisualizer():
    def __init__(self,
                 mask_type ="",
                 sample_num = 83,
                 vector_scale = 0.15,
                 num_lines = 0.25,
                 resamples = 5,
                 results_dir="../../ddpm/testing/results"):

        self.noise_type = mask_type
        self.sample_num = sample_num
        self.vector_scale = vector_scale
        self.num_lines = num_lines
        self.resamples = resamples
        self.base_path = results_dir
        save_dir = f"{results_dir}/pt_visualizer_images/"
        self.prediction_path = f"{save_dir}pt_predictions/"
        self.error_path = f"{save_dir}pt_errors/"
        self.prefixes = ['ddpm', 'initial', 'mask', 'gp_field']

    def build_filename(self, prefix):
        return f"{prefix}{self.sample_num}_{self.noise_type}_resample{self.resamples}_num_lines_{self.num_lines}.pt"

    def visualize_tensor(
            self,
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
            arr = np.flipud(tensor.cpu().numpy())
            plt.imshow(arr, cmap='gray')
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
                plt.quiver(x, y, u.cpu(), v.cpu(), scale=1.0 / vector_scale)
                plt.title(title + " (Vector Field)")
                plt.gca().set_aspect('equal', adjustable='box')
                plt.savefig(os.path.join(save_dir, f"{title}_vector_field.png"))
                plt.close()
            else:
                tensor = crop(tensor)
                for i in range(tensor.shape[0]):
                    plt.imshow(tensor[i].cpu(), cmap='gray')
                    plt.gca().invert_yaxis()
                    plt.title(f"{title} - Channel {i}")
                    plt.colorbar()
                    plt.savefig(os.path.join(save_dir, f"{title}_channel_{i}.png"))
                    plt.close()

        elif tensor.ndim == 4:
            print(f"Tensor has shape {tensor.shape}, saving first item in batch.")
            self.visualize_tensor(
                tensor[0],
                title=title + "_sample_0",
                save_dir=save_dir,
                height_range=height_range,
                width_range=width_range,
                vector_scale=vector_scale
                )

        else:
            print("Unsupported tensor shape:", tensor.shape)

    def load_and_visualize_pt(self, file_path, title="loaded_tensor", save_dir="pt_visualizer_images", **kwargs):
        tensor = torch.load(file_path, map_location='cpu', weights_only=False)
        self.visualize_tensor(tensor, title=title, save_dir=save_dir, **kwargs)
        return tensor

    def visualize(self):
        self.data = {}
        save_dir = self.prediction_path
        for prefix in self.prefixes:
            file_path = os.path.join(self.base_path, self.build_filename(prefix))
            title = self.build_filename(prefix).replace('.pt', '')
            try:
                if prefix == 'mask':
                    tensor = torch.load(file_path, map_location='cpu', weights_only=False)
                    self.visualize_tensor(tensor[0, 0], title, save_dir=save_dir)
                else:
                    tensor = self.load_and_visualize_pt(file_path, title=title, save_dir=save_dir, vector_scale=self.vector_scale)
                self.data[prefix] = tensor
            except Exception as e:
                print(f"Failed to load or visualize {title}: {e}")

    def calc(self):
        data = self.data
        mask_tensor = data.get("mask", None)
        initial_tensor = data.get("initial", None)
        os.makedirs(self.error_path, exist_ok=True)
        if mask_tensor is not None and initial_tensor is not None:
            for key in data:
                if key in ("initial", "mask"):
                    continue
                try:
                    tensor = data[key]
                    save_mse_heatmap(tensor, initial_tensor, mask_tensor, title=f"{key}_vs_initial",
                                     save_path=f"{self.error_path}/{self.noise_type}{self.num_lines}_mse_{key}_vs_initial.png")
                    save_angular_error_heatmap(initial_tensor, tensor, mask_tensor, title=f"{key}_vs_initial",
                                               save_path=f"{self.error_path}/{self.noise_type}{self.num_lines}_angular_{key}_vs_initial.png")
                    save_scaled_error_vectors_scalar_field(tensor, initial_tensor, mask_tensor, title=f"{key}_vs_initial",
                                     save_path=f"{self.error_path}/{self.noise_type}{self.num_lines}_vector_{key}_vs_initial.png")
                    save_percent_heatmap(initial_tensor, tensor, mask_tensor, title=f"{key}_vs_initial",
                                         save_path=f"{self.error_path}/{self.noise_type}{self.num_lines}_PE_{key}_vs_initial.png")
                except Exception as e:
                    print(f"Failed to compute/save errors between {key} and initial: {e}")
        else:
            print("Missing 'initial' or 'mask' tensor — skipping error comparisons.")