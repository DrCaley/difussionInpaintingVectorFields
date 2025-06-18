import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_vector_field_tool import plot_vector_field

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './../')))
from data_prep.data_initializer import DDInitializer
from ddpm.helper_functions.compute_divergence import compute_divergence
from ddpm.utils.noise_utils import DivergenceFreeNoise

# Initialize data and output dir
data_init = DDInitializer()
output_dir = 'noise_images'
os.makedirs(output_dir, exist_ok=True)

# Extract vector components as PyTorch tensors
tensor_to_draw_x = data_init.training_tensor[:, :, 0, 0]
tensor_to_draw_y = data_init.training_tensor[:, :, 1, 0]

if isinstance(tensor_to_draw_x, np.ndarray):
    tensor_to_draw_x = torch.from_numpy(tensor_to_draw_x)
    tensor_to_draw_y = torch.from_numpy(tensor_to_draw_y)

# Rebuild full field tensor in shape (1, 2, H, W) for the noise generator
vec_field_tensor = torch.stack([tensor_to_draw_x, tensor_to_draw_y], dim=0).unsqueeze(0)  # (1, 2, H, W)

# Generate divergence-free noise
noise_gen = DivergenceFreeNoise()
t = torch.tensor([500])  # Dummy timestep
noise = noise_gen.generate(vec_field_tensor.shape, t).squeeze(0)  # (2, H, W)

# Plotting
plot_vector_field(tensor_to_draw_x, tensor_to_draw_y, scale=10, file=os.path.join(output_dir, f"vector_field.png"))

plot_vector_field(noise[0], noise[1], scale=10, file=os.path.join(output_dir, f"noise_field.png"))