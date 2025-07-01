import os
import sys
import csv
import torch
import imageio
import numpy as np
import matplotlib

from ddpm.helper_functions.compute_divergence import compute_divergence

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_vector_field_tool import plot_vector_field, make_heatmap
from noising_process.incompressible_gp.adding_noise.divergence_free_noise import layered_div_free_noise, gaussian_each_step_divergence_free_noise

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './../')))

"""
Generate a circular (rotational) vector field at each time step.
Vector at (x, y) is perpendicular to radius vector from center.
"""
Y, X = np.meshgrid(np.linspace(-1, 1, 94), np.linspace(-1, 1, 44), indexing='ij')
radius = np.sqrt(X**2 + Y**2) + 1e-6  # avoid div-by-zero

X = X / radius
Y = Y / radius

u = torch.tensor(X)
v = torch.tensor(Y)

for layers in torch.linspace(1,50, 50):
    tensor = gaussian_each_step_divergence_free_noise((1,2,100,100), torch.tensor([layers]), device=torch.device('cpu'))
    divergence = compute_divergence(tensor[0][0], tensor[0][1])
    mean, std, var = torch.mean(tensor), torch.std(tensor), torch.var(tensor)

    tensor = (tensor - mean) / std

    plot_vector_field(tensor[0][0], tensor[0][1], scale=60, file=f"{layers}.png", title=f"{layers}.png")
    make_heatmap(divergence, save_path=f"{layers}_div.png", title=f"{layers}_div.png")

gaussian = torch.randn((2, 100, 100))
div_gaussian = compute_divergence(gaussian[0], gaussian[1])

plot_vector_field(gaussian[0], gaussian[1], scale=60, file=f"gaussian.png", title=f"gaussian.png")
make_heatmap(div_gaussian, save_path=f"gaussian_div.png", title=f"gaussian_div.png")