import os
import sys
import csv
import torch
import imageio
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_vector_field_tool import plot_vector_field

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './../')))
from data_prep.data_initializer import DDInitializer
from ddpm.helper_functions.compute_divergence import compute_divergence
from noising_process.incompressible_gp.adding_noise.divergence_free_noise import gaussian_divergence_free_noise, gaussian_each_step_divergence_free_noise, exact_div_free_field_from_stream



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

plot_vector_field(u, v, scale=60)