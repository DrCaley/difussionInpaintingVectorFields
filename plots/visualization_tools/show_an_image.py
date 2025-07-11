import os
import sys
import csv
import torch
import imageio
import numpy as np
import matplotlib
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from ddpm.helper_functions.compute_divergence import compute_divergence

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data_prep.data_initializer import DDInitializer
from data_prep import spliting_data_sets
from plot_vector_field_tool import plot_vector_field, make_heatmap
from noising_process.incompressible_gp.adding_noise.divergence_free_noise import layered_div_free_noise, gaussian_each_step_divergence_free_noise



# I broke it, it's not needed rn -Matt


# Initialize data
data_init = DDInitializer()

tensor_to_draw_x = data_init.training_tensor[:, :, 0, 500]
tensor_to_draw_y = data_init.training_tensor[:, :, 1, 500]



field_files = []

for i in range(20):
    plot_vector_field(tensor_to_draw_x, tensor_to_draw_y, scale=5)

# Create GIFS
imageio.mimsave("Fields.gif", field_files, fps=5)  # Adjust fps as needed

# Remove images
for f in field_files:
    try:
        os.remove(f)
    except OSError as e:
        print(f"Error deleting file {f}: {e}")