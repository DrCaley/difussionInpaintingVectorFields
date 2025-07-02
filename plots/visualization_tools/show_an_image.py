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
from plot_vector_field_tool import plot_vector_field, make_heatmap
from noising_process.incompressible_gp.adding_noise.divergence_free_noise import layered_div_free_noise, gaussian_each_step_divergence_free_noise



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


# Important for the gifs
def load_and_standardize_image(path, size=(256, 256)):
    img = Image.open(path).convert("RGB").resize(size)
    return np.array(img)


# Gifs
div_files = []
field_files = []

for layers in torch.linspace(1,50, 50):
    tensor = gaussian_each_step_divergence_free_noise((1,2,100,100), torch.tensor([layers]), device=torch.device('cpu'))
    divergence = compute_divergence(tensor[0][0], tensor[0][1])

    frame_idx = f"{int(layers.item()):02d}"  # 2-digit zero-padded integer
    plot_vector_field(tensor[0][0], tensor[0][1], scale=60, file=f"{frame_idx}.png", title=f"{frame_idx}.png")
    field_files.append(f"{frame_idx}.png")
    make_heatmap(divergence, save_path=f"{frame_idx}_div.png", title=f"{frame_idx}_div.png")
    div_files.append(f"{frame_idx}_div.png")

gaussian = torch.randn((2, 100, 100))
div_gaussian = compute_divergence(gaussian[0], gaussian[1])

"""
plot_vector_field(gaussian[0], gaussian[1], scale=60, file=f"gaussian.png", title=f"gaussian.png")
make_heatmap(div_gaussian, save_path=f"gaussian_div.png", title=f"gaussian_div.png")
"""

# Create GIFS
images0 = [load_and_standardize_image(f) for f in sorted(div_files)]
imageio.mimsave("Divergences.gif", images0, fps=5)  # Adjust fps as needed
images = [load_and_standardize_image(f) for f in sorted(field_files)]
imageio.mimsave("Fields.gif", images, fps=5)  # Adjust fps as needed

# Remove images
for file_list in [field_files, div_files]:
    for f in file_list:
        try:
            os.remove(f)
        except OSError as e:
            print(f"Error deleting file {f}: {e}")