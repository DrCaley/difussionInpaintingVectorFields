import os
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio
from plot_vector_field_tool import plot_vector_field

# Setup paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './../')))
from data_prep.data_initializer import DDInitializer
from ddpm.helper_functions.compute_divergence import compute_divergence

# Types of noise to test
from noising_process.incompressible_gp.adding_noise.divergence_free_noise import exact_div_free_field_from_stream, gaussian_each_step_divergence_free_noise, generate_div_free_noise, layered_div_free_noise



# Initialize data
data_init = DDInitializer()

# Directory to save images
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')

# CSV
csv_file = os.path.join(output_dir, f"divergences_.csv")
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Number', 'Divergence'])

# I've lost the plot
plot_file = os.path.join(output_dir, f"div_plot_divergence_free_noise.png")


# Generate vector field images
filenames = []
divergences = []

tensor_to_draw_x = data_init.validation_tensor[:, :, 0, 0]
tensor_to_draw_y = data_init.validation_tensor[:, :, 1, 0]

# IMAGE GENERATION LOOP
for i in range(500):
    vx, vy = gaussian_each_step_divergence_free_noise(94, 44, 1)
    tensor_to_draw_x = vx
    tensor_to_draw_y = vy

    # Create vector field files
    filename = os.path.join(output_dir, f"vector_field{i:04}.png")
    plot_vector_field(tensor_to_draw_x, tensor_to_draw_y, step = 2, scale=5, title=f"Vector Field {i}", file=filename)
    filenames.append(filename)

    divergences.append(compute_divergence(tensor_to_draw_x, tensor_to_draw_y).nanmean().item())

    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([i + 1, divergences[i]])

# Create GIFS
images = [imageio.imread(f) for f in sorted(filenames)]
imageio.mimsave(output_dir + "\\vector_fields_divergence_free_noise.gif", images, fps=15)  # Adjust fps as needed

# No more 1000+ image pushes :(
for f in filenames:
    try:
        os.remove(f)
    except OSError as e:
        print(f"Error deleting file {f}: {e}")


# Make the plot
plt.figure(figsize=(20, 10))
plt.plot(divergences, label='Divergence')
plt.xlabel('Field')
plt.ylabel('Divergence')
plt.legend()
plt.title('Fields and Divergences')
plt.savefig(plot_file)