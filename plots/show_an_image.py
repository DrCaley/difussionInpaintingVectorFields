import os
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio
from plot_data_tool import plot_vector_field

# Setup paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './../')))
from data_prep.ocean_image_dataset import OceanImageDataset
from data_prep.data_initializer import DDInitializer
from noising_process.incompressible_gp.adding_noise.compute_divergence import compute_divergence

# Initialize data
data_init = DDInitializer()

# Directory to save images
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')

# CSV
csv_file = os.path.join(output_dir, f"divergences.csv")
with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Number', 'Divergence'])

# I've lost the plot
plot_file = os.path.join(output_dir, f"div_plot.png")


# Generate vector field images
filenames = []
heatmap_filenames = []



fig, axs = plt.subplots(1, 2, figsize=(10, 4))

# Initialize the heat map images with the first frame
im0 = axs[0].imshow(data_init.training_tensor[:, :, 0, 890].numpy(), cmap='viridis')
axs[0].set_title(f'X-Component t=0')
cbar0 = fig.colorbar(im0, ax=axs[0])

im1 = axs[1].imshow(data_init.training_tensor[:, :, 1, 890].numpy(), cmap='viridis')
axs[1].set_title(f'Y-Component t=0')
cbar1 = fig.colorbar(im1, ax=axs[1])

for ax in axs:
    ax.set_xlabel('X')
    ax.set_ylabel('Y')



divergences = []

# IMAGE GENERATION LOOP
for i in range(100):
    tensor_to_draw_x = data_init.training_tensor[:, :, 0, i+500]
    tensor_to_draw_y = data_init.training_tensor[:, :, 1, i+500]

    # Create vector field files
    filename = os.path.join(output_dir, f"vector_field{i:03}.png")
    plot_vector_field(tensor_to_draw_x, tensor_to_draw_y, step = 2, scale=6, title=f"Vector Field {i}", file=filename)
    filenames.append(filename)

    # Heat map gifs
    
    im0.set_data(tensor_to_draw_x.numpy())
    axs[0].set_title(f'X-Component t={i}')
    im1.set_data(tensor_to_draw_y.numpy())
    axs[1].set_title(f'Y-Component t={i}')

    heatmap_file = os.path.join(output_dir, f"xy_heatmap_frame{i:03}.png")
    plt.savefig(heatmap_file, dpi=150, bbox_inches='tight')
    heatmap_filenames.append(heatmap_file)

    divergences.append(compute_divergence(tensor_to_draw_x, tensor_to_draw_y).nanmean().item())

    with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([i + 1, divergences[i]])


# Create GIFS
images = [imageio.imread(f) for f in sorted(filenames)]
imageio.mimsave(output_dir + "\\vector_fields.gif", images, fps=10)  # Adjust fps as needed

heat_images = [imageio.imread(f) for f in heatmap_filenames]
imageio.mimsave(output_dir + "\\heat_map.gif", heat_images, fps=10)


# No more 1000+ image pushes :(
for f in filenames + heatmap_filenames:
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