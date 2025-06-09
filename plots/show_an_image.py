import os
import sys
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

# Initialize data
data_init = DDInitializer()

# Directory to save images
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')

# Generate vector field images
filenames = []
heatmap_filenames = []

for i in range(100):
    tensor_to_draw_x = data_init.training_tensor[:, :, 0, i+500]
    tensor_to_draw_y = data_init.training_tensor[:, :, 1, i+500]

    # Create vector field files
    filename = os.path.join(output_dir, f"vector_field{i:03}.png")
    plot_vector_field(tensor_to_draw_x, tensor_to_draw_y, step = 2, scale=15, title=f"Vector Field {i}", file=filename)
    filenames.append(filename)

    # Heat map gifs
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    im0 = axs[0].imshow(tensor_to_draw_x.numpy(), cmap='viridis')
    axs[0].set_title(f'X-Component t={i}')
    plt.colorbar(im0, ax=axs[0])
    im1 = axs[1].imshow(tensor_to_draw_y.numpy(), cmap='viridis')
    axs[1].set_title(f'Y-Component t={i}')
    plt.colorbar(im1, ax=axs[1])

    for ax in axs:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

    heatmap_file = os.path.join(output_dir, f"xy_heatmap_frame{i:03}.png")
    plt.savefig(heatmap_file, dpi=150, bbox_inches='tight')
    plt.close()    
    heatmap_filenames.append(heatmap_file)
    
    

# Create GIFS
images = [imageio.imread(f) for f in sorted(filenames)]
imageio.mimsave(output_dir + "\\vector_fields.gif", images, fps=10)  # Adjust fps as needed

heat_images = [imageio.imread(f) for f in heatmap_filenames]
imageio.mimsave(output_dir + "\\heat_map.gif", heat_images, fps=10)

