import os
import sys
import imageio
from plot_data_tool import plot_vector_field

# Setup paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './../')))
from data_prep.ocean_image_dataset import OceanImageDataset
from data_prep.data_initializer import DDInitializer

# Initialize data
data_init = DDInitializer()

# Directory to save images
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Outputs')

# Generate vector field images
filenames = []
for i in range(1000):
    tensor_to_draw_x = data_init.training_tensor[:, :, 0, i]
    tensor_to_draw_y = data_init.training_tensor[:, :, 1, i]
    filename = os.path.join(output_dir, f"vector_field{i}.png")
    plot_vector_field(tensor_to_draw_x, tensor_to_draw_y, scale=25, title=f"Vector Field {i}", file=filename)
    filenames.append(filename)

# Create GIF
images = [imageio.imread(f) for f in sorted(filenames)]
imageio.mimsave("vector_fields.gif", images, fps=30)  # Adjust fps as needed