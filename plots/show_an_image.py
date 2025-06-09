import os
import sys
from plot_data_tool import plot_vector_field

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './../')))
from data_prep.ocean_image_dataset import OceanImageDataset
from data_prep.data_initializer import DDInitializer


data_init = DDInitializer()

for i in range(100):
    tensor_to_draw_x = data_init.training_tensor[:,:,0,0]
    tensor_to_draw_y = data_init.training_tensor[:,:,1,0]
    plot_vector_field(tensor_to_draw_x, tensor_to_draw_y, scale=25, file = f"vector_field{i}.png")