import os
import sys
from plot_data_tool import plot_vector_field

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './../')))
from data_prep.ocean_image_dataset import OceanImageDataset
from data_prep.data_initializer import DDInitializer
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.image as mpimg


data_init = DDInitializer()

for i in range(100):
    tensor_to_draw_x = data_init.training_tensor[:,:,0,0]
    tensor_to_draw_y = data_init.training_tensor[:,:,1,0]
    plot_vector_field(tensor_to_draw_x, tensor_to_draw_y, scale=25, file = f"vector_field{i}.png")

fig, ax = plt.subplots()
img = mpimg.imread('vector_field0.png')
im = ax.imshow(img)

def update(frame):
    im.set_array(mpimg.imread(f'vector_field{frame}.png'))
    ax.set_title(f"Frame {frame}")
    return [im]

ani = animation.FuncAnimation(fig, update, frames=100, interval=100)

# Save as video
ani.save('vector_field_animation.mp4', fps=10)

# Show
plt.show()