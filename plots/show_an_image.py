import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './../')))
from data_prep.data_initializer import DDInitializer

# Load data
data_init = DDInitializer()
tensor = data_init.training_tensor[:100]  # shape: (100, H, W, 2)
tensor_np = tensor.numpy()

# Extract u and v fields from last dimension (component axis)
u_stack = tensor_np[:, :, 0, 0]  # shape: (100, H, W)
v_stack = tensor_np[:, :, 1, 0]  # shape: (100, H, W)

H, W = u_stack.shape[1:]
x = np.arange(W)
y = np.arange(H)
X, Y = np.meshgrid(x, y)

# Setup plot
fig, ax = plt.subplots(figsize=(10, 5))
q = ax.quiver(X, Y, u_stack[0], v_stack[0], scale=25)
ax.invert_yaxis()
ax.set_title("Vector Field Frame 0")

# Update function
def update(frame):
    q.set_UVC(u_stack[frame], v_stack[frame])
    ax.set_title(f"Vector Field Frame {frame}")
    return q,

# Create animation
ani = animation.FuncAnimation(fig, update, frames=u_stack.shape[0], interval=100)

# Display
plt.show()

# Save
ani.save("vector_field.gif", writer="pillow", fps=10)