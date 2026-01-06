#!/usr/bin/env python
"""
Quick test of vector field visualization using dummy data.
This verifies the plotting code works without running expensive inpainting.
"""
import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import torch
import matplotlib.pyplot as plt
import numpy as np

# Create dummy data that looks like ocean currents
np.random.seed(42)
H, W = 44, 94

# Create a simple flow pattern
Y, X = np.mgrid[0:H, 0:W]
# Circular-ish flow
orig_u = -0.1 * (Y - H/2) / H + 0.05 * np.random.randn(H, W)
orig_v = 0.1 * (X - W/2) / W + 0.05 * np.random.randn(H, W)

# Create a vertical stripe mask in the middle
mask_np = np.zeros((H, W))
mask_np[:, W//3:2*W//3] = 1.0

# Simulate "inpainted" results with some error
gauss_u = orig_u + 0.1 * np.random.randn(H, W) * mask_np
gauss_v = orig_v + 0.1 * np.random.randn(H, W) * mask_np

divfree_u = orig_u + 0.05 * np.random.randn(H, W) * mask_np  # Better
divfree_v = orig_v + 0.05 * np.random.randn(H, W) * mask_np

# Fake metrics
metrics_g = {'mse': 0.1222, 'angular_error': 95.1}
metrics_d = {'mse': 0.0877, 'angular_error': 45.2}

# ==================== VISUALIZATION ====================
print("Testing vector field visualization...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Subsample for cleaner arrows
skip = 3

def plot_vector_field(ax, u, v, title, mask_overlay=None):
    """Plot vector field with quiver arrows."""
    # Background: magnitude as faint color
    mag = np.sqrt(u**2 + v**2)
    ax.imshow(mag, cmap='Blues', alpha=0.3, origin='upper')
    
    # Quiver plot (subsampled)
    ax.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
              u[::skip, ::skip], -v[::skip, ::skip],  # negative v for correct orientation
              color='black', scale=3, width=0.003)
    
    # Mask boundary
    if mask_overlay is not None:
        ax.contour(mask_overlay, levels=[0.5], colors='red', linewidths=2)
    
    ax.set_title(title, fontsize=12)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)  # Flip y-axis
    ax.set_aspect('equal')
    ax.axis('off')

# Plot all four panels
plot_vector_field(axes[0, 0], orig_u, orig_v, 'Original (Ground Truth)', mask_np)

# Masked input - zero out masked region
masked_u = orig_u * (1 - mask_np)
masked_v = orig_v * (1 - mask_np)
plot_vector_field(axes[0, 1], masked_u, masked_v, 'Masked Input', mask_np)

plot_vector_field(axes[1, 0], gauss_u, gauss_v, 
                  f'Gaussian Inpainting\nMSE={metrics_g["mse"]:.4f}, Angle={metrics_g["angular_error"]:.1f}°', 
                  mask_np)

plot_vector_field(axes[1, 1], divfree_u, divfree_v, 
                  f'Div-free Inpainting\nMSE={metrics_d["mse"]:.4f}, Angle={metrics_d["angular_error"]:.1f}°', 
                  mask_np)

plt.suptitle('Vector Field Visualization Test (Dummy Data)', fontsize=14)
plt.tight_layout()

save_path = BASE_DIR / 'plots' / 'outputs' / 'test_vector_visualization.png'
save_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"Saved to: {save_path}")
plt.show()
