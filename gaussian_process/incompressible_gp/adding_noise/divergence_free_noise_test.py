import torch
import matplotlib.pyplot as plt

def compute_divergence(vx, vy):
    """
    Computes the discrete divergence of a 2D vector field at every point.
    Uses central differences (zero padding at the borders).
    """
    H, W = vx.shape

    # Initialize divergence
    divergence = torch.zeros_like(vx)

    # Central difference for interior points
    dvx_dx = torch.zeros_like(vx)
    dvy_dy = torch.zeros_like(vy)

    dvx_dx[1:-1, :] = (vx[2:, :] - vx[:-2, :]) / 2.0
    dvy_dy[:, 1:-1] = (vy[:, 2:] - vy[:, :-2]) / 2.0

    divergence = dvx_dx + dvy_dy
    return divergence


# === Example ===
# Create a divergence-free field from a stream function
H, W = 128, 128
x = torch.linspace(0, 2 * torch.pi, W)
y = torch.linspace(0, 2 * torch.pi, H)
X, Y = torch.meshgrid(x, y, indexing='ij')
phase_x, phase_y = 2 * torch.pi * torch.rand(2)

# Stream function
psi = torch.sin(3 * X + phase_x) * torch.sin(3 * Y + phase_y)

# vx = ∂ψ/∂y, vy = -∂ψ/∂x (forward difference)
vx = torch.zeros_like(psi)
vy = torch.zeros_like(psi)
vx[:, :-1] = psi[:, 1:] - psi[:, :-1]
vy[:-1, :] = -(psi[1:, :] - psi[:-1, :])

# Compute divergence across all points
div = compute_divergence(vx, vy)

# Print summary stats
print("Mean absolute divergence:", div.abs().mean().item())
print("Max absolute divergence:", div.abs().max().item())

# 🔍 Visualize divergence across all points
plt.imshow(div.T, cmap='bwr', origin='lower')
plt.colorbar(label='Divergence')
plt.title('Divergence at Each Point')
plt.show()
