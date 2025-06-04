import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from perlin_noise import PerlinNoise

# === Gaussian smoothing utilities ===
def gaussian_kernel(kernel_size=5, sigma=1.0):
    ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel

def apply_gaussian_blur(field, kernel):
    field = field.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    kernel = kernel.to(field.device).unsqueeze(0).unsqueeze(0)  # (1, 1, k, k)
    blurred = F.conv2d(field, kernel, padding=kernel.shape[-1] // 2)
    return blurred.squeeze()

# === Gradient and divergence ===
def compute_gradient_2d(field):
    grad_x = torch.zeros_like(field)
    grad_y = torch.zeros_like(field)
    grad_x[:-1, :] = field[1:, :] - field[:-1, :]
    grad_y[:, :-1] = field[:, 1:] - field[:, :-1]
    return grad_x, grad_y

def compute_divergence_2d(vx, vy):
    div = torch.zeros_like(vx)
    div[:-1, :] += vx[1:, :] - vx[:-1, :]
    div[:, :-1] += vy[:, 1:] - vy[:, :-1]
    return div

# === Create 2D grid ===
def create_grid(size):
    coords = torch.linspace(-1, 1, size)
    x, y = torch.meshgrid(coords, coords, indexing='ij')
    return x, y

# === Generate φ and ψ using perlin-noise (pure Python) ===
def generate_perlin_noise_fields(x, y, scale=4.0):
    noise1 = PerlinNoise(octaves=3)
    noise2 = PerlinNoise(octaves=3, seed=42)
    shape = x.shape
    phi = torch.zeros_like(x)
    psi = torch.zeros_like(x)
    for i in range(shape[0]):
        for j in range(shape[1]):
            xi = x[i, j].item() * scale
            yi = y[i, j].item() * scale
            phi[i, j] = noise1([xi, yi])
            psi[i, j] = noise2([xi, yi])
    return phi, psi

# === Main logic ===
size = 128
x, y = create_grid(size)
phi, psi = generate_perlin_noise_fields(x, y, scale=6.0)

# Optional: smooth φ with Gaussian
kernel = gaussian_kernel(kernel_size=7, sigma=1.5)
phi_smooth = apply_gaussian_blur(phi, kernel)

# Compute gradients
grad_phi_x, grad_phi_y = compute_gradient_2d(phi_smooth)
grad_psi_x, grad_psi_y = compute_gradient_2d(psi)

# Construct divergence-free vector field
v_x = grad_phi_y * grad_psi_y - grad_phi_x * grad_psi_x
v_y = grad_phi_x * grad_psi_y - grad_phi_y * grad_psi_x

# Compute divergence to check
div = compute_divergence_2d(v_x, v_y)
print("Mean divergence:", div.abs().mean().item())

# Normalize vectors for display
magnitude = torch.sqrt(v_x**2 + v_y**2) + 1e-8
v_x_norm = v_x / magnitude
v_y_norm = v_y / magnitude

# === Visualize ===
skip = 4
plt.figure(figsize=(8, 8))
plt.quiver(
    x[::skip, ::skip], y[::skip, ::skip],
    v_x_norm[::skip, ::skip], v_y_norm[::skip, ::skip],
    angles='xy', scale_units='xy', scale=20,
    width=0.005, color='black'
)
plt.title("2D Divergence-Free Noise Field (Perlin-noise pure Python)")
plt.axis('equal')
plt.grid(True)
plt.show()
