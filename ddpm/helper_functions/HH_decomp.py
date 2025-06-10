import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftfreq

# Create a 2D grid
N = 64
L = 2 * np.pi
x = np.linspace(0, L, N, endpoint=False)
y = np.linspace(0, L, N, endpoint=False)
X, Y = np.meshgrid(x, y, indexing='ij')

# Define a sample vector field F = (u, v)
# This one has both divergence-free and irrotational parts
u = -np.sin(Y)
v = np.sin(X)

# Compute divergence: ∇ · F = ∂u/∂x + ∂v/∂y
def compute_divergence(u, v, dx):
    dudx = np.gradient(u, dx, axis=0)
    dvdy = np.gradient(v, dx, axis=1)
    return dudx + dvdy

# Solve Poisson's equation in Fourier domain: Δϕ = div(F)
def solve_poisson_fft(rhs, dx):
    N = rhs.shape[0]
    k = fftfreq(N, d=dx) * 2 * np.pi
    kx, ky = np.meshgrid(k, k, indexing='ij')
    k2 = kx**2 + ky**2
    k2[0, 0] = 1.0  # avoid division by zero

    rhs_hat = fft2(rhs)
    phi_hat = rhs_hat / (-k2)
    phi_hat[0, 0] = 0  # zero-mean potential

    phi = np.real(ifft2(phi_hat))
    return phi

# Compute gradient of scalar potential ϕ: ∇ϕ = (∂ϕ/∂x, ∂ϕ/∂y)
def compute_gradient(phi, dx):
    dphidx = np.gradient(phi, dx, axis=0)
    dphidy = np.gradient(phi, dx, axis=1)
    return dphidx, dphidy

# Main computation
dx = L / N
divF = compute_divergence(u, v, dx)
phi = solve_poisson_fft(divF, dx)
grad_phi_x, grad_phi_y = compute_gradient(phi, dx)

# Decomposition:
u_irr = grad_phi_x
v_irr = grad_phi_y

u_sol = u - u_irr
v_sol = v - v_irr

# Plotting
def plot_vector_field(u, v, title):
    plt.figure(figsize=(5, 5))
    plt.quiver(X, Y, u, v)
    plt.title(title)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

plot_vector_field(u, v, "Original Field F")
plot_vector_field(u_irr, v_irr, "Irrotational Component ∇ϕ")
plot_vector_field(u_sol, v_sol, "Divergence-Free Component")