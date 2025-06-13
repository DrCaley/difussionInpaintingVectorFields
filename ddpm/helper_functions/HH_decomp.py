import os
import sys
import torch
import numpy as np
from scipy.fft import fft2, ifft2, fftfreq

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from plots.plot_vector_field_tool import plot_vector_field
from ddpm.helper_functions.compute_divergence import compute_divergence
from ddpm.helper_functions.standardize_data import standardize_data

# === Create a non-square 2D grid ===
Nx, Ny = 6, 6  # Can be any positive integers
Lx, Ly = 2 * np.pi, 2 * np.pi
x = np.linspace(0, Lx, Nx, endpoint=False)
y = np.linspace(0, Ly, Ny, endpoint=False)
X, Y = np.meshgrid(x, y, indexing='ij')

dx = Lx / Nx
dy = Ly / Ny

# === Define a sample vector field F = (u, v) ===
# This one has both divergence-free and irrotational parts
u = -np.sin(Y)          # u(x, y)
v = np.sin(X)           # v(x, y)

u = torch.from_numpy(u).float()
v = torch.from_numpy(v).float()

# === Solve Poisson equation in Fourier space: Δϕ = ∇ · F ===
def solve_poisson_fft(rhs, dx, dy):
    Nx, Ny = rhs.shape
    kx = fftfreq(Nx, d=dx) * 2 * np.pi
    ky = fftfreq(Ny, d=dy) * 2 * np.pi
    kx, ky = np.meshgrid(kx, ky, indexing='ij')
    k2 = kx**2 + ky**2
    k2[0, 0] = 1.0  # Avoid division by zero at the zero frequency

    rhs_hat = fft2(rhs)
    phi_hat = rhs_hat / (-k2)
    phi_hat[0, 0] = 0  # Remove mean of potential
    phi = np.real(ifft2(phi_hat))
    return phi

# === Compute gradient ∇ϕ ===
def compute_gradient(phi, dx, dy):
    dphidx = np.gradient(phi, dx, axis=0)
    dphidy = np.gradient(phi, dy, axis=1)
    return dphidx, dphidy

# === Helmholtz-Hodge Decomposition ===
divF = compute_divergence(u, v)
phi = solve_poisson_fft(divF, dx, dy)
grad_phi_x, grad_phi_y = compute_gradient(phi, dx, dy)

u_irr = grad_phi_x
v_irr = grad_phi_y

u_sol = u - u_irr
v_sol = v - v_irr

# === Convert to PyTorch tensors ===
u_irr = torch.from_numpy(u_irr).float()
v_irr = torch.from_numpy(v_irr).float()

og_total = torch.stack([u, v])
irr_total = torch.stack([u_irr, v_irr])
sol_total = torch.stack([u_sol, v_sol])



plot_vector_field(og_total[0], og_total[1], scale=10, title="Original Field F", file="original_vector_field.png")
plot_vector_field(irr_total[0], irr_total[1], scale=10, title="Irrotational Component ∇ϕ", file="irr_vector_field.png")
plot_vector_field(sol_total[0], sol_total[1], scale=10, title="Divergence-Free Component", file="sol_vector_field.png")


print("Original field divergence: " + str(compute_divergence(og_total[0], og_total[1]).mean()))
print("Irrotational field divergence: " + str(compute_divergence(irr_total[0], irr_total[1]).mean()))
print("Solenoidal field divergence: " + str(compute_divergence(sol_total[0], sol_total[1]).mean()))
print("\n\n\n")

OGstandardize = standardize_data(u.mean(), u.std(), v.mean(), v.std())
IRRstandardize = standardize_data(u_irr.mean(), u_irr.std(), v_irr.mean(), v_irr.std())
SOLstandardize = standardize_data(u_sol.mean(), u_sol.std(), v_sol.mean(), v_sol.std())

og_total = OGstandardize(og_total)
irr_total = IRRstandardize(irr_total)
sol_total = SOLstandardize(sol_total)

plot_vector_field(og_total[0], og_total[1], scale=10, title="Original Field F", file="original_vector_field_stand.png")
plot_vector_field(irr_total[0], irr_total[1], scale=10, title="Irrotational Component ∇ϕ", file="irr_vector_field_stand.png")
plot_vector_field(sol_total[0], sol_total[1], scale=10, title="Divergence-Free Component", file="sol_vector_field_stand.png")

print("Original field divergence AFTER STANDARDIZING: " + str(compute_divergence(og_total[0], og_total[1]).mean()))
print("Irrotational field divergence AFTER STANDARDIZING: " + str(compute_divergence(irr_total[0], irr_total[1]).mean()))
print("Solenoidal field divergence AFTER STANDARDIZING: " + str(compute_divergence(sol_total[0], sol_total[1]).mean()))