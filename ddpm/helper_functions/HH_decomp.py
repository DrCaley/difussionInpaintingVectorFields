import os
import sys
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from data_prep.data_initializer import DDInitializer
#from ddpm.helper_functions.compute_divergence import compute_divergence
from plots.plot_data_tool import plot_vector_field



output_dir = os.path.relpath('plots\outputs')

# CSV
csv_file = os.path.join(output_dir, f"testing_HH_decomps.csv")
with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Number', 'Divergence'])

# I've lost the plot
plot_file = os.path.join(output_dir, f"div_plot_validation_tensor.png")

# Load your velocity field
data_init = DDInitializer()
u = data_init.validation_tensor[:, :, 0, 0]
v = data_init.validation_tensor[:, :, 1, 0]

# Domain shape
Nx, Ny = u.shape[0], u.shape[1]
Lx = 2 * np.pi
Ly = 2 * np.pi
dx = Lx / Nx
dy = Ly / Ny

x = np.linspace(0, Lx, Nx, endpoint=False)
y = np.linspace(0, Ly, Ny, endpoint=False)
X, Y = np.meshgrid(x, y, indexing='ij')


#Compute divergence, but only in valid region
def compute_divergence(u, v, dx):
    dudx = np.gradient(u, dx, axis=0)
    dvdy = np.gradient(v, dx, axis=1)
    return dudx + dvdy



# Mask: 1 = valid region (ocean), 0 = land
mask = np.ones((Nx, Ny), dtype=bool)
mask[Nx//3:Nx//2, :] = 0  # no mask





# This divergence stuff sucks :(
divF = compute_divergence(u, v, dx) 
divF[~mask] = 0.0



# Index map
idx_map = -np.ones_like(mask, dtype=int)
idx_map[mask] = np.arange(np.sum(mask))
N_unknowns = np.sum(mask)

# Build sparse matrix and RHS
A = lil_matrix((N_unknowns, N_unknowns))
b = np.zeros(N_unknowns)

for i in range(Nx):
    for j in range(Ny):
        if not mask[i, j]:
            continue
        index = idx_map[i, j]
        A[index, index] = -2 * (1/dx**2 + 1/dy**2)

        for di, dj, coeff in [(-1, 0, 1/dx**2), (1, 0, 1/dx**2), (0, -1, 1/dy**2), (0, 1, 1/dy**2)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < Nx and 0 <= nj < Ny and mask[ni, nj]:
                A[index, idx_map[ni, nj]] = coeff
            # else: Dirichlet boundary (phi = 0) ⇒ no action needed

        b[index] = divF[i, j]

# Solve
phi_flat = spsolve(A.tocsr(), b)
phi = np.zeros_like(u)
phi[mask] = phi_flat

# Gradient ∇ϕ
def masked_gradient(phi, dx, dy, mask):
    grad_x = np.zeros_like(phi)
    grad_y = np.zeros_like(phi)

    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            if not mask[i, j]:
                continue
            if mask[i + 1, j] and mask[i - 1, j]:
                grad_x[i, j] = (phi[i + 1, j] - phi[i - 1, j]) / (2 * dx)
            if mask[i, j + 1] and mask[i, j - 1]:
                grad_y[i, j] = (phi[i, j + 1] - phi[i, j - 1]) / (2 * dy)

    return grad_x, grad_y

grad_phi_x, grad_phi_y = masked_gradient(phi, dx, dy, mask)

# Decompose
grad_phi_x_tensor = torch.from_numpy(grad_phi_x).float()
grad_phi_y_tensor = torch.from_numpy(grad_phi_y).float()

# Irrotational component
u_irr = grad_phi_x_tensor
v_irr = grad_phi_y_tensor

# Solenoidal component
u_sol = u - u_irr
v_sol = v - v_irr

# Plotting
filename = os.path.join(output_dir, f"vector_field_TEST_OG1.png")
plot_vector_field(u * mask, v * mask, step = 2, scale=5, title=f"OG Vector Field", file=filename)
filename = os.path.join(output_dir, f"vector_field_TEST_IRR1.png")
plot_vector_field(u_irr * mask, v_irr * mask, step = 2, scale=5, title=f"Irr Vector Field", file=filename)
filename = os.path.join(output_dir, f"vector_field_TEST_DIVLESS1.png")
plot_vector_field(u_sol * mask, v_sol * mask, step = 2, scale=5, title=f"Divless Vector Field", file=filename)