import os
import sys
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from plots.plot_data_tool import plot_vector_field
from ddpm.helper_functions.compute_divergence import compute_divergence



# === Setup ===
output_dir = os.path.relpath('plots/outputs')
os.makedirs(output_dir, exist_ok=True)

# Grid setup
Nx, Ny = 64, 64
Lx, Ly = 2 * np.pi, 2 * np.pi
dx, dy = Lx / Nx, Ly / Ny

x = np.linspace(0, Lx, Nx, endpoint=False)
y = np.linspace(0, Ly, Ny, endpoint=False)
X, Y = np.meshgrid(x, y, indexing='ij')


# Another divergence to use
def compute_divergence1(u, v, dx):
    dudx = np.gradient(u, dx, axis=0)
    dvdy = np.gradient(v, dx, axis=1)
    return dudx + dvdy

csv_file = os.path.join(output_dir, "testing_HH_decomps.csv")
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Number', 'OG Divergence', 'IRR Divergence', 'Divless Divergence', 'IRR Divergence 1', 'Divless Divergence 1'])

    for run in range(1000):
        # === Random Vector Field Generation ===
        def generate_random_vector_field(Nx, Ny):
            def rand_smooth_field():
                fx = np.random.randn(Nx, Ny)
                fy = np.random.randn(Nx, Ny)
                f_hat = np.fft.fft2(fx + 1j * fy)
                kx = np.fft.fftfreq(Nx).reshape(-1, 1)
                ky = np.fft.fftfreq(Ny).reshape(1, -1)
                decay = np.exp(-4 * (kx**2 + ky**2))  # low-pass
                f_hat *= decay
                f = np.fft.ifft2(f_hat).real
                return f.astype(np.float32)

            u = rand_smooth_field()
            v = rand_smooth_field()
            return u, v


        u_np, v_np = generate_random_vector_field(Nx, Ny)
        u_tensor = torch.from_numpy(u_np)
        v_tensor = torch.from_numpy(v_np)

        # Mask (simulate land)
        mask = np.ones((Nx, Ny), dtype=bool)
        mask[Nx//3:Nx//2, Ny//3:Ny//2] = 0  # land region


        # Compute divergences (PyTorch version)
        divF_tensor = compute_divergence(u_tensor, v_tensor)
        divF_tensor[~torch.from_numpy(mask)] = 0.0
        divF = divF_tensor.numpy()  # convert back to NumPy for spsolve

        divF_tensor1 = compute_divergence1(u_tensor, v_tensor, dx)
        divF_tensor1[~torch.from_numpy(mask)] = 0.0
        divF1 = divF_tensor1

        # === Poisson Solve for φ ===
        idx_map = -np.ones_like(mask, dtype=int)
        idx_map[mask] = np.arange(np.sum(mask))
        N_unknowns = np.sum(mask)

        A = lil_matrix((N_unknowns, N_unknowns))
        b = np.zeros(N_unknowns)
        b1 = np.zeros(N_unknowns)

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
                b[index] = divF[i, j]
                b1[index] = divF1[i, j]

        phi_flat = spsolve(A.tocsr(), b)
        phi = np.zeros_like(u_np)
        phi[mask] = phi_flat

        phi_flat1 = spsolve(A.tocsr(), b1)
        phi1 = np.zeros_like(u_np)
        phi1[mask] = phi_flat1

        # === Compute ∇φ ===
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

        grad_phi_x_np, grad_phi_y_np = masked_gradient(phi, dx, dy, mask)
        grad_phi_x_tensor = torch.from_numpy(grad_phi_x_np).float()
        grad_phi_y_tensor = torch.from_numpy(grad_phi_y_np).float()

        grad_phi_x_np1, grad_phi_y_np1 = masked_gradient(phi, dx, dy, mask)
        grad_phi_x_tensor1 = torch.from_numpy(grad_phi_x_np1).float()
        grad_phi_y_tensor1 = torch.from_numpy(grad_phi_y_np1).float()

        # === HH Decomposition ===
        u_irr = grad_phi_x_tensor
        v_irr = grad_phi_y_tensor
        u_sol = u_tensor - u_irr
        v_sol = v_tensor - v_irr

        u_irr1 = grad_phi_x_tensor1
        v_irr1 = grad_phi_y_tensor1
        u_sol1 = u_tensor - u_irr1
        v_sol1 = v_tensor - v_irr1

        # === Plots ===
        plot_vector_field(u_tensor * torch.from_numpy(mask), v_tensor * torch.from_numpy(mask), step=2, scale=5,
                        title="OG Vector Field", file=os.path.join(output_dir, "vector_field_TEST_OG.png"))
        plot_vector_field(u_irr * torch.from_numpy(mask), v_irr * torch.from_numpy(mask), step=2, scale=5,
                        title="Irr Vector Field", file=os.path.join(output_dir, "vector_field_TEST_IRR.png"))
        plot_vector_field(u_sol * torch.from_numpy(mask), v_sol * torch.from_numpy(mask), step=2, scale=5,
                        title="Div-Free Vector Field", file=os.path.join(output_dir, "vector_field_TEST_DIVLESS.png"))

        plot_vector_field(u_irr1 * torch.from_numpy(mask), v_irr1 * torch.from_numpy(mask), step=2, scale=5,
                        title="Irr Vector Field 1", file=os.path.join(output_dir, "vector_field_TEST_IRR1.png"))
        plot_vector_field(u_sol1 * torch.from_numpy(mask), v_sol1 * torch.from_numpy(mask), step=2, scale=5,
                        title="Div-Free Vector Field 1", file=os.path.join(output_dir, "vector_field_TEST_DIVLESS1.png"))

        # TESTING DIVERGENCES OF FIELDS
        writer.writerow([run,compute_divergence(u_tensor * torch.from_numpy(mask),v_tensor * torch.from_numpy(mask)).nanmean().item(),compute_divergence(u_irr * torch.from_numpy(mask), v_irr * torch.from_numpy(mask)).nanmean().item(),compute_divergence(u_sol * torch.from_numpy(mask), v_sol * torch.from_numpy(mask)).nanmean().item(), compute_divergence(u_irr1 * torch.from_numpy(mask), v_irr1 * torch.from_numpy(mask)).nanmean().item(), compute_divergence(u_sol1 * torch.from_numpy(mask), v_sol1 * torch.from_numpy(mask)).nanmean().item()])