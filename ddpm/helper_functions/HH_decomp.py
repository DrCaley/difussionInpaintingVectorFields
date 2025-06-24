import os
import sys
import torch
import numpy as np
from scipy.fft import fft2, ifft2, fftfreq

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from ddpm.helper_functions.compute_divergence import compute_divergence


# === INPUT: vector field of shape (H, W, 2) ===
def decompose_vector_field(field_tensor, dx=1.0, dy=1.0):
    """
    Performs Helmholtz-Hodge decomposition on a 2D vector field.

    Args:
        field_tensor (torch.Tensor): shape (H, W, 2), where [:, :, 0] is u and [:, :, 1] is v
        dx, dy (float): grid spacing in x and y

    Returns:
        (u, v), (u_irr, v_irr), (u_sol, v_sol): original, irrotational, divergence-free components
    """
    # === 1. Extract u, v from input ===
    assert field_tensor.ndim == 3 and field_tensor.shape[2] == 2, "Expected shape (H, W, 2)"
    H, W = field_tensor.shape[:2]
    u = field_tensor[:, :, 0]
    v = field_tensor[:, :, 1]

    # === 2. Convert to numpy for FFT ===
    u_np = u.numpy()
    v_np = v.numpy()

    # === 3. Compute divergence ===
    divF = compute_divergence(torch.from_numpy(u_np), torch.from_numpy(v_np)).numpy()

    # === 4. Solve Poisson equation ===
    def solve_poisson_fft(rhs, dx, dy):
        kx = fftfreq(H, d=dx) * 2 * np.pi
        ky = fftfreq(W, d=dy) * 2 * np.pi
        kx, ky = np.meshgrid(kx, ky, indexing='ij')
        k2 = kx**2 + ky**2
        k2[0, 0] = 1.0
        rhs_hat = fft2(rhs)
        phi_hat = rhs_hat / (-k2)
        phi_hat[0, 0] = 0
        return np.real(ifft2(phi_hat))

    phi = solve_poisson_fft(divF, dx, dy)

    # === 5. Compute ∇ϕ ===
    def compute_gradient(phi, dx, dy):
        dphidx = np.gradient(phi, dx, axis=0)
        dphidy = np.gradient(phi, dy, axis=1)
        return dphidx, dphidy

    grad_phi_x, grad_phi_y = compute_gradient(phi, dx, dy)

    # === 6. Compute irrotational and solenoidal parts ===
    u_irr = grad_phi_x
    v_irr = grad_phi_y
    u_sol = u_np - u_irr
    v_sol = v_np - v_irr

    return (u_np, v_np), (u_irr, v_irr), (u_sol, v_sol)