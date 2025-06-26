# Modified from https://github.com/shixun22/helmholtz

import os
import sys
import torch
import numpy as np
from scipy.fft import fft2, ifft2, fftfreq

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from ddpm.helper_functions.compute_divergence import compute_divergence



def decompose_vector_field(field_tensor, dx=1.0, dy=1.0):
    """
    Performs Helmholtz-Hodge decomposition on a 2D vector field.

    Args:
        field_tensor (torch.Tensor): shape (H, W, 2), where [:, :, 0] is u and [:, :, 1] is v
        dx, dy (float): grid spacing in x and y (defaults to 1.0)

    Returns:
        (u, v), (u_irr, v_irr), (u_sol, v_sol): original, irrotational, divergence-free components (all np.ndarrays)
    """
    H, W = field_tensor.shape[:2]
    u = field_tensor[:, :, 0].cpu().numpy()
    v = field_tensor[:, :, 1].cpu().numpy()

    # FFT of the vector field
    u_f = np.fft.fftn(u) #fft2(u)
    v_f = np.fft.fftn(v) #fft2(v)

    # Frequency components (account for dx, dy)
    kx = np.fft.fftfreq(H).reshape(H, 1) #fftfreq(H, d=dx).reshape(H, 1)
    ky = np.fft.fftfreq(W).reshape(1, W) #fftfreq(W, d=dy).reshape(1, W)
    k2 = kx**2 + ky**2
    k2[0, 0] = 1.0  # avoid division by zero at zero frequency

    # Project onto irrotational (compressive) component
    div_f = u_f * kx + v_f * ky
    compressive_overk = div_f / k2

    u_irr = np.fft.ifftn(compressive_overk * kx).real
    v_irr = np.fft.ifftn(compressive_overk * ky).real

    # Solenoidal (divergence-free) part is residual
    u_sol = u - u_irr
    v_sol = v - v_irr

    # Make everything tensors here
    u = torch.tensor(u)
    v = torch.tensor(v)
    u_irr = torch.tensor(u_irr)
    v_irr = torch.tensor(v_irr)
    u_sol = torch.tensor(u_sol)
    v_sol = torch.tensor(v_sol)
    
    """ Useful for testing:
    # Optional: diagnostics
    div_sol_f = fft2(u_sol) * kx + fft2(v_sol) * ky
    div_sol = np.real(ifft2(div_sol_f))


    # Prints
    print("Given diagnostics: \n")
    print(f"div_solenoidal min: {np.abs(div_sol).min():.2e}")
    print(f"div_solenoidal max: {np.abs(div_sol).max():.2e}")
    print(f"div_solenoidal mean: {np.abs(div_sol).mean():.2e}")

    print("\nTesting our divergence computations: \n")
    print(f"Initial div: {compute_divergence(u, v)}")
    print(f"Solenoidal div: {compute_divergence(u_sol, v_sol)}")
    print(f"Comptressive div: {compute_divergence(u_irr, v_irr)}")

    print("\nVariances:\n")
    print(f"original var x/y: {u.var():.3e}, {v.var():.3e}")
    print(f"solenoidal var x/y: {u_sol.var():.3e}, {v_sol.var():.3e}")
    print(f"compressive var x/y: {u_irr.var():.3e}, {v_irr.var():.3e}")
    """
    
    return (u, v), (u_irr, v_irr), (u_sol, v_sol)
