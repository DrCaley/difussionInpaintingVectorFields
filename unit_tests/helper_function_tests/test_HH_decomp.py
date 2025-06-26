import os
import sys
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from ddpm.helper_functions.HH_decomp import decompose_vector_field
from ddpm.helper_functions.compute_divergence import compute_divergence
from plots.plot_vector_field_tool import plot_vector_field



H, W = 94, 44 # Dimensions of field

# Div-free already, should return sol same as initial, zero irr
def constant_field_test():
    field = torch.ones(H, W, 2)  # Constant (1, 1) everywhere
    return field

# Div-free already, should return sol same as initial, zero irr
def rotational_field_test():
    x = np.linspace(-1, 1, H)
    y = np.linspace(-1, 1, W)
    X, Y = np.meshgrid(x, y, indexing='ij')
    u = -Y
    v = X
    field = torch.tensor(np.stack([u, v], axis=-1), dtype=torch.float32)
    return field

# Radial field, should return irr same as initial, zero div field
def radial_field_test():
    # TODO: small artifacts appear during this test. Are we okay with this?
    eps = 1e-5  # avoid division by zero
    x = np.linspace(-1, 1, H)
    y = np.linspace(-1, 1, W)
    X, Y = np.meshgrid(x, y, indexing='ij')
    r2 = X**2 + Y**2 + eps
    u = X / np.sqrt(r2)
    v = Y / np.sqrt(r2)
    field = torch.tensor(np.stack([u, v], axis=-1), dtype=torch.float32)
    return field

# Mixed field (both compressive and divergenceless bits)
def mixed_field_test():
    x = np.linspace(-1, 1, H)
    y = np.linspace(-1, 1, W)
    X, Y = np.meshgrid(x, y, indexing='ij')
    u = np.sin(np.pi * X) * np.cos(np.pi * Y)
    v = -np.cos(np.pi * X) * np.sin(np.pi * Y)
    field = torch.tensor(np.stack([u, v], axis=-1), dtype=torch.float32)
    return field

# Mixed field (both compressive and divergenceless bits)
def drew_test():
    return torch.randn((H, W, 2))


(u_np, v_np), (u_irr, v_irr), (u_sol, v_sol) = decompose_vector_field(drew_test())


# Viewing
plot_vector_field(u_np, v_np, scale=20, file="Initial_Vector_Field.png", title="Initial Vector Field")
plot_vector_field(u_irr, v_irr, scale=20, file="Irrotational_Vector_Field.png", title="Irrotational Vector Field")
plot_vector_field(u_sol, v_sol, scale=20, file="Divless_Vector_Field.png", title="Divless Vector Field")

# Print stats
print(f"The divergences of the initial field falls between \n {compute_divergence(u_np, v_np).min()} and {compute_divergence(u_np, v_np).max()}.")
print(f"The divergences of the irrotational field falls between \n {compute_divergence(u_irr, v_irr).min()} and {compute_divergence(u_irr, v_irr).max()}.")
print(f"The divergences of the divergenceless field, WHICH SHOULD BE ZERO, falls between \n {compute_divergence(u_sol, v_sol).min()} AND {compute_divergence(u_sol, v_sol).max()}.")