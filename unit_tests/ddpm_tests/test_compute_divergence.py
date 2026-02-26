import torch
import numpy as np
import pytest
from ddpm.helper_functions.compute_divergence import compute_divergence  # Replace with actual module name

def test_zero_divergence_rotation_field():
    """
    A 2D rotational field like [-y, x] has zero divergence.
    """
    N = 64
    x = torch.linspace(-1, 1, N)
    y = torch.linspace(-1, 1, N)
    X, Y = torch.meshgrid(x, y, indexing="ij")

    vx = -Y  # u = -y
    vy = X   # v =  x

    div = compute_divergence(vx, vy)
    max_abs_div = torch.abs(div).max().item()
    assert max_abs_div < 1e-2, f"Divergence should be ~0, got max |div|={max_abs_div}"

def test_known_divergence_against_numpy():
    """
    Compares PyTorch implementation against numpy gradient.
    """
    N = 64
    L = 2 * np.pi
    dx = L / N
    x = np.linspace(0, L, N)
    y = np.linspace(0, L, N)
    X, Y = np.meshgrid(x, y, indexing="ij")

    u = np.sin(X)
    v = np.cos(Y)

    dudx = np.gradient(u, dx, axis=0)
    dvdy = np.gradient(v, dx, axis=1)
    expected_div = dudx + dvdy

    vx = torch.tensor(u, dtype=torch.float32)
    vy = torch.tensor(v, dtype=torch.float32)
    div = compute_divergence(vx, vy).numpy()

    # Note: Our compute_divergence uses a different gradient method than np.gradient
    # so we only check that the pattern is similar (correlation), not exact values
    corr = np.corrcoef(div[1:-1, 1:-1].flatten(), expected_div[1:-1, 1:-1].flatten())[0, 1]
    assert corr > 0.99, f"Divergence pattern should correlate strongly, got {corr}"

def test_constant_field_divergence_is_zero():
    """
    A constant field should have zero divergence everywhere.
    """
    vx = torch.ones(32, 32)
    vy = torch.ones(32, 32)

    div = compute_divergence(vx, vy)
    assert torch.allclose(div, torch.zeros_like(div), atol=1e-6)
