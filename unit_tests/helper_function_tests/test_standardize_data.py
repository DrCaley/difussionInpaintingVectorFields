import os
import sys
import torch
from math import isclose
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from ddpm.helper_functions import standardize_data



# === Tests ===

def test_shape_preservation():
    tensor = torch.rand(2, 4, 4)
    std = standardize_data()
    out = std(tensor)
    assert out.shape == tensor.shape, "Shape should remain unchanged"

def test_max_magnitude_is_one():
    # Construct a field where max magnitude is known
    u = torch.tensor([[0.0, 3.0], [0.0, 0.0]])
    v = torch.tensor([[0.0, 4.0], [0.0, 0.0]])
    tensor = torch.stack([u, v])  # Shape: [2, 2, 2]
    std = standardize_data()
    out = std(tensor)

    # Magnitude of (3, 4) = 5, so normalized vector = (0.6, 0.8), magnitude = 1
    magnitude = torch.sqrt(out[0]**2 + out[1]**2)
    max_mag = magnitude.max().item()
    assert isclose(max_mag, 1.0, rel_tol=1e-5), f"Expected max magnitude 1.0, got {max_mag}"

def test_direction_preservation():
    u = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    v = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    tensor = torch.stack([u, v])  # Only x-components

    std = standardize_data()
    out = std(tensor)

    # The x-components should now be divided by 4 (max of u)
    expected = u / 4.0
    assert torch.allclose(out[0], expected, atol=1e-6), "X components not correctly scaled"
    assert torch.allclose(out[1], v / 4.0, atol=1e-6), "Y components should remain zero"

def test_zero_vector_field():
    tensor = torch.zeros(2, 3, 3)
    std = standardize_data()
    try:
        out = std(tensor)
        assert torch.all(out == 0), "Zero field should remain zero"
    except Exception as e:
        assert False, f"Zero field should not crash the code: {e}"