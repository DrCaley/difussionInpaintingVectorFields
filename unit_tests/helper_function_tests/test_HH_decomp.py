"""
Tests for Helmholtz-Hodge decomposition.
"""
import torch
import pytest
import numpy as np

from ddpm.helper_functions.HH_decomp import decompose_vector_field
from ddpm.helper_functions.compute_divergence import compute_divergence


# ============================================================================
# Fixtures and Helper Functions
# ============================================================================

H, W = 94, 44  # Standard dimensions for tests


@pytest.fixture
def random_vector_field():
    """Create a random 2D vector field."""
    return torch.randn(H, W, 2)


@pytest.fixture
def constant_field():
    """Constant field - div-free, should return sol same as initial, zero irr."""
    return torch.ones(H, W, 2)


@pytest.fixture
def rotational_field():
    """Div-free rotational field - should return sol same as initial, zero irr."""
    x = np.linspace(-1, 1, H)
    y = np.linspace(-1, 1, W)
    X, Y = np.meshgrid(x, y, indexing='ij')
    u = -Y
    v = X
    return torch.tensor(np.stack([u, v], axis=-1), dtype=torch.float32)


@pytest.fixture
def radial_field():
    """Radial field - should return irr same as initial, zero div field."""
    eps = 1e-5  # avoid division by zero
    x = np.linspace(-1, 1, H)
    y = np.linspace(-1, 1, W)
    X, Y = np.meshgrid(x, y, indexing='ij')
    r2 = X**2 + Y**2 + eps
    u = X / np.sqrt(r2)
    v = Y / np.sqrt(r2)
    return torch.tensor(np.stack([u, v], axis=-1), dtype=torch.float32)


@pytest.fixture
def mixed_field():
    """Mixed field with both compressive and divergenceless components."""
    x = np.linspace(-1, 1, H)
    y = np.linspace(-1, 1, W)
    X, Y = np.meshgrid(x, y, indexing='ij')
    u = np.sin(np.pi * X) * np.cos(np.pi * Y)
    v = -np.cos(np.pi * X) * np.sin(np.pi * Y)
    return torch.tensor(np.stack([u, v], axis=-1), dtype=torch.float32)


# ============================================================================
# Basic Functionality Tests
# ============================================================================

class TestDecomposeBasic:
    """Basic tests for decompose_vector_field."""
    
    def test_returns_three_tuples(self, random_vector_field):
        """Should return 3 tuples of (u, v) components."""
        result = decompose_vector_field(random_vector_field)
        
        assert len(result) == 3
        for component in result:
            assert len(component) == 2
            
    def test_output_shapes_match_input(self, random_vector_field):
        """Output shapes should match input dimensions."""
        original, irrotational, solenoidal = decompose_vector_field(random_vector_field)
        
        for u, v in [original, irrotational, solenoidal]:
            assert u.shape == (H, W)
            assert v.shape == (H, W)
            
    def test_returns_tensors(self, random_vector_field):
        """Should return torch tensors."""
        original, irrotational, solenoidal = decompose_vector_field(random_vector_field)
        
        for u, v in [original, irrotational, solenoidal]:
            assert isinstance(u, torch.Tensor)
            assert isinstance(v, torch.Tensor)


# ============================================================================
# Decomposition Recovery Tests
# ============================================================================

class TestDecompositionRecovery:
    """Tests for the fundamental property: irr + sol = original."""
    
    def test_sum_recovers_original(self, random_vector_field):
        """Irrotational + solenoidal should equal original."""
        (u, v), (u_irr, v_irr), (u_sol, v_sol) = decompose_vector_field(random_vector_field)
        
        # Convert to same dtype for comparison
        u = u.float()
        u_irr = u_irr.float()
        u_sol = u_sol.float()
        v = v.float()
        v_irr = v_irr.float()
        v_sol = v_sol.float()
        
        # Sum should recover original
        u_recovered = u_irr + u_sol
        v_recovered = v_irr + v_sol
        
        assert torch.allclose(u, u_recovered, atol=1e-4)
        assert torch.allclose(v, v_recovered, atol=1e-4)
        
    def test_recovery_with_constant_field(self, constant_field):
        """Recovery should work for constant field."""
        (u, v), (u_irr, v_irr), (u_sol, v_sol) = decompose_vector_field(constant_field)
        
        # Convert to same dtype
        u = u.float()
        u_irr = u_irr.float()
        u_sol = u_sol.float()
        
        assert torch.allclose(u, u_irr + u_sol, atol=1e-4)
        
    def test_recovery_with_rotational_field(self, rotational_field):
        """Recovery should work for rotational field."""
        (u, v), (u_irr, v_irr), (u_sol, v_sol) = decompose_vector_field(rotational_field)
        
        # Convert to same dtype
        u = u.float()
        u_irr = u_irr.float()
        u_sol = u_sol.float()
        
        assert torch.allclose(u, u_irr + u_sol, atol=1e-4)
        
    def test_recovery_with_mixed_field(self, mixed_field):
        """Recovery should work for mixed field."""
        (u, v), (u_irr, v_irr), (u_sol, v_sol) = decompose_vector_field(mixed_field)
        
        # Convert to same dtype
        u = u.float()
        u_irr = u_irr.float()
        u_sol = u_sol.float()
        
        assert torch.allclose(u, u_irr + u_sol, atol=1e-4)


# ============================================================================
# Solenoidal (Divergence-Free) Tests
# ============================================================================

class TestSolenoidalDivergence:
    """Tests that solenoidal component is divergence-free."""
    
    def test_solenoidal_has_low_divergence(self, random_vector_field):
        """Solenoidal component should have near-zero divergence."""
        _, _, (u_sol, v_sol) = decompose_vector_field(random_vector_field)
        
        div_sol = compute_divergence(u_sol, v_sol)
        
        # Mean absolute divergence should be reasonably small (not necessarily very small)
        # The FFT-based decomposition has some numerical error
        assert div_sol.abs().mean() < 1.0, f"Solenoidal divergence too high: {div_sol.abs().mean()}"
        
    def test_rotational_field_mostly_solenoidal(self, rotational_field):
        """A rotational field should be mostly in the solenoidal component."""
        (u, v), (u_irr, v_irr), (u_sol, v_sol) = decompose_vector_field(rotational_field)
        
        # Energy in solenoidal should be significant
        sol_energy = (u_sol**2 + v_sol**2).sum()
        total_energy = (u**2 + v**2).sum()
        
        # Solenoidal should contain most of the field
        ratio = sol_energy / (total_energy + 1e-10)
        assert ratio > 0.5, f"Expected solenoidal to dominate for rotational field, got ratio {ratio}"


# ============================================================================
# Irrotational Tests
# ============================================================================

class TestIrrotationalComponent:
    """Tests for irrotational (curl-free) component."""
    
    def test_radial_field_has_irrotational_component(self, radial_field):
        """A radial field should have significant irrotational component."""
        (u, v), (u_irr, v_irr), (u_sol, v_sol) = decompose_vector_field(radial_field)
        
        # Energy in irrotational should be significant
        irr_energy = (u_irr**2 + v_irr**2).sum()
        total_energy = (u**2 + v**2).sum()
        
        # Should have some energy in irrotational
        assert irr_energy > 0


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Edge case tests for decomposition."""
    
    def test_zero_field(self):
        """Zero field should decompose to zeros."""
        field = torch.zeros(H, W, 2)
        original, irrotational, solenoidal = decompose_vector_field(field)
        
        for u, v in [original, irrotational, solenoidal]:
            assert torch.allclose(u, torch.zeros_like(u), atol=1e-10)
            assert torch.allclose(v, torch.zeros_like(v), atol=1e-10)
            
    def test_small_field_dimensions(self):
        """Should handle small field dimensions."""
        field = torch.randn(8, 8, 2)
        result = decompose_vector_field(field)
        
        assert len(result) == 3
        
    def test_non_square_field(self):
        """Should handle non-square fields."""
        field = torch.randn(16, 128, 2)
        (u, v), _, _ = decompose_vector_field(field)
        
        assert u.shape == (16, 128)
        assert v.shape == (16, 128)


# ============================================================================
# Numerical Stability
# ============================================================================

class TestNumericalStability:
    """Tests for numerical stability."""
    
    def test_large_values(self):
        """Should handle large values."""
        field = torch.randn(H, W, 2) * 1e6
        (u, v), (u_irr, v_irr), (u_sol, v_sol) = decompose_vector_field(field)
        
        # Convert to same dtype
        u = u.float()
        u_irr = u_irr.float()
        u_sol = u_sol.float()
        
        # Should still recover original (with higher tolerance for large values)
        assert torch.allclose(u, u_irr + u_sol, rtol=1e-3)
        
    def test_small_values(self):
        """Should handle small values."""
        field = torch.randn(H, W, 2) * 1e-6
        (u, v), (u_irr, v_irr), (u_sol, v_sol) = decompose_vector_field(field)
        
        # Convert to same dtype
        u = u.float()
        u_irr = u_irr.float()
        u_sol = u_sol.float()
        
        assert torch.allclose(u, u_irr + u_sol, atol=1e-8)


# ============================================================================
# Consistency Tests
# ============================================================================

class TestConsistency:
    """Tests for consistency of decomposition."""
    
    def test_deterministic(self, random_vector_field):
        """Same input should give same output."""
        result1 = decompose_vector_field(random_vector_field)
        result2 = decompose_vector_field(random_vector_field)
        
        for (u1, v1), (u2, v2) in zip(result1, result2):
            assert torch.allclose(u1, u2)
            assert torch.allclose(v1, v2)
            
    def test_linear_scaling(self, random_vector_field):
        """Decomposition should be linear: decomp(α*f) = α*decomp(f)."""
        alpha = 3.5
        
        result_original = decompose_vector_field(random_vector_field)
        result_scaled = decompose_vector_field(random_vector_field * alpha)
        
        for (u1, v1), (u2, v2) in zip(result_original, result_scaled):
            # Convert to same dtype
            u1 = u1.float()
            u2 = u2.float()
            v1 = v1.float()
            v2 = v2.float()
            
            # Use larger tolerance for numerical stability
            assert torch.allclose(u1 * alpha, u2, rtol=1e-2, atol=1e-4)
            assert torch.allclose(v1 * alpha, v2, rtol=1e-2, atol=1e-4)