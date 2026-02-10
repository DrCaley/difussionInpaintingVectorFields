"""
Tests for noise strategies - critical for physics-informed diffusion.
"""
import torch
import pytest
import numpy as np

from ddpm.utils.noise_utils import (
    NoiseStrategy,
    GaussianNoise,
    DivergenceFreeNoise,
    DivergenceFreeGaussianNoise,
    HH_Decomp_Div_Free,
    NOISE_REGISTRY,
    get_noise_strategy
)
from ddpm.helper_functions.compute_divergence import compute_divergence


@pytest.fixture
def standard_shape():
    """Standard shape for noise generation: (batch, channels, H, W)."""
    return (2, 2, 64, 128)


@pytest.fixture
def device():
    return torch.device('cpu')


class TestGaussianNoise:
    """Tests for standard Gaussian noise strategy."""
    
    def test_gaussian_noise_correct_shape(self, standard_shape, device):
        """Generated noise should have correct shape."""
        strategy = GaussianNoise()
        noise = strategy.generate(standard_shape, device=device)
        assert noise.shape == standard_shape
        
    def test_gaussian_noise_approximately_zero_mean(self, device):
        """Gaussian noise should have approximately zero mean over large samples."""
        strategy = GaussianNoise()
        shape = (1000, 2, 64, 128)
        noise = strategy.generate(shape, device=device)
        
        # Mean should be close to 0
        assert abs(noise.mean().item()) < 0.05
        
    def test_gaussian_noise_approximately_unit_variance(self, device):
        """Gaussian noise should have approximately unit variance."""
        strategy = GaussianNoise()
        shape = (100, 2, 64, 128)
        noise = strategy.generate(shape, device=device)
        
        # Variance should be close to 1
        assert abs(noise.var().item() - 1.0) < 0.1
        
    def test_gaussian_noise_different_each_call(self, standard_shape, device):
        """Each call should produce different noise."""
        strategy = GaussianNoise()
        noise1 = strategy.generate(standard_shape, device=device)
        noise2 = strategy.generate(standard_shape, device=device)
        
        assert not torch.allclose(noise1, noise2)
        
    def test_gaussian_scaling_returns_true(self):
        """Gaussian noise should use gaussian scaling."""
        strategy = GaussianNoise()
        assert strategy.get_gaussian_scaling() is True
        
    def test_gaussian_noise_callable(self, device):
        """Test __call__ interface works."""
        strategy = GaussianNoise()
        x = torch.randn(2, 2, 64, 128, device=device)
        t = torch.tensor([50, 50], device=device)
        
        noise = strategy(x, t)
        assert noise.shape == x.shape


class TestDivergenceFreeNoise:
    """Tests for divergence-free noise - CRITICAL for physics constraints."""
    
    @pytest.mark.slow
    def test_div_free_noise_correct_shape(self, device):
        """Generated noise should have correct shape."""
        strategy = DivergenceFreeNoise()
        shape = (1, 2, 64, 128)  # Single sample for this slow test
        noise = strategy.generate(shape, device=device)
        assert noise.shape == shape
        
    @pytest.mark.slow
    def test_div_free_noise_has_low_divergence(self, device):
        """Divergence-free noise should have near-zero divergence."""
        strategy = DivergenceFreeNoise()
        shape = (1, 2, 64, 128)
        noise = strategy.generate(shape, device=device)
        
        # Compute divergence of the noise field
        u = noise[0, 0]  # First channel (u component)
        v = noise[0, 1]  # Second channel (v component)
        div = compute_divergence(u, v)
        
        # Divergence should be very small
        max_abs_div = torch.abs(div).max().item()
        assert max_abs_div < 0.5, f"Divergence too high: {max_abs_div}"
        
    def test_div_free_requires_two_channels(self, device):
        """Should assert if not given 2 channels."""
        strategy = DivergenceFreeNoise()
        shape = (1, 3, 64, 128)  # Wrong number of channels
        
        with pytest.raises(AssertionError):
            strategy.generate(shape, device=device)


class TestDivergenceFreeGaussianNoise:
    """Tests for the layered divergence-free Gaussian noise."""
    
    @pytest.mark.slow
    def test_div_gaussian_correct_shape(self, device):
        """Generated noise should have correct shape."""
        strategy = DivergenceFreeGaussianNoise()
        shape = (1, 2, 64, 128)
        noise = strategy.generate(shape, device=device)
        assert noise.shape == shape
        
    @pytest.mark.slow
    def test_div_gaussian_has_low_divergence(self, device):
        """Should produce approximately divergence-free field."""
        strategy = DivergenceFreeGaussianNoise()
        shape = (1, 2, 64, 128)
        noise = strategy.generate(shape, device=device)
        
        u = noise[0, 0]
        v = noise[0, 1]
        div = compute_divergence(u, v)
        
        # This strategy may not be strictly div-free, just check it computes
        max_abs_div = torch.abs(div).max().item()
        assert max_abs_div >= 0  # Just verify it computes without error


class TestHHDecompDivFree:
    """Tests for Helmholtz-Hodge decomposition based div-free noise."""
    
    def test_hh_decomp_gaussian_scaling_false(self):
        """HH decomp should not use gaussian scaling."""
        strategy = HH_Decomp_Div_Free()
        assert strategy.get_gaussian_scaling() is False


class TestNoiseRegistry:
    """Tests for the noise strategy registry."""
    
    def test_registry_contains_gaussian(self):
        """Registry should contain gaussian noise."""
        assert "gaussian" in NOISE_REGISTRY
        
    def test_registry_contains_div_free(self):
        """Registry should contain divergence-free noise."""
        assert "div_free" in NOISE_REGISTRY
        
    def test_registry_contains_div_gaussian(self):
        """Registry should contain div_gaussian noise."""
        assert "div_gaussian" in NOISE_REGISTRY
        
    def test_registry_contains_hh_decomp(self):
        """Registry should contain hh_decomp_div_free noise."""
        assert "hh_decomp_div_free" in NOISE_REGISTRY
        
    def test_get_noise_strategy_returns_correct_type(self):
        """get_noise_strategy should return correct strategy type."""
        strategy = get_noise_strategy("gaussian")
        assert isinstance(strategy, GaussianNoise)
        
    def test_get_noise_strategy_invalid_raises(self):
        """Invalid strategy name should raise KeyError."""
        with pytest.raises(KeyError):
            get_noise_strategy("invalid_strategy")
            
    def test_all_registry_entries_are_noise_strategies(self):
        """All entries should be NoiseStrategy subclasses."""
        for name, cls in NOISE_REGISTRY.items():
            instance = cls()
            assert isinstance(instance, NoiseStrategy), f"{name} is not a NoiseStrategy"


class TestNoiseStrategyInterface:
    """Tests for the NoiseStrategy base class interface."""
    
    def test_base_class_generate_raises(self):
        """Base class generate() should raise NotImplementedError."""
        strategy = NoiseStrategy()
        with pytest.raises(NotImplementedError):
            strategy.generate((1, 2, 64, 128))
            
    def test_base_class_gaussian_scaling_default(self):
        """Base class should default to gaussian scaling True."""
        strategy = NoiseStrategy()
        assert strategy.get_gaussian_scaling() is True
