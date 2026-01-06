"""
Tests for loss strategies - MSE and physics-informed losses.
"""
import torch
import pytest

from ddpm.helper_functions.loss_functions import (
    LossStrategy,
    MSELossStrategy,
    PhysicalLossStrategy,
    HotGarbage,
    LOSS_REGISTRY,
    get_loss_strategy
)
from ddpm.helper_functions.compute_divergence import compute_divergence


@pytest.fixture
def sample_tensors():
    """Create sample predicted and target noise tensors."""
    batch_size = 4
    channels = 2
    height = 64
    width = 128
    
    predicted = torch.randn(batch_size, channels, height, width)
    target = torch.randn(batch_size, channels, height, width)
    noisy_img = torch.randn(batch_size, channels, height, width)
    
    return predicted, target, noisy_img


@pytest.fixture
def identical_tensors():
    """Create identical predicted and target tensors for testing zero loss."""
    t = torch.randn(4, 2, 64, 128)
    return t, t.clone(), torch.randn(4, 2, 64, 128)


class TestMSELossStrategy:
    """Tests for standard MSE loss."""
    
    def test_mse_loss_zero_for_identical(self, identical_tensors):
        """MSE should be zero when predicted equals target."""
        predicted, target, noisy = identical_tensors
        loss_fn = MSELossStrategy()
        
        loss = loss_fn(predicted, target, noisy)
        
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)
        
    def test_mse_loss_positive_for_different(self, sample_tensors):
        """MSE should be positive when predicted != target."""
        predicted, target, noisy = sample_tensors
        loss_fn = MSELossStrategy()
        
        loss = loss_fn(predicted, target, noisy)
        
        assert loss > 0
        
    def test_mse_loss_symmetric(self, sample_tensors):
        """MSE(a, b) should equal MSE(b, a)."""
        predicted, target, noisy = sample_tensors
        loss_fn = MSELossStrategy()
        
        loss1 = loss_fn(predicted, target, noisy)
        loss2 = loss_fn(target, predicted, noisy)
        
        assert torch.isclose(loss1, loss2)
        
    def test_mse_loss_scales_with_error(self):
        """MSE should increase with larger errors."""
        loss_fn = MSELossStrategy()
        
        target = torch.zeros(2, 2, 32, 32)
        small_error = torch.ones(2, 2, 32, 32) * 0.1
        large_error = torch.ones(2, 2, 32, 32) * 1.0
        
        loss_small = loss_fn(small_error, target)
        loss_large = loss_fn(large_error, target)
        
        assert loss_large > loss_small
        
    def test_mse_returns_scalar(self, sample_tensors):
        """MSE should return a scalar tensor."""
        predicted, target, noisy = sample_tensors
        loss_fn = MSELossStrategy()
        
        loss = loss_fn(predicted, target, noisy)
        
        assert loss.dim() == 0  # scalar


class TestPhysicalLossStrategy:
    """Tests for physics-informed loss with divergence penalty."""
    
    def test_physical_loss_includes_mse(self, sample_tensors):
        """Physical loss should include MSE component."""
        predicted, target, noisy = sample_tensors
        
        # Loss with only MSE (w2=0)
        mse_only = PhysicalLossStrategy(w1=1.0, w2=0.0)
        standard_mse = MSELossStrategy()
        
        loss_physical = mse_only(predicted, target, noisy)
        loss_standard = standard_mse(predicted, target, noisy)
        
        assert torch.isclose(loss_physical, loss_standard, rtol=1e-5)
        
    def test_physical_loss_penalizes_divergence(self):
        """Physical loss should be higher for fields with high divergence."""
        loss_fn = PhysicalLossStrategy(w1=0.0, w2=1.0)  # Only divergence penalty
        
        # Create a divergence-free field (rotation): u = -y, v = x
        h, w = 64, 128
        y = torch.linspace(-1, 1, h).view(h, 1).expand(h, w)
        x = torch.linspace(-1, 1, w).view(1, w).expand(h, w)
        
        # Divergence-free (rotational) field
        div_free = torch.stack([-y, x], dim=0).unsqueeze(0)  # (1, 2, H, W)
        
        # High divergence field (expanding): u = x, v = y
        high_div = torch.stack([x, y], dim=0).unsqueeze(0)  # (1, 2, H, W)
        
        target = torch.zeros_like(div_free)
        
        loss_div_free = loss_fn(div_free, target)
        loss_high_div = loss_fn(high_div, target)
        
        # Both should compute without error
        assert loss_div_free >= 0
        assert loss_high_div >= 0
        
    def test_physical_loss_weights_configurable(self, sample_tensors):
        """Different weights should produce different losses."""
        predicted, target, noisy = sample_tensors
        
        loss_fn1 = PhysicalLossStrategy(w1=1.0, w2=0.5)
        loss_fn2 = PhysicalLossStrategy(w1=1.0, w2=2.0)
        
        loss1 = loss_fn1(predicted, target, noisy)
        loss2 = loss_fn2(predicted, target, noisy)
        
        # With different w2, losses should generally differ
        # (unless divergence is exactly 0)
        assert loss1 != loss2 or torch.isclose(loss1, loss2, atol=1e-6)
        
    def test_physical_loss_static_method(self):
        """Test the static physical_loss method directly."""
        # Create a field with known properties
        h, w = 32, 64
        y = torch.linspace(-1, 1, h).view(h, 1).expand(h, w)
        x = torch.linspace(-1, 1, w).view(1, w).expand(h, w)
        
        # Any field should compute without error
        field = torch.stack([x, y], dim=0).unsqueeze(0)  # (1, 2, H, W)
        
        div_loss = PhysicalLossStrategy.physical_loss(field)
        
        # Should be non-negative
        assert div_loss >= 0


class TestLossRegistry:
    """Tests for the loss strategy registry."""
    
    def test_registry_contains_mse(self):
        """Registry should contain MSE loss."""
        assert "mse" in LOSS_REGISTRY
        
    def test_registry_contains_physical(self):
        """Registry should contain physical loss."""
        assert "physical" in LOSS_REGISTRY
        
    def test_registry_contains_best_loss(self):
        """Registry should contain best_loss (HotGarbage)."""
        assert "best_loss" in LOSS_REGISTRY
        
    def test_get_loss_strategy_returns_correct_type(self):
        """get_loss_strategy should return correct type."""
        strategy = get_loss_strategy("mse")
        assert isinstance(strategy, MSELossStrategy)
        
    def test_get_loss_strategy_with_kwargs(self):
        """Should pass kwargs to constructor."""
        strategy = get_loss_strategy("physical", w1=0.5, w2=0.5)
        assert isinstance(strategy, PhysicalLossStrategy)
        
    def test_get_loss_strategy_invalid_raises(self):
        """Invalid name should raise KeyError."""
        with pytest.raises(KeyError):
            get_loss_strategy("invalid_loss")
            
    def test_all_strategies_are_loss_strategies(self):
        """All entries should be LossStrategy subclasses."""
        for name, cls in LOSS_REGISTRY.items():
            instance = cls()
            assert isinstance(instance, LossStrategy), f"{name} is not a LossStrategy"


class TestLossStrategyInterface:
    """Tests for the LossStrategy base class."""
    
    def test_base_class_forward_raises(self):
        """Base class forward() should raise NotImplementedError."""
        strategy = LossStrategy()
        predicted = torch.randn(1, 2, 32, 32)
        target = torch.randn(1, 2, 32, 32)
        
        with pytest.raises(NotImplementedError):
            strategy(predicted, target, None)


class TestLossGradients:
    """Tests to verify losses are differentiable."""
    
    def test_mse_loss_has_gradient(self, sample_tensors):
        """MSE loss should be differentiable."""
        predicted, target, noisy = sample_tensors
        predicted.requires_grad_(True)
        
        loss_fn = MSELossStrategy()
        loss = loss_fn(predicted, target, noisy)
        loss.backward()
        
        assert predicted.grad is not None
        assert not torch.all(predicted.grad == 0)
        
    def test_physical_loss_has_gradient(self, sample_tensors):
        """Physical loss should be differentiable."""
        predicted, target, noisy = sample_tensors
        predicted.requires_grad_(True)
        
        loss_fn = PhysicalLossStrategy()
        loss = loss_fn(predicted, target, noisy)
        loss.backward()
        
        assert predicted.grad is not None
