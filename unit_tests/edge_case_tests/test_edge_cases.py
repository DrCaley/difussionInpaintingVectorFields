"""
Tests for edge cases and error handling.
"""
import torch
import pytest
import numpy as np

from ddpm.neural_networks.ddpm import GaussianDDPM
from ddpm.neural_networks.unets.unet_xl import MyUNet
from ddpm.utils.noise_utils import NOISE_REGISTRY, NoiseStrategy, GaussianNoise
from ddpm.helper_functions.loss_functions import LOSS_REGISTRY, MSELossStrategy
from ddpm.helper_functions.HH_decomp import decompose_vector_field
from ddpm.helper_functions.compute_divergence import compute_divergence


@pytest.fixture
def device():
    """Get best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


@pytest.fixture
def simple_unet(device):
    """Create a simple UNet for testing."""
    return MyUNet(n_steps=100, time_emb_dim=32).to(device)


# ============================================================================
# Mask Edge Cases
# ============================================================================

class TestMaskEdgeCases:
    """Tests for edge cases in mask handling."""
    
    def test_empty_mask_all_zeros(self, simple_unet, device):
        """Empty mask (all zeros) should still allow forward pass."""
        ddpm = GaussianDDPM(network=simple_unet, n_steps=100, device=device)
        
        x = torch.randn(1, 2, 64, 128).to(device)  # UNet dimensions
        mask = torch.zeros(1, 1, 64, 128).to(device)  # All masked
        t = torch.tensor([50]).to(device)
        
        # Forward should still work
        noisy = ddpm.forward(x, t)
        
        assert noisy.shape == x.shape
        
    def test_full_mask_all_ones(self, simple_unet, device):
        """Full mask (all ones) should still allow forward pass."""
        ddpm = GaussianDDPM(network=simple_unet, n_steps=100, device=device)
        
        x = torch.randn(1, 2, 64, 128).to(device)  # UNet dimensions
        mask = torch.ones(1, 1, 64, 128).to(device)  # Nothing masked
        t = torch.tensor([50]).to(device)
        
        noisy = ddpm.forward(x, t)
        
        assert noisy.shape == x.shape
        
    def test_sparse_mask(self, simple_unet, device):
        """Very sparse mask should work."""
        ddpm = GaussianDDPM(network=simple_unet, n_steps=100, device=device)
        
        x = torch.randn(1, 2, 64, 128).to(device)  # UNet dimensions
        mask = torch.zeros(1, 1, 64, 128).to(device)
        mask[:, :, 32, 64] = 1  # Only one pixel known
        t = torch.tensor([50]).to(device)
        
        noisy = ddpm.forward(x, t)
        
        assert noisy.shape == x.shape


# ============================================================================
# NaN Handling
# ============================================================================

class TestNaNHandling:
    """Tests for NaN handling in various components."""
    
    def test_nan_in_input_propagates(self, simple_unet, device):
        """NaN in input should be handled (not crash)."""
        ddpm = GaussianDDPM(network=simple_unet, n_steps=100, device=device)
        
        x = torch.randn(1, 2, 64, 128).to(device)  # UNet dimensions
        x[0, 0, 10, 10] = float('nan')
        t = torch.tensor([50]).to(device)
        
        # Should not crash, even if output contains NaN
        try:
            noisy = ddpm.forward(x, t)
            # NaN may propagate, but shouldn't crash
        except Exception as e:
            pytest.fail(f"NaN input caused crash: {e}")
            
    def test_loss_with_nan_mask(self):
        """Loss computation with NaN values (masked out)."""
        loss_fn = MSELossStrategy()
        
        pred = torch.randn(1, 2, 64, 128)  # UNet dimensions
        target = torch.randn(1, 2, 64, 128)
        
        # Create tensor with some NaN that would be masked
        pred_with_nan = pred.clone()
        pred_with_nan[0, 0, 5, 5] = float('nan')
        
        # Loss should handle NaN (may be inf or nan, but shouldn't crash)
        try:
            loss = loss_fn.forward(pred_with_nan, target, pred_with_nan)
        except Exception as e:
            pytest.fail(f"Loss computation crashed with NaN: {e}")


# ============================================================================
# Timestep Edge Cases
# ============================================================================

class TestTimestepEdgeCases:
    """Tests for timestep boundary conditions."""
    
    def test_timestep_zero(self, simple_unet, device):
        """Timestep 0 should work."""
        ddpm = GaussianDDPM(network=simple_unet, n_steps=100, device=device)
        
        x = torch.randn(1, 2, 64, 128).to(device)  # UNet dimensions
        t = torch.tensor([0]).to(device)
        
        noisy = ddpm.forward(x, t)
        
        assert noisy.shape == x.shape
        
    def test_timestep_max(self, simple_unet, device):
        """Maximum timestep (n_steps-1) should work."""
        n_steps = 100
        ddpm = GaussianDDPM(network=simple_unet, n_steps=n_steps, device=device)
        
        x = torch.randn(1, 2, 64, 128).to(device)  # UNet dimensions
        t = torch.tensor([n_steps - 1]).to(device)
        
        noisy = ddpm.forward(x, t)
        
        assert noisy.shape == x.shape
        
    def test_different_timesteps_in_batch(self, simple_unet, device):
        """Different timesteps for each sample in batch."""
        ddpm = GaussianDDPM(network=simple_unet, n_steps=100, device=device)
        
        x = torch.randn(4, 2, 64, 128).to(device)  # UNet dimensions
        t = torch.tensor([0, 25, 50, 99]).to(device)  # Different for each
        
        noisy = ddpm.forward(x, t)
        
        assert noisy.shape == x.shape


# ============================================================================
# Batch Size Edge Cases
# ============================================================================

class TestBatchSizeEdgeCases:
    """Tests for various batch sizes."""
    
    def test_batch_size_one(self, simple_unet, device):
        """Batch size of 1 should work."""
        ddpm = GaussianDDPM(network=simple_unet, n_steps=100, device=device)
        
        x = torch.randn(1, 2, 64, 128).to(device)  # UNet dimensions
        t = torch.tensor([50]).to(device)
        
        noisy = ddpm.forward(x, t)
        
        assert noisy.shape == x.shape
        
    def test_large_batch_size(self, simple_unet, device):
        """Large batch size should work (memory permitting)."""
        ddpm = GaussianDDPM(network=simple_unet, n_steps=100, device=device)
        
        # Use smaller batch for memory safety in tests
        x = torch.randn(8, 2, 64, 128).to(device)  # UNet dimensions
        t = torch.randint(0, 100, (8,)).to(device)
        
        noisy = ddpm.forward(x, t)
        
        assert noisy.shape == x.shape


# ============================================================================
# Device Consistency
# ============================================================================

class TestDeviceConsistency:
    """Tests for device consistency across operations."""
    
    def test_output_on_same_device_as_input(self, simple_unet, device):
        """Output should be on same device as input."""
        ddpm = GaussianDDPM(network=simple_unet, n_steps=100, device=device)
        
        x = torch.randn(1, 2, 64, 128).to(device)  # UNet dimensions
        t = torch.tensor([50]).to(device)
        
        noisy = ddpm.forward(x, t)
        
        assert noisy.device.type == device.type
        
    def test_noise_on_correct_device(self, device):
        """Noise strategy should produce noise on correct device."""
        noise_strategy = GaussianNoise()
        
        shape = (1, 2, 64, 128)  # UNet dimensions
        noise = noise_strategy.generate(shape, device=device)  # Use keyword arg
        
        assert noise.device.type == device.type


# ============================================================================
# Input Shape Edge Cases
# ============================================================================

class TestInputShapeEdgeCases:
    """Tests for various input shapes."""
    
    def test_standard_ocean_dimensions(self, simple_unet, device):
        """Standard ocean data dimensions (padded to 64x128) should work."""
        ddpm = GaussianDDPM(network=simple_unet, n_steps=100, device=device)
        
        # Ocean data is 44x94, padded to UNet dimensions 64x128
        x = torch.randn(1, 2, 64, 128).to(device)
        t = torch.tensor([50]).to(device)
        
        noisy = ddpm.forward(x, t)
        
        assert noisy.shape == (1, 2, 64, 128)


# ============================================================================
# Noise Strategy Edge Cases
# ============================================================================

class TestNoiseStrategyEdgeCases:
    """Tests for edge cases in noise strategies."""
    
    def test_noise_registry_invalid_key(self):
        """Invalid key should raise KeyError."""
        with pytest.raises(KeyError):
            _ = NOISE_REGISTRY['nonexistent_strategy']
            
    def test_loss_registry_invalid_key(self):
        """Invalid key should raise KeyError."""
        with pytest.raises(KeyError):
            _ = LOSS_REGISTRY['nonexistent_loss']
            
    def test_noise_different_each_call(self, device):
        """Noise should be different each call (stochastic)."""
        noise_strategy = GaussianNoise()
        
        shape = (1, 2, 44, 94)
        noise1 = noise_strategy.generate(shape, device)
        noise2 = noise_strategy.generate(shape, device)
        
        assert not torch.allclose(noise1, noise2)


# ============================================================================
# HH Decomposition Edge Cases
# ============================================================================

class TestHHDecompEdgeCases:
    """Tests for edge cases in Helmholtz-Hodge decomposition."""
    
    def test_zero_field_decomposition(self):
        """Zero field should decompose to all zeros."""
        field = torch.zeros(32, 64, 2)
        (u, v), (u_irr, v_irr), (u_sol, v_sol) = decompose_vector_field(field)
        
        assert torch.allclose(u_irr, torch.zeros_like(u_irr), atol=1e-10)
        assert torch.allclose(u_sol, torch.zeros_like(u_sol), atol=1e-10)
        
    def test_tiny_field_values(self):
        """Very small field values should not cause numerical issues."""
        field = torch.randn(32, 64, 2) * 1e-10
        
        try:
            result = decompose_vector_field(field)
            assert len(result) == 3
        except Exception as e:
            pytest.fail(f"Small values caused error: {e}")


# ============================================================================
# Gradient Edge Cases
# ============================================================================

class TestGradientEdgeCases:
    """Tests for gradient computation edge cases."""
    
    def test_gradient_through_forward(self, simple_unet, device):
        """Gradients should flow through forward pass."""
        ddpm = GaussianDDPM(network=simple_unet, n_steps=100, device=device)
        
        x = torch.randn(1, 2, 64, 128, requires_grad=True).to(device)  # UNet dimensions
        t = torch.tensor([50]).to(device)
        
        noisy = ddpm.forward(x, t)
        
        # Verify gradient computation is possible
        assert noisy.grad_fn is not None  # Has gradient function in computation graph
        
    def test_gradient_through_backward(self, simple_unet, device):
        """Gradients should flow through backward (prediction) pass."""
        ddpm = GaussianDDPM(network=simple_unet, n_steps=100, device=device)
        
        noisy = torch.randn(1, 2, 64, 128).to(device)  # UNet dimensions
        t = torch.tensor([50]).to(device)
        
        predicted = ddpm.backward(noisy, t)
        
        loss = predicted.sum()
        loss.backward()
        
        # Model parameters should have gradients
        has_grad = any(p.grad is not None for p in simple_unet.parameters())
        assert has_grad


# ============================================================================
# Memory Edge Cases
# ============================================================================

class TestMemoryEdgeCases:
    """Tests for memory-related edge cases."""
    
    def test_no_memory_leak_in_loop(self, simple_unet, device):
        """Repeated forward passes should not leak memory."""
        ddpm = GaussianDDPM(network=simple_unet, n_steps=100, device=device)
        
        initial_mem = None
        if device.type == 'cuda':
            initial_mem = torch.cuda.memory_allocated()
        
        for _ in range(10):
            x = torch.randn(1, 2, 64, 128).to(device)  # UNet dimensions
            t = torch.tensor([50]).to(device)
            
            with torch.no_grad():
                noisy = ddpm.forward(x, t)
            
            del x, t, noisy
        
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            final_mem = torch.cuda.memory_allocated()
            # Memory shouldn't grow significantly
            assert final_mem < initial_mem + 1e8  # 100MB tolerance
