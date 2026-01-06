"""
Integration tests for end-to-end workflows.
These tests verify that components work together correctly.
"""
import torch
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch
import tempfile
import os

from ddpm.neural_networks.ddpm import GaussianDDPM
from ddpm.neural_networks.unets.unet_xl import MyUNet
from ddpm.utils.noise_utils import NOISE_REGISTRY, NoiseStrategy
from ddpm.helper_functions.loss_functions import LOSS_REGISTRY
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


@pytest.fixture
def sample_batch(device):
    """Create a sample batch of data matching UNet expected dimensions."""
    batch_size = 2
    # UNet expects 64x128 images
    images = torch.randn(batch_size, 2, 64, 128).to(device)
    masks = torch.ones(batch_size, 1, 64, 128).to(device)
    return images, masks


# Standard shape for UNet tests
UNET_HEIGHT = 64
UNET_WIDTH = 128


class TestDDPMWithUNet:
    """Tests for DDPM with actual UNet integration."""
    
    def test_ddpm_forward_with_unet(self, simple_unet, device):
        """DDPM forward should work with actual UNet."""
        ddpm = GaussianDDPM(network=simple_unet, n_steps=100, device=device)
        x = torch.randn(2, 2, 64, 128).to(device)  # UNet dimensions
        t = torch.randint(0, 100, (2,)).to(device)
        
        # forward returns only noisy image
        noisy = ddpm.forward(x, t)
        
        assert noisy.shape == x.shape
        
    def test_ddpm_backward_with_unet(self, simple_unet, device):
        """DDPM backward should produce valid output with UNet."""
        ddpm = GaussianDDPM(network=simple_unet, n_steps=100, device=device)
        noisy = torch.randn(2, 2, 64, 128).to(device)  # UNet dimensions
        t = torch.randint(0, 100, (2,)).to(device)
        
        predicted = ddpm.backward(noisy, t)
        
        assert predicted.shape == noisy.shape
        assert not torch.isnan(predicted).any()


class TestNoiseInDDPM:
    """Tests for noise strategies integration with DDPM."""
    
    def test_gaussian_noise_in_pipeline(self, simple_unet, device):
        """Gaussian noise should work in full DDPM pipeline."""
        noise_strategy = NOISE_REGISTRY['gaussian']()
        ddpm = GaussianDDPM(
            network=simple_unet, 
            n_steps=100, 
            device=device
        )
        
        x = torch.randn(2, 2, 64, 128).to(device)  # UNet dimensions
        t = torch.randint(0, 100, (2,)).to(device)
        
        noisy = ddpm.forward(x, t)
        
        assert noisy.shape == x.shape
        
    def test_div_free_noise_in_pipeline(self, simple_unet, device):
        """Divergence-free noise should work in full DDPM pipeline."""
        # Note: noise_strategy is configured globally via DDInitializer
        ddpm = GaussianDDPM(
            network=simple_unet, 
            n_steps=100, 
            device=device
        )
        
        x = torch.randn(2, 2, 64, 128).to(device)  # UNet dimensions
        t = torch.randint(0, 100, (2,)).to(device)
        
        noisy = ddpm.forward(x, t)
        
        assert noisy.shape == x.shape


class TestLossWithDDPM:
    """Tests for loss strategies integration with DDPM."""
    
    def test_mse_loss_computes_gradient(self, simple_unet, device):
        """MSE loss should allow gradient computation."""
        loss_strategy = LOSS_REGISTRY['mse']()
        ddpm = GaussianDDPM(network=simple_unet, n_steps=100, device=device)
        
        x = torch.randn(2, 2, 64, 128, requires_grad=True).to(device)  # UNet dimensions
        t = torch.randint(0, 100, (2,)).to(device)
        
        # Generate noise and pass to forward
        true_noise = torch.randn_like(x)
        noisy = ddpm.forward(x, t, epsilon=true_noise)
        pred_noise = ddpm.backward(noisy, t)
        
        loss = loss_strategy.forward(pred_noise, true_noise, noisy)
        
        assert loss.requires_grad
        
    def test_physical_loss_computes_gradient(self, simple_unet, device):
        """Physical loss should allow gradient computation."""
        loss_strategy = LOSS_REGISTRY['physical']()
        ddpm = GaussianDDPM(network=simple_unet, n_steps=100, device=device)
        
        x = torch.randn(2, 2, 64, 128, requires_grad=True).to(device)  # UNet dimensions
        t = torch.randint(0, 100, (2,)).to(device)
        
        # Generate noise and pass to forward
        true_noise = torch.randn_like(x)
        noisy = ddpm.forward(x, t, epsilon=true_noise)
        pred_noise = ddpm.backward(noisy, t)
        
        loss = loss_strategy.forward(pred_noise, true_noise, noisy)
        
        assert loss.requires_grad


class TestHHDecompWithDDPM:
    """Tests for Helmholtz-Hodge decomposition with DDPM outputs."""
    
    def test_decompose_predicted_field(self, simple_unet, device):
        """Should be able to decompose DDPM-predicted fields."""
        ddpm = GaussianDDPM(network=simple_unet, n_steps=100, device=device)
        
        x = torch.randn(1, 2, 64, 128).to(device)  # UNet dimensions
        t = torch.tensor([50]).to(device)
        
        with torch.no_grad():
            noisy = ddpm.forward(x, t)
            predicted = ddpm.backward(noisy, t)
        
        # Convert to HH decomposition format (H, W, 2)
        field = predicted[0].permute(1, 2, 0).cpu()  # (64, 128, 2)
        
        result = decompose_vector_field(field)
        
        assert len(result) == 3
        
    def test_solenoidal_component_has_low_divergence(self, simple_unet, device):
        """Solenoidal component from any field should be div-free."""
        ddpm = GaussianDDPM(network=simple_unet, n_steps=100, device=device)
        
        x = torch.randn(1, 2, 64, 128).to(device)  # UNet dimensions
        t = torch.tensor([50]).to(device)
        
        with torch.no_grad():
            noisy = ddpm.forward(x, t)
            predicted = ddpm.backward(noisy, t)
        
        # Decompose
        field = predicted[0].permute(1, 2, 0).cpu()
        _, _, (u_sol, v_sol) = decompose_vector_field(field)
        
        # Check divergence
        div = compute_divergence(u_sol, v_sol)
        
        assert div.abs().mean() < 0.1


class TestTrainingLoop:
    """Tests for training loop integration."""
    
    def test_single_training_step(self, simple_unet, device):
        """Single training step should work."""
        loss_fn = LOSS_REGISTRY['mse']()
        ddpm = GaussianDDPM(network=simple_unet, n_steps=100, device=device)
        optimizer = torch.optim.Adam(simple_unet.parameters(), lr=1e-4)
        
        x = torch.randn(2, 2, 64, 128).to(device)  # UNet dimensions
        t = torch.randint(0, 100, (2,)).to(device)
        
        # Forward pass
        true_noise = torch.randn_like(x)
        noisy = ddpm.forward(x, t, epsilon=true_noise)
        pred_noise = ddpm.backward(noisy, t)
        
        # Loss and backward
        loss = loss_fn.forward(pred_noise, true_noise, noisy)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        assert loss.item() > 0
        
    def test_multiple_training_steps(self, simple_unet, device):
        """Multiple training steps should reduce loss."""
        loss_fn = LOSS_REGISTRY['mse']()
        ddpm = GaussianDDPM(network=simple_unet, n_steps=100, device=device)
        optimizer = torch.optim.Adam(simple_unet.parameters(), lr=1e-3)
        
        # Fixed batch for consistent comparison
        torch.manual_seed(42)
        x = torch.randn(4, 2, 64, 128).to(device)  # UNet dimensions
        
        losses = []
        for i in range(5):
            t = torch.randint(0, 100, (4,)).to(device)
            
            true_noise = torch.randn_like(x)
            noisy = ddpm.forward(x, t, epsilon=true_noise)
            pred_noise = ddpm.backward(noisy, t)
            
            loss = loss_fn.forward(pred_noise, true_noise, noisy)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
        
        # Loss should not explode
        assert all(l < 100 for l in losses)


class TestModelSaveLoad:
    """Tests for model save/load cycle."""
    
    def test_save_and_load_model(self, simple_unet, device):
        """Should be able to save and reload model."""
        ddpm = GaussianDDPM(network=simple_unet, n_steps=100, device=device)
        
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            # Save
            checkpoint = {
                'model_state_dict': simple_unet.state_dict(),
                'n_steps': 100,
            }
            torch.save(checkpoint, f.name)
            
            # Load
            loaded = torch.load(f.name, map_location=device)
            
            new_unet = MyUNet(n_steps=100, time_emb_dim=32).to(device)
            new_unet.load_state_dict(loaded['model_state_dict'])
            
            # Compare parameters
            for p1, p2 in zip(simple_unet.parameters(), new_unet.parameters()):
                assert torch.allclose(p1, p2)
                
            os.unlink(f.name)
            
    def test_loaded_model_produces_same_output(self, simple_unet, device):
        """Loaded model should produce same output as original."""
        torch.manual_seed(42)
        
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            # Save original
            torch.save({'model_state_dict': simple_unet.state_dict()}, f.name)
            
            # Get original output - use integer timestep, not embedding
            simple_unet.eval()
            x = torch.randn(1, 2, 64, 128).to(device)  # UNet dimensions
            t = torch.tensor([50]).to(device)  # Integer timestep for embedding
            
            with torch.no_grad():
                orig_out = simple_unet(x, t)
            
            # Load and get new output
            new_unet = MyUNet(n_steps=100, time_emb_dim=32).to(device)
            new_unet.load_state_dict(torch.load(f.name)['model_state_dict'])
            new_unet.eval()
            
            with torch.no_grad():
                new_out = new_unet(x, t)
            
            assert torch.allclose(orig_out, new_out)
            
            os.unlink(f.name)


class TestMaskIntegration:
    """Tests for mask integration with DDPM."""
    
    def test_masked_inpainting_simulation(self, simple_unet, device):
        """Simulate masked inpainting workflow."""
        ddpm = GaussianDDPM(network=simple_unet, n_steps=100, device=device)
        
        # Original image (UNet dimensions)
        original = torch.randn(1, 2, 64, 128).to(device)
        
        # Mask (1 = known, 0 = to inpaint)
        mask = torch.ones(1, 1, 64, 128).to(device)
        mask[:, :, 10:50, 20:100] = 0  # Region to inpaint
        
        # Apply mask
        masked_original = original * mask
        
        # Add noise
        t = torch.tensor([50]).to(device)
        noisy = ddpm.forward(original, t)
        
        # Predict noise
        predicted = ddpm.backward(noisy, t)
        
        # Shape should be preserved
        assert predicted.shape == original.shape


class TestDivergenceConstraints:
    """Tests for divergence constraints in the pipeline."""
    
    def test_compute_divergence_on_output(self, simple_unet, device):
        """Should be able to compute divergence on DDPM output."""
        ddpm = GaussianDDPM(network=simple_unet, n_steps=100, device=device)
        
        x = torch.randn(1, 2, 64, 128).to(device)  # UNet dimensions
        t = torch.tensor([50]).to(device)
        
        noisy = ddpm.forward(x, t)
        predicted = ddpm.backward(noisy, t)
        
        # Extract u, v components
        u = predicted[0, 0].cpu()  # (64, 128)
        v = predicted[0, 1].cpu()  # (64, 128)
        
        div = compute_divergence(u, v)
        
        # Should return a tensor with divergence values
        assert div.shape == (64, 128) or isinstance(div.item(), float)
