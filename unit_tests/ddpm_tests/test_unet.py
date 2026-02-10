"""
Tests for the MyUNet architecture - the backbone neural network.
"""
import torch
import pytest

from ddpm.neural_networks.unets.unet_xl import MyUNet, sinusoidal_embedding, my_block


@pytest.fixture
def unet():
    """Create a UNet for testing."""
    return MyUNet(n_steps=1000, time_emb_dim=100)


@pytest.fixture
def sample_input():
    """Standard input shape: (batch, channels, H, W)."""
    return torch.randn(2, 2, 64, 128)


class TestUNetArchitecture:
    """Tests for UNet structure and output shapes."""
    
    def test_unet_output_shape_matches_input(self, unet, sample_input):
        """Output should have same shape as input."""
        t = torch.tensor([[500], [750]])  # batch of time steps
        
        output = unet(sample_input, t)
        
        assert output.shape == sample_input.shape
        
    def test_unet_batch_size_one(self, unet):
        """Should work with batch size of 1."""
        x = torch.randn(1, 2, 64, 128)
        t = torch.tensor([[500]])
        
        output = unet(x, t)
        
        assert output.shape == (1, 2, 64, 128)
        
    def test_unet_larger_batch(self, unet):
        """Should work with larger batches."""
        x = torch.randn(8, 2, 64, 128)
        t = torch.tensor([[i * 100] for i in range(8)])
        
        output = unet(x, t)
        
        assert output.shape == (8, 2, 64, 128)
        
    def test_unet_different_timesteps_produce_different_outputs(self, unet, sample_input):
        """Different timesteps should produce different outputs."""
        t_early = torch.tensor([[100], [100]])
        t_late = torch.tensor([[900], [900]])
        
        output_early = unet(sample_input, t_early)
        output_late = unet(sample_input, t_late)
        
        assert not torch.allclose(output_early, output_late, atol=1e-5)
        
    def test_unet_same_timestep_same_output(self, unet, sample_input):
        """Same input and timestep should give same output (deterministic)."""
        t = torch.tensor([[500], [500]])
        
        output1 = unet(sample_input, t)
        output2 = unet(sample_input, t)
        
        assert torch.allclose(output1, output2)


class TestSinusoidalEmbedding:
    """Tests for the sinusoidal time embedding."""
    
    def test_embedding_correct_shape(self):
        """Embedding should have shape (n_steps, dim)."""
        n_steps = 1000
        dim = 100
        
        embedding = sinusoidal_embedding(n_steps, dim)
        
        assert embedding.shape == (n_steps, dim)
        
    def test_embedding_values_bounded(self):
        """Embedding values should be in [-1, 1] (sin/cos range)."""
        embedding = sinusoidal_embedding(1000, 100)
        
        assert embedding.min() >= -1.0
        assert embedding.max() <= 1.0
        
    def test_embedding_different_timesteps_are_different(self):
        """Different timesteps should have different embeddings."""
        embedding = sinusoidal_embedding(1000, 100)
        
        assert not torch.allclose(embedding[0], embedding[500])
        assert not torch.allclose(embedding[100], embedding[900])
        
    def test_embedding_is_deterministic(self):
        """Same parameters should give same embedding."""
        emb1 = sinusoidal_embedding(100, 64)
        emb2 = sinusoidal_embedding(100, 64)
        
        assert torch.allclose(emb1, emb2)


class TestMyBlock:
    """Tests for the custom convolutional block."""
    
    def test_block_output_shape(self):
        """Block should maintain spatial dimensions with padding=1."""
        block = my_block(
            shape=(16, 64, 128),
            in_c=16,
            out_c=32,
            kernel_size=3,
            stride=1,
            padding=1
        )
        
        x = torch.randn(2, 16, 64, 128)
        output = block(x)
        
        # Spatial dims should be preserved, channels should change
        assert output.shape == (2, 32, 64, 128)
        
    def test_block_with_different_channels(self):
        """Block should handle various channel configurations."""
        block = my_block(
            shape=(2, 64, 128),
            in_c=2,
            out_c=16
        )
        
        x = torch.randn(1, 2, 64, 128)
        output = block(x)
        
        assert output.shape == (1, 16, 64, 128)
        
    def test_block_normalization(self):
        """Block with normalize=True should apply LayerNorm."""
        block_norm = my_block(shape=(16, 32, 64), in_c=16, out_c=16, normalize=True)
        block_no_norm = my_block(shape=(16, 32, 64), in_c=16, out_c=16, normalize=False)
        
        x = torch.randn(2, 16, 32, 64)
        
        # Both should work without error
        out_norm = block_norm(x)
        out_no_norm = block_no_norm(x)
        
        assert out_norm.shape == out_no_norm.shape


class TestUNetTimeConditioning:
    """Tests for time-conditioning in UNet."""
    
    def test_time_embedding_not_trainable(self, unet):
        """Time embedding should be frozen (not trainable)."""
        assert not unet.time_embed.weight.requires_grad
        
    def test_time_embedding_shape(self, unet):
        """Time embedding weight should have correct shape."""
        # n_steps x time_emb_dim
        assert unet.time_embed.weight.shape == (1000, 100)


class TestUNetGradients:
    """Tests for gradient flow through UNet."""
    
    def test_unet_has_gradients(self, unet, sample_input):
        """UNet should have gradients after backward pass."""
        sample_input.requires_grad_(True)
        t = torch.tensor([[500], [500]])
        
        output = unet(sample_input, t)
        loss = output.sum()
        loss.backward()
        
        assert sample_input.grad is not None
        
    def test_unet_parameters_have_gradients(self, unet, sample_input):
        """UNet parameters should have gradients after backward."""
        t = torch.tensor([[500], [500]])
        
        output = unet(sample_input, t)
        loss = output.sum()
        loss.backward()
        
        # Check that at least some parameters have gradients
        has_grad = False
        for param in unet.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
                
        assert has_grad


class TestUNetMemory:
    """Tests for memory efficiency."""
    
    def test_unet_eval_mode(self, unet, sample_input):
        """UNet should work in eval mode."""
        unet.eval()
        t = torch.tensor([[500], [500]])
        
        with torch.no_grad():
            output = unet(sample_input, t)
            
        assert output.shape == sample_input.shape
