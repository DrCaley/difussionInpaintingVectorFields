"""
Tests for the GaussianDDPM class - the core diffusion model.
"""
import torch
import pytest
from ddpm.neural_networks.ddpm import GaussianDDPM


@pytest.fixture
def simple_ddpm():
    """Create a simple DDPM with an identity network for testing."""
    identity_net = torch.nn.Identity()
    return GaussianDDPM(
        network=identity_net,
        n_steps=100,
        min_beta=1e-4,
        max_beta=0.02,
        device='cpu'
    )


@pytest.fixture
def sample_image():
    """Create a sample image tensor for testing."""
    return torch.randn(2, 2, 64, 128)  # batch=2, channels=2, H=64, W=128


class TestDDPMForward:
    """Tests for the forward diffusion process."""
    
    def test_ddpm_forward_adds_noise(self, simple_ddpm, sample_image):
        """Verify that forward diffusion increases variance (adds noise)."""
        t = torch.tensor([50, 50])  # middle timestep
        epsilon = torch.randn_like(sample_image)
        
        noisy = simple_ddpm(sample_image, t, epsilon)
        
        # Noisy image should be different from original
        assert not torch.allclose(noisy, sample_image, atol=1e-5)
        
    def test_ddpm_forward_output_shape(self, simple_ddpm, sample_image):
        """Verify output shape matches input shape."""
        t = torch.tensor([25, 75])
        epsilon = torch.randn_like(sample_image)
        
        noisy = simple_ddpm(sample_image, t, epsilon)
        
        assert noisy.shape == sample_image.shape
        
    def test_ddpm_forward_uses_provided_noise(self, simple_ddpm, sample_image):
        """Verify that the same noise produces the same output."""
        t = torch.tensor([50, 50])
        epsilon = torch.randn_like(sample_image)
        
        noisy1 = simple_ddpm(sample_image, t, epsilon)
        noisy2 = simple_ddpm(sample_image, t, epsilon)
        
        assert torch.allclose(noisy1, noisy2)
        
    def test_ddpm_forward_generates_noise_if_not_provided(self, simple_ddpm, sample_image):
        """Verify noise is generated if epsilon=None."""
        t = torch.tensor([50, 50])
        
        noisy1 = simple_ddpm(sample_image, t, epsilon=None)
        noisy2 = simple_ddpm(sample_image, t, epsilon=None)
        
        # Different calls should produce different results (random noise)
        assert not torch.allclose(noisy1, noisy2)

    def test_ddpm_one_step_vs_cumulative(self, simple_ddpm, sample_image):
        """Verify one_step=True uses alpha[t] vs alpha_bar[t]."""
        t = torch.tensor([50, 50])
        epsilon = torch.randn_like(sample_image)
        
        noisy_cumulative = simple_ddpm(sample_image, t, epsilon, one_step=False)
        noisy_single = simple_ddpm(sample_image, t, epsilon, one_step=True)
        
        # These should be different since they use different scaling
        assert not torch.allclose(noisy_cumulative, noisy_single)


class TestDDPMSchedule:
    """Tests for the noise schedule (betas, alphas, alpha_bars)."""
    
    def test_betas_are_positive(self, simple_ddpm):
        """All betas should be positive."""
        assert (simple_ddpm.betas > 0).all()
        
    def test_betas_monotonically_increase(self, simple_ddpm):
        """Betas should increase from min_beta to max_beta."""
        diffs = simple_ddpm.betas[1:] - simple_ddpm.betas[:-1]
        assert (diffs >= 0).all()
        
    def test_alphas_are_one_minus_betas(self, simple_ddpm):
        """Alphas should equal 1 - betas."""
        expected = 1 - simple_ddpm.betas
        assert torch.allclose(simple_ddpm.alphas, expected)
        
    def test_alpha_bars_monotonically_decrease(self, simple_ddpm):
        """Alpha_bars should decrease (more noise at later timesteps)."""
        diffs = simple_ddpm.alpha_bars[1:] - simple_ddpm.alpha_bars[:-1]
        assert (diffs <= 0).all()
        
    def test_alpha_bars_are_cumulative_products(self, simple_ddpm):
        """Alpha_bars should be cumulative products of alphas."""
        for i in range(len(simple_ddpm.alpha_bars)):
            expected = torch.prod(simple_ddpm.alphas[:i + 1])
            assert torch.isclose(simple_ddpm.alpha_bars[i], expected, rtol=1e-5)
            
    def test_alpha_bar_at_start_near_one(self, simple_ddpm):
        """At t=0, alpha_bar should be close to 1 (little noise)."""
        assert simple_ddpm.alpha_bars[0] > 0.99
        
    def test_alpha_bar_at_end_near_zero(self, simple_ddpm):
        """At t=n_steps-1, alpha_bar should be small (lots of noise)."""
        # With n_steps=50 and default beta range, may not be very small
        assert simple_ddpm.alpha_bars[-1] < 0.5


class TestDDPMBackward:
    """Tests for the backward (denoising) process."""
    
    def test_backward_calls_network(self):
        """Verify backward passes through the network."""
        # Create a mock network that we can verify was called
        class MockNetwork(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.called = False
                
            def forward(self, x, t):
                self.called = True
                return torch.randn_like(x)
        
        mock_net = MockNetwork()
        ddpm = GaussianDDPM(mock_net, n_steps=100, device='cpu')
        
        x = torch.randn(1, 2, 64, 128)
        t = torch.tensor([[50]])
        
        ddpm.backward(x, t)
        
        assert mock_net.called


class TestDDPMDevice:
    """Tests for device handling."""
    
    def test_ddpm_tensors_on_correct_device(self):
        """Verify all tensors are on the specified device."""
        device = 'cpu'
        identity_net = torch.nn.Identity()
        ddpm = GaussianDDPM(identity_net, n_steps=50, device=device)
        
        assert ddpm.betas.device.type == device
        assert ddpm.alphas.device.type == device
        assert ddpm.alpha_bars.device.type == device
