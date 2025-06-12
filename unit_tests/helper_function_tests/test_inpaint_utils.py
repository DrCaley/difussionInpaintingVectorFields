import torch
import pytest
from unittest.mock import MagicMock

from data_prep.data_initializer import DDInitializer
from ddpm.neural_networks.ddpm import MyDDPMGaussian

from ddpm.utils.inpainting_utils import calculate_mse, avg_pixel_value

dd = DDInitializer()

@pytest.fixture
def dummy_data():
    # 4x4 test image, channel=1
    original = torch.tensor([
        [[1, 2, 3, 4],
         [5, 6, 7, 8],
         [9, 10, 11, 12],
         [13, 14, 15, 16]]
    ], dtype=torch.float32)

    predicted = original + 1  # introduce uniform error

    mask = torch.tensor([
        [[0, 1, 1, 0],
         [0, 1, 1, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0]]
    ], dtype=torch.float32)

    return original, predicted, mask


def test_calculate_mse(dummy_data):
    original, predicted, mask = dummy_data
    expected_mse = torch.mean((predicted * mask - original * mask) ** 2)
    result = calculate_mse(original, predicted, mask)
    assert torch.isclose(result, expected_mse, rtol=1e-5)


def test_avg_pixel_value(dummy_data):
    original, predicted, mask = dummy_data
    avg_pixel = torch.sum(torch.abs(original * mask)) / mask.sum()
    avg_diff = torch.sum(torch.abs((predicted * mask) - (original * mask))) / mask.sum()
    expected = avg_diff * (100 / avg_pixel)

    result = avg_pixel_value(original, predicted, mask)
    assert torch.isclose(result, expected, rtol=1e-5)


def test_inpaint_generate_new_images_runs():
    # Mock DDPM and supporting structures
    dummy_ddpm = MagicMock()
    dummy_ddpm.n_steps = 3
    dummy_ddpm.device = torch.device(dd.get_device())
    dummy_ddpm.alphas = torch.tensor([0.9, 0.8, 0.7])
    dummy_ddpm.alpha_bars = torch.tensor([0.9, 0.72, 0.5])
    dummy_ddpm.betas = torch.tensor([0.1, 0.2, 0.3])
    dummy_ddpm.backward.side_effect = lambda x, t: torch.randn_like(x)

    # Correct way to mock __call__
    def dummy_ddpm_call(unnoised_img, t, epsilon, one_step=True):
        return unnoised_img + 0.1 * epsilon

    dummy_ddpm.side_effect = dummy_ddpm_call

    # Mock noise strategy
    dummy_noise_strategy = lambda x, t: torch.randn_like(x)
    dummy_dd = MagicMock()
    dummy_dd.get_noise_strategy.return_value = dummy_noise_strategy

    # Inputs
    input_image = torch.rand(2, 2, 4, 4)  # batch of 2
    mask = torch.ones_like(input_image)

    # Patch dd
    import ddpm.utils.inpainting_utils as inpaint_utils
    inpaint_utils.dd = dummy_dd

    out = inpaint_utils.inpaint_generate_new_images(
        dummy_ddpm,
        input_image,
        mask,
        n_samples=2,
        resample_steps=1,
        channels=1,
        height=4,
        width=4
    )

    assert isinstance(out, torch.Tensor)
    assert out.shape == input_image.shape

def test_forward_reconstructs_image():
    # Setup
    device = torch.device(dd.get_device())
    batch_size, channels, height, width = 1, 2, 64, 128
    t_idx = 50  # middle of the diffusion process

    # Get noise strategy from DDInitializer
    dd = DDInitializer()
    noise_strat = dd.get_noise_strategy()

    # Create dummy input image
    x0 = torch.rand(batch_size, channels, height, width).to(device)

    # Create dummy DDPM
    dummy_net = torch.nn.Identity()
    ddpm = MyDDPMGaussian(network=dummy_net, n_steps=1000, device=device)
    ddpm.eval()

    # Use real forward process with noise from noise_strat
    t_tensor = torch.tensor([t_idx]).to(device).float()
    known_noise = noise_strat(x0, t_tensor)

    x_t = ddpm(x0, t_tensor.long(), epsilon=known_noise, one_step=True)

    # Mock the backward method to return the same known noise
    ddpm.backward = MagicMock(return_value=known_noise)

    # Apply your DDPM denoising equation
    alpha_t = ddpm.alphas[t_idx]
    alpha_bar_t = ddpm.alpha_bars[t_idx]

    if dd.get_attribute('gaussian_scaling'):
        reconstructed = (1 / alpha_t.sqrt()) * (x_t - (1 - alpha_t) / (1 - alpha_bar_t).sqrt() * known_noise)
    else:
        reconstructed = (1 / alpha_t.sqrt()) * (x_t - known_noise)

    # Measure how close we are
    mse = torch.nn.functional.mse_loss(reconstructed, x0)

    print(f"MSE with dd noise strategy: {mse.item()}")
    assert mse < 1e-3, "Reconstructed image is not close enough to the original using DD noise"

def test_backward_learns_to_predict_noise():
    # Define dummy DDPM
    class DummyDDPM:
        def __init__(self):
            self.n_steps = 1
            self.alphas = [torch.tensor(0.9)]
            self.alpha_bars = [torch.tensor(0.9)]

        def backward(self, x_t, t):
            return x_t - x_0  # Perfectly predicts (x_t - x_0) assuming epsilon = x_t - x_0

    dd = DDInitializer()
    noise_strat = dd.get_noise_strategy()

    # Clean image
    x_0 = torch.ones(1, 2, 4, 4) * 0.5
    epsilon_true = noise_strat(x_0, t = torch.tensor([1]))
    alpha_bar_t = torch.tensor(0.9)

    # Forward diffusion
    if dd.get_attribute('gaussian_scaling'):
        x_t = (alpha_bar_t.sqrt() * x_0 + (1 - alpha_bar_t).sqrt() * epsilon_true)
    else:
        x_t = (alpha_bar_t.sqrt() * x_0 + epsilon_true)

    # Time step
    t = torch.tensor([[0]])

    # Backward
    ddpm = DummyDDPM()
    epsilon_pred = ddpm.backward(x_t, t)

    # Check closeness
    assert torch.allclose(epsilon_pred, epsilon_true, atol=1e-2)

