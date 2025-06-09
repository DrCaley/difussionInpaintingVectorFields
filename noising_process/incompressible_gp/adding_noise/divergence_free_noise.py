import torch
import yaml
import os
import torchvision.transforms as T

from data_prep.data_initializer import DDInitializer
from noising_process.incompressible_gp.adding_noise.compute_divergence import compute_divergence
from plots.plot_data_tool import plot_vector_field

# Get the directory where this script lives
base_dir = os.path.dirname(os.path.abspath(__file__))
dd = DDInitializer()

n_steps = dd.get_attribute('n_steps')
min_beta = dd.get_attribute('min_beta')
max_beta = dd.get_attribute('max_beta')

u_mean = dd.get_attribute('u_mean')
v_mean = dd.get_attribute('v_mean')

alpha_bars = dd.get_alpha_bars()

def exact_div_free_field_from_stream(H, W, freq, device='cpu'):
    x = torch.linspace(0, 2 * torch.pi, W, device=device)
    y = torch.linspace(0, 2 * torch.pi, H, device=device)
    X, Y = torch.meshgrid(x, y, indexing='ij')

    phase_x, phase_y = 2 * torch.pi * torch.rand(2, device=device)
    psi = torch.sin(freq * X + phase_x) * torch.sin(freq * Y + phase_y)

    vx = torch.zeros_like(psi)
    vy = torch.zeros_like(psi)

    vx[:-1, :] = psi[1:, :] - psi[:-1, :]
    vx[-1, :] = 0

    vy[:, :-1] = -(psi[:, 1:] - psi[:, :-1])
    vy[:, -1] = 0

    return vx, vy

def divergence_free_noise(data_set: torch.Tensor, t: torch.Tensor, device='cpu') -> torch.Tensor:

    batch, _, height, width = data_set.shape
    output = torch.zeros((batch, 2, height, width), device=device)

    for i in range(batch):
        for j in range(t[i].item()):
            freq = torch.normal(torch.tensor(0.0, device=device), std=1.0)
            vx, vy = exact_div_free_field_from_stream(width, height, freq, device=device)
            output[i, 0] += vx  # Accumulate vx
            output[i, 1] += vy  # Accumulate vy

    return output

def normalized_divergence_free_noise(data_set: torch.Tensor, t: torch.Tensor, device='cpu') -> torch.Tensor:
    unnormalized_noise = divergence_free_noise(data_set, t, device=device)  # shape: (B, 2, H, W)
    normalized_batches = []

    for batch in unnormalized_noise:
        vx = batch[0]
        vy = batch[1]

        magnitude = torch.sqrt(vx**2 + vy**2)
        max_val = torch.max(magnitude)

        normalized_batch = batch / max_val
        normalized_batches.append(normalized_batch)

    normalized_noise = torch.stack(normalized_batches, dim=0)  # shape: (B, 2, H, W)
    return normalized_noise

def gaussian_at_end_divergence_free_noise(data_set: torch.Tensor, t: torch.Tensor, device='cpu') -> torch.Tensor:

    normalized_noise = normalized_divergence_free_noise(data_set, t, device=device)

    for i in range(normalized_noise.shape[0]):
        magnitude = torch.normal(torch.tensor((u_mean + v_mean) / 2, device=device), std=torch.sqrt(alpha_bars[t[i]]))
        normalized_noise[i] *= magnitude

    return normalized_noise

def gaussian_each_step_divergence_free_noise(data_set: torch.Tensor, t: torch.Tensor, device='cpu') -> torch.Tensor:
    os.makedirs("noise_images", exist_ok=True)

    batch, _, height, width = data_set.shape
    output = torch.zeros((batch, 2, height, width), device=device)

    for i in range(batch):
        t_i = int(t[i].item())  # Make this once and use below
        alpha_bar_val = alpha_bars[t_i].to(device)  # Move to device if needed

        for _ in range(t_i):
            mean = torch.tensor((u_mean + v_mean) / 2, device=device)
            std = torch.sqrt(alpha_bar_val)

            freq = torch.normal(mean, std)
            vx, vy = exact_div_free_field_from_stream(width, height, freq, device=device)

            magnitude = torch.sqrt(vx ** 2 + vy ** 2)
            max_val = torch.max(magnitude)

            if max_val > 0:
                vx /= max_val
                vy /= max_val

            mag_scale = torch.normal(mean, std)
            vx *= mag_scale
            vy *= mag_scale

            output[i, 0] += vx
            output[i, 1] += vy

    return output


import torch
import torch.nn.functional as F


def stream_function_noise(input_tensor: torch.Tensor, sigma=1.5) -> torch.Tensor:

    assert input_tensor.dim() == 4 and input_tensor.size(1) == 2, \
        "Input must have shape (B, 2, H, W)"

    B, _, H, W = input_tensor.shape
    device = input_tensor.device

    psi = torch.randn(B, 1, H, W, device=device)

    blur = T.GaussianBlur(kernel_size=7, sigma=sigma)
    psi = blur(psi)

    kernel_x = torch.tensor([[-0.5, 0, 0.5]], device=device).reshape(1, 1, 1, 3)
    kernel_y = torch.tensor([[-0.5], [0], [0.5]], device=device).reshape(1, 1, 3, 1)

    dpsi_dx = F.conv2d(psi, kernel_x, padding=(0, 1))
    dpsi_dy = F.conv2d(psi, kernel_y, padding=(1, 0))

    vx = dpsi_dy
    vy = -dpsi_dx

    divergence_free = torch.cat([vx, vy], dim=1)
    return divergence_free



