import torch
import os

# Get the directory where this script lives
base_dir = os.path.dirname(os.path.abspath(__file__))

def get_dd_initializer():
    from data_prep.data_initializer import DDInitializer
    return DDInitializer()

# Used for plain divergence free noise

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

def gaussian_each_step_divergence_free_noise(shape: torch.Size, t: torch.Tensor, device='cpu') -> torch.Tensor:
    dd = get_dd_initializer()

    u_mean = dd.get_attribute('u_training_mean')
    v_mean = dd.get_attribute('v_training_mean')

    alpha_bars = dd.get_alpha_bars()

    os.makedirs("noise_images", exist_ok=True)

    batch, _, height, width = shape
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

# Used for gaussian divergence free noise

def generate_div_free_noise(batch_size, height, width, device=get_dd_initializer().get_device()):
    psi = torch.randn(batch_size, 1, height, width, device=get_dd_initializer().get_device())

    dy = psi[:, :, 1:, :] - psi[:, :, :-1, :]
    dx = psi[:, :, :, 1:] - psi[:, :, :, :-1]

    dpsi_dy = F.pad(dy, (0, 0, 1, 0))
    dpsi_dx = F.pad(dx, (1, 0, 0, 0))

    u = dpsi_dy
    v = -dpsi_dx

    noise = torch.cat([u, v], dim=1)  # shape: (B, 2, H, W)
    return noise

def layered_div_free_noise(batch_size, height, width, device=get_dd_initializer().get_device(), n_layers=10):
    noise = torch.zeros(batch_size, 2, height, width, device=device)
    for _ in range(n_layers):
        noise += generate_div_free_noise(batch_size, height, width, device)
    return noise / (n_layers ** 0.5)





