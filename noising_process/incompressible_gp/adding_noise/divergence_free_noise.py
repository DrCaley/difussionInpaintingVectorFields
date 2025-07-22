import torch
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from ddpm.helper_functions.HH_decomp import decompose_vector_field


def get_dd_initializer():
    from data_prep.data_initializer import DDInitializer
    return DDInitializer()

# Used for plain divergence free noise

def exact_div_free_field_from_stream(H, W, freq, device='cpu'):
    dd = get_dd_initializer()
    device = dd.get_device()

    # Create a slightly larger meshgrid for safe finite differences
    x = torch.linspace(0, 2 * torch.pi, W + 1, device=device)
    y = torch.linspace(0, 2 * torch.pi, H + 1, device=device)
    X, Y = torch.meshgrid(x, y, indexing='ij')

    phase_x, phase_y = 2 * torch.pi * torch.rand(2, device=device)
    psi = torch.sin(freq * X + phase_x) * torch.sin(freq * Y + phase_y)

    # Compute forward differences
    vx = psi[1:, :-1] - psi[:-1, :-1]  # dψ/dy (crop to [H, W])
    vy = -(psi[:-1, 1:] - psi[:-1, :-1])  # -dψ/dx (crop to [H, W])

    return vx, vy  # Both are shape (H, W)



# Beta
def gaussian_each_step_divergence_free_noise(shape: torch.Size, t: torch.Tensor, device='cpu') -> torch.Tensor:
    dd = get_dd_initializer()
    device = dd.get_device()

    u_mean = dd.get_attribute('u_training_mean')
    v_mean = dd.get_attribute('v_training_mean')

    betas = dd.get_betas()

    batch, _, height, width = shape
    output = torch.zeros((batch, 2, height, width), device=device)

    for i in range(batch):
        t_i = int(t[i].item())  # Make this once and use below
        beta_val = betas[t_i].to(device)  # Move to device if needed

        for _ in range(t_i):
            mean = torch.tensor((u_mean + v_mean) / 2, device=device)
            std = torch.sqrt(beta_val)

            freq = torch.normal(mean, std)
            vx, vy = exact_div_free_field_from_stream(width, height, freq * 10, device=device)

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

    mean, std, var = torch.mean(output), torch.std(output), torch.var(output)
    output = (output - mean) / std

    return output

# Alpha
def gaussian_divergence_free_noise(shape: torch.Size, t: torch.Tensor, device='cpu') -> torch.Tensor:
    dd = get_dd_initializer()
    device = dd.get_device()

    u_mean = dd.get_attribute('u_training_mean')
    v_mean = dd.get_attribute('v_training_mean')

    alpha_bars = dd.get_alpha_bars()

    batch, _, height, width = shape
    output = torch.zeros((batch, 2, height, width), device=device)

    for i in range(batch):
        t_i = int(t[i].item())  # Make this once and use below
        alpha_bar_val = alpha_bars[t_i].to(device)  # Move to device if needed

        mean = torch.tensor(alpha_bar_val, device=device)
        std = torch.sqrt(1 - alpha_bar_val)

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

def generate_div_free_noise(batch_size, height, width, device=None):
    dd = get_dd_initializer()
    device = dd.get_device()
    psi = torch.randn(batch_size, 1, height, width, device=device)

    dy = psi[:, :, 1:, :] - psi[:, :, :-1, :]
    dx = psi[:, :, :, 1:] - psi[:, :, :, :-1]

    dpsi_dy = F.pad(dy, (0, 0, 1, 0))
    dpsi_dx = F.pad(dx, (1, 0, 0, 0))

    u = dpsi_dy
    v = -dpsi_dx

    noise = torch.cat([u, v], dim=1)  # shape: (B, 2, H, W)
    return noise


# More noise

def layered_div_free_noise(batch_size, height, width, device=None, n_layers=10):
    dd = get_dd_initializer()
    device = dd.get_device()
    noise = torch.zeros(batch_size, 2, height, width, device=device)
    for _ in range(n_layers):
        noise += generate_div_free_noise(batch_size, height, width, device)
    return noise / ( (n_layers ** 0.5) * (2 ** (0.5)) )


# HH Decomp Div-Free noise
def hh_decomped_div_free_noise(batch_size, height, width, device=None):
    dd = get_dd_initializer()
    device = dd.get_device()

    output = torch.zeros(batch_size, 2, height, width, device=device)

    for i in range(batch_size):
        # Generate Gaussian noise vector field
        vec_field = torch.randn(height, width, 2, device=device)

        # Move to CPU for NumPy + FFT handling (scipy.fft only works with NumPy)
        vec_field_cpu = vec_field.detach().cpu()

        # Helmholtz-Hodge decomposition
        (_, _), (_, _), (u_sol, v_sol) = decompose_vector_field(vec_field_cpu)

        # Convert solenoidal part back to torch and send to device
        u_sol = torch.from_numpy(u_sol).to(device)
        v_sol = torch.from_numpy(v_sol).to(device)

        output[i, 0] = u_sol
        output[i, 1] = v_sol

    return output  # Shape: (B, 2, H, W)