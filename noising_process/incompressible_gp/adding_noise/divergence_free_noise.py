import torch
import yaml
import os

from noising_process.incompressible_gp.adding_noise.compute_divergence import compute_divergence
from plots.plot_data_tool import plot_vector_field

# Get the directory where this script lives
base_dir = os.path.dirname(os.path.abspath(__file__))

# Construct possible config file paths
possible_paths = [
    os.path.join(base_dir, '../../../data.yaml'),
    os.path.join(base_dir, 'data.yaml')
]

for path in possible_paths:
    if os.path.exists(path):
        with open(path, 'r') as file:
            config = yaml.safe_load(file)
        break
else:
    raise FileNotFoundError(f"Could not find data.yaml in any expected location: {possible_paths}")

n_steps, min_beta, max_beta = config['n_steps'], config['min_beta'], config['max_beta']
u_mean, v_mean = config['u_training_mean'], config['v_training_mean']

betas = torch.linspace(min_beta, max_beta, n_steps)
alphas = (1 - betas)
alpha_bars = torch.tensor([torch.prod(alphas[:i + 1]) for i in range(len(alphas))])

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
        for j in range(t[i]):
            freq = torch.normal(torch.tensor( (u_mean + v_mean) / 2, device=device), std=torch.sqrt(alpha_bars[t[i]]))
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

import os

def gaussian_each_step_divergence_free_noise(data_set: torch.Tensor, t: torch.Tensor, device='cpu') -> torch.Tensor:
    # Ensure output directory exists
    os.makedirs("noise_images", exist_ok=True)

    batch, _, height, width = data_set.shape
    output = torch.zeros((batch, 2, height, width), device=device)

    for i in range(batch):
        for j in range(int(t[i].item())):
            freq = torch.normal(torch.tensor((u_mean + v_mean) / 2, device=device), std=torch.sqrt(alpha_bars[int(t[i])]))
            vx, vy = exact_div_free_field_from_stream(width, height, freq, device=device)

            magnitude = torch.sqrt(vx ** 2 + vy ** 2)
            max_val = torch.max(magnitude)
            vx /= max_val
            vy /= max_val

            magnitude = torch.normal(torch.tensor((u_mean + v_mean) / 2, device=device), std=torch.sqrt(alpha_bars[int(t[i])]))
            vx *= magnitude
            vy *= magnitude

            output[i, 0] += vx
            output[i, 1] += vy

    return output

