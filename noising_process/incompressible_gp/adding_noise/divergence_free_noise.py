import torch
from noising_process.incompressible_gp.adding_noise.compute_divergence import compute_divergence

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

    betas = torch.linspace(0.0001, 0.02, 1000).to(device)
    alphas = (1 - betas).to(device)
    alpha_bars = torch.tensor([torch.prod(alphas[:i + 1]) for i in range(len(alphas))]).to(device)

    batch, _, height, width = data_set.shape
    output = torch.zeros((batch, 2, height, width), device=device)

    for i in range(batch):
        for j in range(t[i]):
            freq = torch.normal(torch.tensor(0.0, device=device), std=torch.sqrt(alpha_bars[i]))
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

def gaussian_divergence_free_noise(data_set: torch.Tensor, t: torch.Tensor, device='cpu') -> torch.Tensor:

    betas = torch.linspace(0.0001, 0.02, 1000).to(device)
    alphas = (1 - betas).to(device)
    alpha_bars = torch.tensor([torch.prod(alphas[:i + 1]) for i in range(len(alphas))]).to(device)
    normalized_noise = normalized_divergence_free_noise(data_set, t, device=device)

    for i in range(normalized_noise.shape[0]):
        magnitude = torch.normal(torch.tensor(0.0, device=device), std=torch.sqrt(alpha_bars[i]))
        normalized_noise[i] *= magnitude

    return normalized_noise
