import torch
from gaussian_process.incompressible_gp.adding_noise.divergence_free_noise_test import compute_divergence

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

def divergence_free_noise(tensor: torch.Tensor, t: torch.Tensor, device='cpu') -> torch.Tensor:
    batch, _, height, width = tensor.shape
    output = torch.zeros((batch, 2, height, width), device=device)

    for i in range(batch):
        for j in range(t[i]):
            freq = torch.normal(torch.tensor(0.01),std=1)  # Not sure about this
            vx, vy = exact_div_free_field_from_stream(width, height, freq, device=device)
            output[i, 0] += vx  # Accumulate vx
            output[i, 1] += vy  # Accumulate vy
        print(compute_divergence(output[i, 0], output[i, 1]).sum())

    return output
