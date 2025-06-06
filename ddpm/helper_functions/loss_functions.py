import torch
import torch.nn as nn

from noising_process.incompressible_gp.adding_noise.compute_divergence import compute_divergence

w1 = 1
w2 = 0

mean_square_error = nn.MSELoss()

def physical_loss(predicted: torch.Tensor) -> torch.Tensor:
    """
    Computes the mean squared divergence across a batch of predicted vector fields.
    `predicted` shape: (batch_size, 2, H, W) — where 2 corresponds to (u,v).
    """
    batch_divs = []
    for field in predicted:
        u, v = field[0], field[1]  # Get components
        div = compute_divergence(u, v)  # Shape (H, W)
        batch_divs.append(div.pow(2).mean())  # MSE of divergence for one field
    return torch.stack(batch_divs).mean()


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, predicted_noise, noise):
        loss = w1 * mean_square_error(predicted_noise, noise) + w2 * physical_loss(predicted_noise)
        return loss
