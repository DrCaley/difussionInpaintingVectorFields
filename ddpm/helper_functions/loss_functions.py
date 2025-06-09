import torch
import torch.nn as nn

from noising_process.incompressible_gp.adding_noise.compute_divergence import compute_divergence

# Base class for loss strategies
class LossStrategy(nn.Module):
    def forward(self, predicted_noise: torch.Tensor, target_noise: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Loss strategy must implement forward()")


# === Standard MSE Loss ===
class MSELossStrategy(LossStrategy):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss()

    def forward(self, predicted_noise, target_noise):
        return self.loss_fn(predicted_noise, target_noise)


# === Physical Loss (Weighted Divergence + MSE) ===
class PhysicalLossStrategy(LossStrategy):
    def __init__(self, w1=1.0, w2=1.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.w1 = w1
        self.w2 = w2

    def forward(self, predicted_noise, target_noise):
        mse_loss = self.mse(predicted_noise, target_noise)
        div_loss = self.physical_loss(predicted_noise)
        return self.w1 * mse_loss + self.w2 * div_loss

    @staticmethod
    def physical_loss(predicted: torch.Tensor) -> torch.Tensor:
        batch_divs = []
        for field in predicted:
            u, v = field[0], field[1]
            div = compute_divergence(u, v)
            batch_divs.append(div.pow(2).mean())
        return torch.stack(batch_divs).mean()

LOSS_REGISTRY = {
    "mse": MSELossStrategy,
    "physical": PhysicalLossStrategy
}

def get_loss_strategy(name: str, **kwargs) -> LossStrategy:
    return LOSS_REGISTRY[name](**kwargs)

