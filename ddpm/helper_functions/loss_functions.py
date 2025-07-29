import torch
import torch.nn as nn
from ddpm.helper_functions.compute_divergence import compute_divergence

# Base class for loss strategies
class LossStrategy(nn.Module):
    def forward(self, predicted_noise: torch.Tensor, target_noise: torch.Tensor, noisy_img : torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Loss strategy must implement forward()")

# wrapper for using loss functions with polar coordinates
class PolarLossWrapper(LossStrategy):
    def __init__(self, base_loss: LossStrategy, coordinate_mode: str = "cartesian"):
        """
        Wraps a base loss strategy to optionally convert inputs from polar to cartesian.
        coordinate_mode: either "cartesian" or "polar"
        """
        super().__init__()
        assert coordinate_mode in ["cartesian", "polar"], "coordinate_mode must be 'cartesian' or 'polar'"
        self.base_loss = base_loss
        self.coordinate_mode = coordinate_mode

    def forward(self, predicted_noise, target_noise, noisy_img=None) -> torch.Tensor:
        if self.coordinate_mode == "polar":
            # Convert polar → cartesian
            predicted_noise = self.polar_to_cartesian(predicted_noise)
            target_noise = self.polar_to_cartesian(target_noise)
        return self.base_loss(predicted_noise, target_noise, noisy_img)

    @staticmethod
    def polar_to_cartesian(tensor: torch.Tensor) -> torch.Tensor:
        """
        Expects tensor shape (B, 2, H, W): (magnitude, angle)
        Returns (B, 2, H, W): (u, v)
        """
        mag = tensor[:, 0]
        ang = tensor[:, 1]
        u = mag * torch.cos(ang)
        v = mag * torch.sin(ang)
        return torch.stack((u, v), dim=1)

# === Standard MSE Loss ===
class MSELossStrategy(LossStrategy):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss()

    def forward(self, predicted_noise, target_noise, noisy_img = None) -> torch.Tensor:
        return self.loss_fn(predicted_noise, target_noise)

# === Physical Loss (Weighted Divergence + MSE) ===
class PhysicalLossStrategy(LossStrategy):
    def __init__(self, w1=1.0, w2=1.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.w1 = w1
        self.w2 = w2

    def forward(self, predicted_noise, target_noise, noisy_img = None) -> torch.Tensor:
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

class HotGarbage(LossStrategy):
    def __init__(self, w1=1.0, w2=1.0):
        super().__init__()
        self.w1 = w1
        self.w2 = w2

        from data_prep.data_initializer import DDInitializer
        dd = DDInitializer()

        self.standardizer = dd.get_standardizer()
        self.mse = nn.MSELoss()

    def forward(self, predicted_noise, target_noise, noisy_img):
        # 1. MSE Loss between noise predictions
        mse_loss = self.mse(predicted_noise, target_noise)

        # 2. Unstandardize predicted and target velocity fields
        prediction = noisy_img - predicted_noise
        real = noisy_img - target_noise

        unstandardized_prediction = self.standardizer.unstandardize(prediction)
        unstandardized_real = self.standardizer.unstandardize(real)

        # 3. Physical loss: divergence between prediction and real
        div_loss = self.physical_loss(unstandardized_prediction, unstandardized_real) * 100

        # 4. Weighted combination
        return self.w1 * mse_loss + self.w2 * div_loss

    @staticmethod
    def physical_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes the mean squared error between the divergence of predicted and real fields.
        """
        batch_divs = []
        for pred_field, target_field in zip(predicted, target):
            u_pred, v_pred = pred_field[0], pred_field[1]
            u_real, v_real = target_field[0], target_field[1]

            div_pred = compute_divergence(u_pred, v_pred)
            div_real = compute_divergence(u_real, v_real)

            div_mse = (div_pred - div_real).pow(2).mean()
            batch_divs.append(div_mse)

        return torch.stack(batch_divs).mean()


LOSS_REGISTRY = {
    "mse": MSELossStrategy,
    "physical": PhysicalLossStrategy,
    "best_loss": HotGarbage,
}

def get_loss_strategy(name: str, **kwargs) -> LossStrategy:
    return LOSS_REGISTRY[name](**kwargs)

