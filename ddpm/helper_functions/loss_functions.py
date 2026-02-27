"""Loss strategies for DDPM training.

Each strategy computes a scalar loss from (predicted, target, noisy_img).
The ``noisy_img`` parameter is optional for pure reconstruction losses
but required for physics-based losses that reconstruct x₀.

See ``ddpm.protocols.LossStrategyProtocol`` for the formal contract.

Registry
--------
Concrete strategies are registered in ``LOSS_REGISTRY`` and resolved
by name via ``get_loss_strategy()``.
"""

import torch
import torch.nn as nn
from ddpm.helper_functions.compute_divergence import compute_divergence


class LossStrategy(nn.Module):
    """Base class for all loss strategies.

    Extends ``nn.Module`` so loss parameters (if any) are visible to
    the optimizer.  Subclasses must override ``forward()``.

    See Also
    --------
    ddpm.protocols.LossStrategyProtocol
    """
    def forward(self, predicted_noise: torch.Tensor, target_noise: torch.Tensor, noisy_img : torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Loss strategy must implement forward()")


class MSELossStrategy(LossStrategy):
    """Simple mean-squared-error loss: ||ε̂ − ε||².

    Ignores ``noisy_img`` — operates directly on the predicted
    and target noise (or x₀, depending on prediction target).
    """
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss()

    def forward(self, predicted_noise, target_noise, noisy_img = None) -> torch.Tensor:
        return self.loss_fn(predicted_noise, target_noise)


class PhysicalLossStrategy(LossStrategy):
    """MSE + divergence penalty on the predicted noise.

    ``loss = w1 * MSE(ε̂, ε) + w2 * mean(div(ε̂)²)``

    The divergence penalty encourages the model to produce
    noise with low divergence, biasing outputs toward physically
    plausible (incompressible) velocity fields.

    Parameters
    ----------
    w1, w2 : float
        Weights for MSE and divergence terms.
    """
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
    """MSE + divergence comparison on unstandardized fields.

    Reconstructs x₀ = x_t − ε̂, unstandardizes both predicted and
    target, then penalizes the divergence difference in physical
    units.  Couples directly to the ``DDInitializer`` singleton
    for the standardizer — this is a known coupling concern.

    ``loss = w1 * MSE(ε̂, ε) + w2 * 100 * MSE(div(f̂), div(f))``

    where f̂ and f are the unstandardized predicted and target fields.

    Parameters
    ----------
    w1, w2 : float
        Weights for MSE and physics terms.
    """
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
    "best_loss": HotGarbage
}

def get_loss_strategy(name: str, **kwargs) -> LossStrategy:
    return LOSS_REGISTRY[name](**kwargs)

