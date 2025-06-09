import torch
import numpy as np
from typing import Optional

class NoiseStrategy:
    def __call__(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.generate(x.shape, t=t, device=x.device)

    def generate(
        self,
        shape: torch.Size,
        t: Optional[torch.Tensor] = None,
        device: torch.device = None
    ) -> torch.Tensor:
        raise NotImplementedError("Noise strategy must implement generate()")

class GaussianNoise(NoiseStrategy):
    def generate(
        self,
        shape: torch.Size,
        t: Optional[torch.Tensor] = None,
        device: torch.device = None
    ) -> torch.Tensor:
        return torch.randn(shape, device=device)

class DivergenceFreeNoise(NoiseStrategy):
    def generate(
        self,
        shape: torch.Size,
        t: Optional[torch.Tensor] = None,
        device: torch.device = None
    ) -> torch.Tensor:
        assert shape[1] == 2, "Divergence-free noise expects 2 channels"
        batch, _, H, W = shape
        potential = torch.randn((batch, 1, H, W), device=device)
        dx = (torch.roll(potential, -1, dims=3) - torch.roll(potential, 1, dims=3)) / 2
        dy = (torch.roll(potential, -1, dims=2) - torch.roll(potential, 1, dims=2)) / 2
        return torch.cat([-dy, dx], dim=1)

NOISE_REGISTRY = {
    "gaussian": GaussianNoise,
    "div_free": DivergenceFreeNoise
}

def get_noise_strategy(name: str) -> NoiseStrategy:
    return NOISE_REGISTRY[name]()
