import torch
from typing import Optional

from noising_process.incompressible_gp.adding_noise.divergence_free_noise import \
    gaussian_each_step_divergence_free_noise, layered_div_free_noise


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
        return gaussian_each_step_divergence_free_noise(shape=shape, t=t, device=device)

class DivergenceFreeGaussianNoise(NoiseStrategy):
    def generate(
        self,
        shape: torch.Size,
        t: Optional[torch.Tensor] = None,
        device: torch.device = None
    ) -> torch.Tensor:
        assert shape[1] == 2, "Divergence-free noise expects 2 channels"
        batch, _, H, W = shape
        return layered_div_free_noise(batch, H, W, device=device)

NOISE_REGISTRY = {
    "gaussian": GaussianNoise,
    "div_free": DivergenceFreeNoise,
    "div_gaussian": DivergenceFreeGaussianNoise
}

def get_noise_strategy(name: str) -> NoiseStrategy:
    return NOISE_REGISTRY[name]()
