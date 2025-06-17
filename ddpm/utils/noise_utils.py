import torch
from typing import Optional

from noising_process.incompressible_gp.adding_noise.divergence_free_noise import \
    gaussian_each_step_divergence_free_noise, layered_div_free_noise
from noise_database.noise_query import QueryNoise


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

    def get_gaussian_scaling(self) -> bool: return True

class GaussianNoise(NoiseStrategy):
    def generate(
        self,
        shape: torch.Size,
        t: Optional[torch.Tensor] = None,
        device: torch.device = None
    ) -> torch.Tensor:
        return torch.randn(shape, device=device)

    def get_gaussian_scaling(self):
        return True

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

    def get_gaussian_scaling(self):
        return False

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

    def get_gaussian_scaling(self):
        return True
class CachedNoisedStrategy(NoiseStrategy):
    def __init__(self,):
        self.noise_query = QueryNoise(noise_dir="../../noise")

    def generate(
            self,
            shape: torch.Size,
            t: Optional[torch.Tensor] = None,
            device: Optional[torch.device] = None
            ) -> torch.Tensor:
        assert shape[1] == 2, "Expected 2 channels (u, v)"
        batch_size = shape[0]
        assert t is not None and len(t) == batch_size, "Timestep tensor must match batch size"

        # Query noise for each t, assume each is (1, H, W)
        noises = []
        for timestep in t:
            sample = self.noise_query.get(int(timestep.item()))  # shape (1, H, W)
            sample = sample.squeeze(0)  # → (H, W)
            # Duplicate channels → (2, H, W)
            sample = sample.unsqueeze(0).repeat(2, 1, 1)
            noises.append(sample)

        return torch.stack(noises, dim=0)  # → (B, 2, H, W)

    def get_gaussian_scaling(self) -> bool:
        return False

NOISE_REGISTRY = {
    "gaussian": GaussianNoise,
    "div_free": DivergenceFreeNoise,
    "div_gaussian": DivergenceFreeGaussianNoise,
    "cached_div" : CachedNoisedStrategy
}

def get_noise_strategy(name: str) -> NoiseStrategy:
    return NOISE_REGISTRY[name]()
