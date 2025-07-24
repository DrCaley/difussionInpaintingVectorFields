import os

import torch
from typing import Optional

from noising_process.incompressible_gp.adding_noise.divergence_free_noise import \
    gaussian_each_step_divergence_free_noise, layered_div_free_noise, gaussian_divergence_free_noise, hh_decomped_div_free_noise


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
        return gaussian_each_step_divergence_free_noise(shape=shape, t=torch.tensor([50]), device=device)

    def get_gaussian_scaling(self):
        return True

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
    
class HH_Decomp_Div_Free(NoiseStrategy):
    def generate(
        self,
        shape: torch.Size,
        t: Optional[torch.Tensor] = None,
        device: torch.device = None
    ) -> torch.Tensor:
        assert shape[1] == 2, "Divergence-free noise expects 2 channels"
        batch, _, H, W = shape
        return hh_decomped_div_free_noise(batch, H, W, device=device)

    def get_gaussian_scaling(self):
        return False

    def generate(
            self,
            shape: torch.Size,
            t: Optional[torch.Tensor] = None,
            device: Optional[torch.device] = None
            ) -> torch.Tensor:
        assert shape[1] == 2, "Expected 2 channels (u, v)"
        batch_size = shape[0]
        assert t is not None and len(t) == batch_size, "Timestep tensor must match batch size"

        noises = []
        for timestep in t:
            sample = self.noise_query.get(int(timestep.item()))  # shape: (1, 2, H, W)

            if sample.ndim == 4 and sample.shape[1] == 2:
                sample = sample.squeeze(0)  # (2, H, W)
            elif sample.ndim == 3:
                pass  # assume already (2, H, W)
            else:
                raise RuntimeError(f"Unexpected noise shape: {sample.shape}")

            noises.append(sample)

        return torch.stack(noises, dim=0)  # Final shape: (B, 2, H, W)

    def get_gaussian_scaling(self) -> bool:
        return False

NOISE_REGISTRY = {
    "gaussian": GaussianNoise,
    "div_free": DivergenceFreeNoise,
    "div_gaussian": DivergenceFreeGaussianNoise,
    "hh_decomp_div_free": HH_Decomp_Div_Free,
}

def get_noise_strategy(name: str) -> NoiseStrategy:
    return NOISE_REGISTRY[name]()
