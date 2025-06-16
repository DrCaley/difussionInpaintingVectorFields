from abc import ABC, abstractmethod
import torch

# === Abstract base class ===
class Standardizer(ABC):
    @abstractmethod
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def unstandardize(self, tensor: torch.Tensor) -> torch.Tensor:
        pass

# === Z-score implementation ===
class ZScoreStandardizer(Standardizer):
    def __init__(self, u_mean, u_std, v_mean, v_std):
        self.u_mean = u_mean
        self.u_std = u_std
        self.v_mean = v_mean
        self.v_std = v_std

    def __call__(self, tensor):
        u = (tensor[0:1] - self.u_mean) / self.u_std
        v = (tensor[1:2] - self.v_mean) / self.v_std
        return torch.cat((u, v), dim=0)

    def unstandardize(self, tensor):
        u = tensor[0:1] * self.u_std + self.u_mean
        v = tensor[1:2] * self.v_std + self.v_mean
        return torch.cat((u, v), dim=0)

class MaxMagnitudeStandardizer(Standardizer):
    def __init__(self):
        self.last_max_mag = None  # Store the last magnitude used

    def __call__(self, tensor):
        self.last_max_mag = self.get_max_magnitude(tensor)
        return tensor / self.last_max_mag if self.last_max_mag > 0 else tensor

    def get_max_magnitude(self, tensor):
        u = tensor[0]
        v = tensor[1]
        return torch.sqrt(u**2 + v**2).max()

    def unstandardize(self, tensor):
        if self.last_max_mag is None:
            raise ValueError("Must call __call__ before unstandardize.")
        return tensor * self.last_max_mag

class UnitVectors:
    def __init__(self, eps=1e-8):
        self.eps = eps

    def __call__(self, tensor):
        mag = torch.sqrt(tensor[0:1]**2 + tensor[1:2]**2 + self.eps)
        u = tensor[0:1] / mag
        v = tensor[1:2] / mag
        return torch.cat((u, v), dim=0)


# === Registry ===
STANDARDIZER_REGISTRY = {
    "zscore": ZScoreStandardizer,
    "maxmag": MaxMagnitudeStandardizer,
    "units": UnitVectors,
}