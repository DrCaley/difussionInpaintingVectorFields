"""Data standardization strategies for DDPM training.

Each strategy normalizes (2, H, W) vector fields to roughly unit
variance and provides an inverse to recover physical magnitudes.

See ``ddpm.protocols.StandardizerProtocol`` for the formal contract.

Compatibility constraint
------------------------
If the noise strategy is divergence-free, the standardizer MUST use
a unified scale (same divisor for u and v) to preserve the div-free
property.  ``UnifiedZScoreStandardizer`` is the only built-in
standardizer that satisfies this.

Registry
--------
Concrete strategies are registered in ``STANDARDIZER_REGISTRY``.
"""

from abc import ABC, abstractmethod
import torch


class Standardizer(ABC):
    """Base class for all data standardization strategies.

    Subclasses must implement ``__call__`` (standardize) and
    ``unstandardize`` (inverse).

    See Also
    --------
    ddpm.protocols.StandardizerProtocol
    """

    @abstractmethod
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Standardize a (2, H, W) or (B, 2, H, W) vector field."""
        pass

    @abstractmethod
    def unstandardize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Inverse of ``__call__`` — recover physical units."""
        pass

class ZScoreStandardizer(Standardizer):
    """Per-component z-score: separate mean/std for u and v.

    WARNING: This breaks the divergence-free property because
    ``div(standardized) = du/(std_u·dx) + dv/(std_v·dy) ≠ 0``
    when ``std_u ≠ std_v``.  Use ``UnifiedZScoreStandardizer``
    with div-free noise strategies.
    """
    def __init__(self, u_training_mean, u_training_std, v_training_mean, v_training_std):
        self.u_mean = u_training_mean
        self.u_std = u_training_std
        self.v_mean = v_training_mean
        self.v_std = v_training_std

    def __call__(self, tensor):
        u = (tensor[0:1] - self.u_mean) / self.u_std
        v = (tensor[1:2] - self.v_mean) / self.v_std
        return torch.cat((u, v), dim=0)

    def unstandardize(self, tensor):
        u = tensor[0:1] * self.u_std + self.u_mean
        v = tensor[1:2] * self.v_std + self.v_mean
        return torch.cat((u, v), dim=0)


class UnifiedZScoreStandardizer(Standardizer):
    """Z-score standardizer using same std for both components.

    IMPORTANT: This preserves the divergence-free property of vector fields.

    The standard ZScoreStandardizer uses different stds for u and v, which breaks
    divergence-free fields because:
        div_std = (1/std_u) * du/dx + (1/std_v) * dv/dy  !=  0

    With unified std:
        div_std = (1/std) * (du/dx + dv/dy) = (1/std) * 0 = 0  (preserved!)
    """
    def __init__(self, shared_mean, shared_std):
        self.mean = shared_mean
        self.std = shared_std

    def __call__(self, tensor):
        return (tensor - self.mean) / self.std

    def unstandardize(self, tensor):
        return tensor * self.std + self.mean

class MaxMagnitudeStandardizer(Standardizer):
    """Divide by the maximum velocity magnitude.

    WARNING: Stateful — stores ``last_max_mag`` from the most recent
    ``__call__``.  Not safe for batched unstandardize across differently
    scaled inputs.
    """
    def __init__(self):
        self.last_max_mag = None  # Store the last magnitude used

    def __call__(self, tensor):
        self.last_max_mag = self.get_max_magnitude(tensor)
        return tensor / self.last_max_mag if self.last_max_mag > 0 else tensor

    def get_max_magnitude(self, tensor):
        u = tensor[:, 0]  # shape [1, H, W]
        v = tensor[:, 1]  # shape [1, H, W]
        return torch.sqrt(u**2 + v**2).max().item()

    def unstandardize(self, tensor):
        if self.last_max_mag is None:
            raise ValueError("Must call __call__ before unstandardize.")
        return tensor * self.last_max_mag

class UnitVectorNormalizer(Standardizer):
    """Normalize each pixel's vector to unit length.

    WARNING: Stateful — stores per-pixel magnitudes for inversion.
    Destroys magnitude information; only direction is preserved.
    """
    def __init__(self):
        self.eps = 1e-8
        self.last_magnitudes = None  # Store per-vector magnitudes

    def __call__(self, tensor):
        # Compute per-vector magnitude
        mag = torch.sqrt(tensor[0:1]**2 + tensor[1:2]**2 + self.eps)
        self.last_magnitudes = mag  # Save magnitudes for later

        u = tensor[0:1] / mag
        v = tensor[1:2] / mag
        return torch.cat((u, v), dim=0)
    
    def unstandardize(self, tensor):
        if self.last_magnitudes is None:
            raise ValueError("Must call __call__ before unstandardize.")

        u = tensor[0:1] * self.last_magnitudes
        v = tensor[1:2] * self.last_magnitudes
        return torch.cat((u, v), dim=0)


# === Registry ===
STANDARDIZER_REGISTRY = {
    "zscore": ZScoreStandardizer,
    "zscore_unified": UnifiedZScoreStandardizer,
    "maxmag": MaxMagnitudeStandardizer,
    "units": UnitVectorNormalizer,
}