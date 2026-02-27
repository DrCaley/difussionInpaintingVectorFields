"""Noise generation strategies for the DDPM forward process.

Every strategy must produce (B, 2, H, W) noise with approximately
unit variance so the DDPM noise-schedule assumptions hold.

See ``ddpm.protocols.NoiseStrategyProtocol`` for the formal contract.

Registry
--------
Concrete strategies are registered in ``NOISE_REGISTRY`` at the bottom
of this module and resolved by name via ``get_noise_strategy()``.
"""

import os

import torch
from typing import Optional

from noising_process.incompressible_gp.adding_noise.divergence_free_noise import (
    gaussian_each_step_divergence_free_noise,
    layered_div_free_noise,
    gaussian_divergence_free_noise,
    hh_decomped_div_free_noise,
)


class NoiseStrategy:
    """Base class for all noise generation strategies.

    Subclasses must override ``generate()`` and optionally
    ``get_gaussian_scaling()``.  The ``__call__`` convenience method
    is inherited and should NOT be overridden.

    See Also
    --------
    ddpm.protocols.NoiseStrategyProtocol
    """

    def __call__(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Convenience: generate noise matching the shape/device of *x*."""
        return self.generate(x.shape, t=t, device=x.device)

    def generate(
        self,
        shape: torch.Size,
        t: Optional[torch.Tensor] = None,
        device: torch.device = None,
    ) -> torch.Tensor:
        """Generate noise of the given shape.  Must be overridden."""
        raise NotImplementedError("Noise strategy must implement generate()")

    def get_gaussian_scaling(self) -> bool:
        """Return True if the DDPM forward process should apply √(1−ᾱ) scaling."""
        return True

class GaussianNoise(NoiseStrategy):
    """Standard i.i.d. Gaussian noise — N(0, I).

    The simplest noise strategy.  Each element is drawn independently
    from a standard normal.  Uses Gaussian scaling (√(1−ᾱ)) in the
    forward process.
    """

    def generate(
        self,
        shape: torch.Size,
        t: Optional[torch.Tensor] = None,
        device: torch.device = None,
    ) -> torch.Tensor:
        return torch.randn(shape, device=device)

    def get_gaussian_scaling(self) -> bool:
        return True

class DivergenceFreeNoise(NoiseStrategy):
    """GP-based divergence-free noise via multi-step accumulation.

    Delegates to ``gaussian_each_step_divergence_free_noise()`` from the
    incompressible GP subsystem.  Timestep-dependent: clamped to t ≥ 1
    to avoid NaNs.

    Requires unified standardizer to preserve div-free after rescaling.
    """

    def generate(
        self,
        shape: torch.Size,
        t: Optional[torch.Tensor] = None,
        device: torch.device = None,
    ) -> torch.Tensor:
        assert shape[1] == 2, "Divergence-free noise expects 2 channels"
        batch, _, H, W = shape
        if t is None:
            t = torch.zeros(batch, dtype=torch.long, device=device)
        elif isinstance(t, int):
            t = torch.full((batch,), t, dtype=torch.long, device=device)
        else:
            t = t.to(device).reshape(-1)
        # Avoid t=0 causing NaNs in divergence-free noise generation.
        t = torch.clamp(t, min=1)
        return gaussian_each_step_divergence_free_noise(shape=shape, t=t, device=device)

    def get_gaussian_scaling(self) -> bool:
        return True

class DivergenceFreeGaussianNoise(NoiseStrategy):
    """Layered divergence-free noise from summed GP fields.

    Delegates to ``layered_div_free_noise()``.  Timestep-independent.

    Requires unified standardizer to preserve div-free after rescaling.
    """

    def generate(
        self,
        shape: torch.Size,
        t: Optional[torch.Tensor] = None,
        device: torch.device = None,
    ) -> torch.Tensor:
        assert shape[1] == 2, "Divergence-free noise expects 2 channels"
        batch, _, H, W = shape
        return layered_div_free_noise(batch, H, W, device=device)

    def get_gaussian_scaling(self) -> bool:
        return True
    
class HH_Decomp_Div_Free(NoiseStrategy):
    """Divergence-free noise via Helmholtz–Hodge decomposition.

    Generates Gaussian noise, then projects out the irrotational part
    using ``hh_decomped_div_free_noise()``.  Does NOT use Gaussian
    scaling (``get_gaussian_scaling() == False``) — the projection
    step changes the magnitude distribution.

    Requires unified standardizer to preserve div-free after rescaling.

    Note
    ----
    A previous version of this class had a duplicate ``generate()``
    method that referenced ``self.noise_query`` — a field that was
    never initialized.  That dead code has been removed.
    """

    def generate(
        self,
        shape: torch.Size,
        t: Optional[torch.Tensor] = None,
        device: torch.device = None,
    ) -> torch.Tensor:
        assert shape[1] == 2, "Divergence-free noise expects 2 channels"
        batch, _, H, W = shape
        return hh_decomped_div_free_noise(batch, H, W, device=device)

    def get_gaussian_scaling(self) -> bool:
        return False

class SpectralDivFreeNoise(NoiseStrategy):
    """
    Divergence-free noise using a streamfunction construction.

    We sample a scalar potential psi and define
        u = dpsi/dy,
        v = -dpsi/dx,
    using centered finite differences on the same grid operator
    used by downstream divergence checks. This yields fields with
    near-zero discrete divergence under that operator.

    The output is then rescaled to unit variance so DDPM forward/
    reverse scaling assumptions remain valid.
    """

    def __init__(self):
        super().__init__()
        self._variance_scale: dict[tuple[int, int, str], float] = {}

    @staticmethod
    def _from_streamfunction(psi: torch.Tensor) -> torch.Tensor:
        """Build a 2-channel div-free field from scalar streamfunction psi.
        psi shape: (B, H, W)
        returns: (B, 2, H, W)
        """
        u = torch.zeros_like(psi)
        v = torch.zeros_like(psi)

        # u = dpsi/dy (row direction)
        u[:, 1:-1, :] = (psi[:, 2:, :] - psi[:, :-2, :]) / 2.0
        # v = -dpsi/dx (column direction)
        v[:, :, 1:-1] = -(psi[:, :, 2:] - psi[:, :, :-2]) / 2.0

        return torch.stack([u, v], dim=1)

    def _get_variance_scale(
        self,
        H: int,
        W: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> float:
        """Compute (and cache) the std of the raw projected noise so we can
        rescale to unit variance.  Uses a deterministic seed for
        reproducibility of the calibration constant."""
        key = (H, W, str(dtype))
        if key not in self._variance_scale:
            rng_state = torch.random.get_rng_state()
            torch.manual_seed(0)
            psi = torch.randn(128, H, W, device=device, dtype=dtype)
            test = self._from_streamfunction(psi)
            self._variance_scale[key] = test.std().item()
            torch.random.set_rng_state(rng_state)
        return self._variance_scale[key]

    def generate(
        self,
        shape: torch.Size,
        t: Optional[torch.Tensor] = None,
        device: torch.device = None,
    ) -> torch.Tensor:
        B, C, H, W = shape
        assert C == 2, "SpectralDivFreeNoise requires 2 channels (u, v)"

        psi = torch.randn(B, H, W, device=device)
        noise = self._from_streamfunction(psi)

        # Rescale so per-element variance ≈ 1
        scale = self._get_variance_scale(H, W, device, noise.dtype)
        if scale > 0:
            noise = noise / scale

        return noise

    def get_gaussian_scaling(self) -> bool:
        return True


class ForwardDiffDivFreeNoise(NoiseStrategy):
    """
    Divergence-free noise using a streamfunction with **forward** differences.

    We sample a scalar potential ψ of shape (B, H+1, W+1) and define:
        u[i,j] = ψ[i+1,j] - ψ[i,j]       (forward diff in row direction)
        v[i,j] = -(ψ[i,j+1] - ψ[i,j])    (neg forward diff in col direction)

    This yields fields with EXACTLY zero discrete divergence under the
    forward-difference operator:
        div = (u[i,j+1] - u[i,j]) + (v[i+1,j] - v[i,j]) = 0

    Unlike SpectralDivFreeNoise (which is only div-free under central
    differences), this is div-free in the intuitive pixel-neighbor sense.

    No boundary artifacts: ψ is (H+1, W+1) so every pixel gets valid
    forward differences without padding or zero-fill.

    Output is rescaled to unit variance for DDPM compatibility.
    """

    def __init__(self):
        super().__init__()
        self._variance_scale: dict[tuple[int, int, str], float] = {}

    @staticmethod
    def _from_streamfunction(psi: torch.Tensor) -> torch.Tensor:
        """Build a 2-channel div-free field from scalar streamfunction psi.
        psi shape: (B, H+1, W+1)
        returns: (B, 2, H, W)  where H = psi.shape[1]-1, W = psi.shape[2]-1
        """
        # u[i,j] = psi[i+1,j] - psi[i,j], cols 0..W-1
        u = psi[:, 1:, :-1] - psi[:, :-1, :-1]
        # v[i,j] = -(psi[i,j+1] - psi[i,j]), rows 0..H-1
        v = -(psi[:, :-1, 1:] - psi[:, :-1, :-1])

        return torch.stack([u, v], dim=1)

    def _get_variance_scale(
        self,
        H: int,
        W: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> float:
        key = (H, W, str(dtype))
        if key not in self._variance_scale:
            rng_state = torch.random.get_rng_state()
            torch.manual_seed(0)
            psi = torch.randn(128, H + 1, W + 1, device=device, dtype=dtype)
            test = self._from_streamfunction(psi)
            self._variance_scale[key] = test.std().item()
            torch.random.set_rng_state(rng_state)
        return self._variance_scale[key]

    def generate(
        self,
        shape: torch.Size,
        t: Optional[torch.Tensor] = None,
        device: torch.device = None,
    ) -> torch.Tensor:
        B, C, H, W = shape
        assert C == 2, "ForwardDiffDivFreeNoise requires 2 channels (u, v)"

        psi = torch.randn(B, H + 1, W + 1, device=device)
        noise = self._from_streamfunction(psi)

        scale = self._get_variance_scale(H, W, device, noise.dtype)
        if scale > 0:
            noise = noise / scale

        return noise

    def get_gaussian_scaling(self) -> bool:
        return True


class ForwardDiffEqualizedDivFreeNoise(NoiseStrategy):
    """
    Spectrally-equalized divergence-free noise via a coloured streamfunction.

    Uses the same forward-difference curl as ``ForwardDiffDivFreeNoise``
    (and is therefore EXACTLY divergence-free under the same discrete
    operator), but **colours ψ in Fourier space** so that the resulting
    velocity field has an approximately *flat* (white) power spectrum.

    Problem solved
    --------------
    The plain forward-diff curl is a high-pass filter:
        |curl ψ̂(k)|² ∝ 4 sin²(πk_y/(H+1)) + 4 sin²(πk_x/(W+1))
    White ψ therefore produces velocity noise with almost no low-frequency
    energy.  DDPM models trained on such noise cannot reconstruct the
    large-scale (low-k) structure of ocean currents.

    Fix
    ---
    Before taking the curl we multiply ψ̂(k) by 1 / √G(k), where
        G(k) = 4 sin²(πk_y/(H+1)) + 4 sin²(πk_x/(W+1))
    is the energy transfer function of the forward-difference curl.
    The product G × 1/G = 1 yields (approximately) white velocity noise.
    The DC component (k = 0) is set to zero — the mean of ψ is
    irrelevant because the curl of a constant is zero.

    Output is rescaled to unit per-element variance for DDPM compatibility.
    """

    def __init__(self):
        super().__init__()
        self._variance_scale: dict[tuple[int, int, str], float] = {}
        self._filter_cache: dict[tuple[int, int, str], torch.Tensor] = {}

    # ── coloring filter ──────────────────────────────────────────────
    def _get_coloring_filter(
        self,
        H: int,
        W: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return 1/√G filter for the (H+1, W+1) ψ-grid DFT.

        G(m, n) = 4 sin²(π m / (H+1)) + 4 sin²(π n / (W+1))

        The filter is cached so it is computed only once per resolution.
        """
        key = (H, W, str(dtype))
        if key in self._filter_cache:
            return self._filter_cache[key].to(device)

        Hp, Wp = H + 1, W + 1
        m = torch.arange(Hp, dtype=dtype)
        n = torch.arange(Wp, dtype=dtype)

        sin2_y = torch.sin(torch.pi * m / Hp) ** 2          # (Hp,)
        sin2_x = torch.sin(torch.pi * n / Wp) ** 2          # (Wp,)
        G = 4.0 * sin2_y.unsqueeze(1) + 4.0 * sin2_x.unsqueeze(0)  # (Hp, Wp)

        filt = torch.zeros_like(G)
        nonzero = G > 1e-12
        filt[nonzero] = 1.0 / G[nonzero].sqrt()

        self._filter_cache[key] = filt
        return filt.to(device)

    # ── same curl operator as ForwardDiffDivFreeNoise ────────────────
    @staticmethod
    def _from_streamfunction(psi: torch.Tensor) -> torch.Tensor:
        """Forward-diff curl:  u = dψ/dy,  v = −dψ/dx."""
        u = psi[:, 1:, :-1] - psi[:, :-1, :-1]
        v = -(psi[:, :-1, 1:] - psi[:, :-1, :-1])
        return torch.stack([u, v], dim=1)

    # ── variance calibration ─────────────────────────────────────────
    def _get_variance_scale(
        self,
        H: int,
        W: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> float:
        key = (H, W, str(dtype))
        if key not in self._variance_scale:
            rng_state = torch.random.get_rng_state()
            torch.manual_seed(0)
            psi_white = torch.randn(128, H + 1, W + 1, device=device, dtype=dtype)
            filt = self._get_coloring_filter(H, W, device, dtype)
            psi_fft = torch.fft.fft2(psi_white)
            psi_colored = torch.fft.ifft2(psi_fft * filt.unsqueeze(0)).real
            test = self._from_streamfunction(psi_colored)
            self._variance_scale[key] = test.std().item()
            torch.random.set_rng_state(rng_state)
        return self._variance_scale[key]

    # ── generate ─────────────────────────────────────────────────────
    def generate(
        self,
        shape: torch.Size,
        t: Optional[torch.Tensor] = None,
        device: torch.device = None,
    ) -> torch.Tensor:
        B, C, H, W = shape
        assert C == 2, "ForwardDiffEqualizedDivFreeNoise requires 2 channels"

        # 1. White-noise streamfunction on the enlarged grid
        psi_white = torch.randn(B, H + 1, W + 1, device=device)

        # 2. Colour in Fourier space:  ψ̂ → ψ̂ / √G(k)
        filt = self._get_coloring_filter(H, W, device, psi_white.dtype)
        psi_colored = torch.fft.ifft2(
            torch.fft.fft2(psi_white) * filt.unsqueeze(0)
        ).real

        # 3. Forward-diff curl (exactly div-free)
        noise = self._from_streamfunction(psi_colored)

        # 4. Rescale to unit per-element variance
        scale = self._get_variance_scale(H, W, device, noise.dtype)
        if scale > 0:
            noise = noise / scale

        return noise

    def get_gaussian_scaling(self) -> bool:
        return True


NOISE_REGISTRY = {
    "gaussian": GaussianNoise,
    "div_free": DivergenceFreeNoise,
    "div_gaussian": DivergenceFreeGaussianNoise,
    "hh_decomp_div_free": HH_Decomp_Div_Free,
    "spectral_div_free": SpectralDivFreeNoise,
    "forward_diff_div_free": ForwardDiffDivFreeNoise,
    "fwd_diff_eq_divfree": ForwardDiffEqualizedDivFreeNoise,
}

def get_noise_strategy(name: str) -> NoiseStrategy:
    return NOISE_REGISTRY[name]()


def get_noise_type_name(strategy: NoiseStrategy) -> str:
    """Reverse-lookup: return the registry key for a NoiseStrategy instance."""
    for name, cls in NOISE_REGISTRY.items():
        if isinstance(strategy, cls):
            return name
    return "gaussian"  # safe fallback
