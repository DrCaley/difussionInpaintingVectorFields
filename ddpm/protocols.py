"""
Formal protocol definitions for the DDPM inpainting system.

This module defines the structural contracts (interfaces) that all
building blocks must satisfy.  Using ``typing.Protocol`` (PEP 544)
rather than ABC inheritance: any class whose methods match the
protocol is automatically compatible — no registration required.

Architecture overview
---------------------
The DDPM inpainting pipeline is a composition of six pluggable
building blocks:

    ┌──────────────┐
    │   Data Prep  │  OceanImageDataset → standardize → OceanInpaintDataset
    └──────┬───────┘
           │ (x0, t, noise, mask, known_values)
           ▼
    ┌──────────────┐     ┌──────────────────┐
    │   Noiser     │◄────│  NoiseStrategy   │  Generates ε ~ p(ε)
    │  (DDPM fwd)  │     │  (pluggable)     │  Must satisfy: unit variance,
    └──────┬───────┘     └──────────────────┘  optional div-free guarantee
           │ x_t = √ᾱ·x0 + √(1−ᾱ)·ε   (or unscaled if non-Gaussian)
           ▼
    ┌──────────────┐     ┌──────────────────┐
    │   Denoiser   │◄────│  DenoiserNetwork │  UNet that predicts ε or x0
    │  (UNet)      │     │  (pluggable)     │  Input: 2ch (uncond) or 5ch (cond)
    └──────┬───────┘     └──────────────────┘
           │ ε̂  or  x̂₀
           ▼
    ┌──────────────┐     ┌──────────────────┐
    │   Loss       │◄────│  LossStrategy    │  MSE, physics-informed, etc.
    └──────┬───────┘     │  (pluggable)     │
           │             └──────────────────┘
           ▼
    ┌──────────────┐     ┌──────────────────┐
    │  Inpainting  │◄────│  MaskGenerator   │  Defines known/unknown regions
    │  (inference) │     │  (pluggable)     │
    └──────┬───────┘     └──────────────────┘
           │
           ▼
    ┌──────────────┐     ┌──────────────────┐
    │ Post-process │◄────│ DivFreeProjection│  Optional div-free cleanup
    └──────────────┘     │  (pluggable)     │
                         └──────────────────┘

Component dependencies
----------------------
These are the actual data-flow dependencies, validated by
``validate_component_compatibility()`` at the bottom of this module.

- **NoiseStrategy → Standardizer**: If the noise strategy is div-free,
  the standardizer MUST be unified (same std for u,v) to preserve the
  div-free property after standardization.

- **NoiseStrategy → GaussianDDPM.forward()**: The ``get_gaussian_scaling()``
  flag determines whether ε is scaled by √(1−ᾱ) in the forward process.
  Non-Gaussian noise (e.g. precomputed GP fields) skips this scaling.

- **DenoiserNetwork → TrainInpaint**: The UNet type (``film``, ``concat``,
  or future ``unconditional``) must match the training loop's input
  construction.  A FiLM UNet expects 5ch split as 2+3; a concat UNet
  expects 5ch concatenated; an unconditional UNet expects 2ch only.

- **PredictionTarget → InpaintingAlgorithm**: An ``x0``-prediction model
  should use ``x0_full_reverse_inpaint()``; an ``eps``-prediction model
  should use ``repaint_standard()`` or ``mask_aware_inpaint()``.

- **DivFreeProjection → NoiseStrategy**: The projection operator's
  discrete derivative must match the noise construction.  Forward-diff
  noise requires ``forward_diff_project_div_free()``; spectral noise
  can use ``spectral_project_div_free()``.
"""

from __future__ import annotations

from typing import Optional, Protocol, Tuple, runtime_checkable

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 1. Noise Strategy
# ---------------------------------------------------------------------------

@runtime_checkable
class NoiseStrategyProtocol(Protocol):
    """Contract for noise generation strategies.

    A noise strategy generates the stochastic perturbation ε used in the
    DDPM forward process:  x_t = √ᾱ·x₀ + (scale)·ε.

    Implementors
    ------------
    - ``GaussianNoise``:            standard i.i.d. N(0,1)
    - ``SpectralDivFreeNoise``:     central-diff streamfunction curl
    - ``ForwardDiffDivFreeNoise``:  forward-diff streamfunction curl (exact)
    - ``DivergenceFreeNoise``:      GP-based multi-step accumulation
    - ``DivergenceFreeGaussianNoise``: layered div-free sum
    - ``HH_Decomp_Div_Free``:      Helmholtz decomposition of Gaussian noise

    Invariant
    ---------
    ``generate()`` MUST return a tensor with approximately unit variance
    so the DDPM noise schedule assumptions hold.
    """

    def __call__(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Convenience: generate noise matching the shape/device of ``x``."""
        ...

    def generate(
        self,
        shape: torch.Size,
        t: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Generate noise of the given shape.

        Parameters
        ----------
        shape : (B, 2, H, W)
            Must have 2 channels for vector-field noise.
        t : optional timestep tensor (B,)
            Some strategies (e.g. GP-based) need the timestep.
        device : target device

        Returns
        -------
        Tensor of shape ``shape`` with approximately unit variance.
        """
        ...

    def get_gaussian_scaling(self) -> bool:
        """Whether the DDPM forward process should scale ε by √(1−ᾱ).

        Returns True for strategies whose output has standard Gaussian
        marginals (most strategies).  Returns False for precomputed
        noise fields that carry their own magnitude.
        """
        ...


# ---------------------------------------------------------------------------
# 2. Loss Strategy
# ---------------------------------------------------------------------------

@runtime_checkable
class LossStrategyProtocol(Protocol):
    """Contract for training loss functions.

    Implementors
    ------------
    - ``MSELossStrategy``:      ||ε̂ − ε||²
    - ``PhysicalLossStrategy``: MSE + divergence penalty on ε̂
    - ``HotGarbage``:           MSE + div comparison on unstandardized fields

    Notes
    -----
    The ``noisy_img`` parameter is provided so loss functions can
    reconstruct the predicted clean image (x₀ ≈ x_t − ε̂) if needed
    for physics-based penalties.
    """

    def __call__(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        noisy_img: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the scalar loss.

        Parameters
        ----------
        predicted : (B, 2, H, W)
            Model output (predicted noise or predicted x₀).
        target : (B, 2, H, W)
            Ground-truth (true noise or true x₀).
        noisy_img : (B, 2, H, W), optional
            The noisy input x_t — only needed by physics losses that
            reconstruct x₀ = x_t − ε̂.
        """
        ...


# ---------------------------------------------------------------------------
# 3. Standardizer
# ---------------------------------------------------------------------------

@runtime_checkable
class StandardizerProtocol(Protocol):
    """Contract for data standardization.

    Standardizers normalize ocean current fields to zero-mean,
    unit-variance (or similar) for stable DDPM training.

    Implementors
    ------------
    - ``ZScoreStandardizer``:         per-component (u,v) z-score
    - ``UnifiedZScoreStandardizer``:  shared std — preserves div-free!
    - ``MaxMagnitudeStandardizer``:   divide by max magnitude
    - ``UnitVectorNormalizer``:       per-pixel unit vectors

    Compatibility constraint
    ------------------------
    If the noise strategy is divergence-free, the standardizer MUST
    use a unified scale (same divisor for u and v) to preserve the
    div-free property.  Otherwise:
        div(standardized) = du/(std_u·dx) + dv/(std_v·dy) ≠ 0
    even when div(original) = 0.
    """

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Standardize a (2, H, W) or (B, 2, H, W) vector field."""
        ...

    def unstandardize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Inverse of ``__call__`` — recover physical units."""
        ...


# ---------------------------------------------------------------------------
# 4. Mask Generator
# ---------------------------------------------------------------------------

@runtime_checkable
class MaskGeneratorProtocol(Protocol):
    """Contract for inpainting mask generators.

    Convention
    ----------
    ``1.0`` = missing / to-inpaint,  ``0.0`` = known / observed.

    This is the opposite of some vision libraries but consistent
    throughout this codebase.

    Implementors
    ------------
    - ``RobotPathGenerator``:     BFS greedy coverage path
    - ``StraightLinePathGenerator``: random-angle transect
    - ``RandomMaskGenerator``:    rectangular known window
    - ``GaussianNoiseBinaryMaskGenerator``: thresholded Gaussian
    - ``NoMask``:                 everything known (generation only)
    - ... and many more (see ``ddpm/helper_functions/masks/``)
    """

    def generate_mask(self, image_shape: Optional[tuple] = None) -> torch.Tensor:
        """Return a (1, H, W) or (H, W) binary mask.

        Values: 1.0 = missing, 0.0 = known.
        """
        ...

    def __str__(self) -> str:
        """Human-readable name for logging/plotting."""
        ...

    def get_num_lines(self) -> int:
        """Number of observation paths/lines (for path-based masks)."""
        ...


# ---------------------------------------------------------------------------
# 5. Denoiser Network
# ---------------------------------------------------------------------------

@runtime_checkable
class DenoiserNetworkProtocol(Protocol):
    """Contract for the neural network inside the DDPM.

    The denoiser takes a noisy vector field (possibly concatenated with
    conditioning) and a timestep embedding, and predicts either the
    noise ε or the clean image x₀.

    Implementors
    ------------
    - ``MyUNet``       (unet_xl.py):     2-channel unconditional
    - ``MyUNet_Inpaint`` (unet_inpaint.py): 5-channel concat conditioning
    - ``MyUNet_FiLM``  (unet_film.py):   5-channel FiLM conditioning
    - ``PConvUNet``    (pconv_base.py):   partial convolution

    The caller (GaussianDDPM or TrainInpaint) is responsible for
    constructing the correct input tensor width before calling forward.
    """

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict noise (or x₀) from noisy input and timestep.

        Parameters
        ----------
        x : (B, C_in, H, W)
            C_in = 2 for unconditional, 5 for conditioned models.
        t : (B,) integer timesteps ∈ [0, T)

        Returns
        -------
        (B, 2, H, W) — predicted noise or clean image (same spatial dims).
        """
        ...


# ---------------------------------------------------------------------------
# 6. Divergence-Free Projection
# ---------------------------------------------------------------------------

@runtime_checkable
class DivFreeProjectionProtocol(Protocol):
    """Contract for post-hoc divergence-free projection.

    A projection takes an arbitrary (B, 2, H, W) vector field and
    returns the closest divergence-free field under its discrete
    derivative operator.

    Implementors
    ------------
    - ``forward_diff_project_div_free``:  CG streamfunction solver,
      consistent with ``ForwardDiffDivFreeNoise``.
    - ``spectral_project_div_free``:      FFT Helmholtz, consistent
      with ``SpectralDivFreeNoise`` (periodic BC).
    - ``project_div_free_2d``:            Jacobi Poisson solver (5-pt).

    Compatibility constraint
    ------------------------
    The projection's discrete operator MUST match the noise strategy's
    construction method.  Mixing operators (e.g. spectral projection on
    forward-diff noise) breaks idempotency.
    """

    def __call__(self, vel: torch.Tensor) -> torch.Tensor:
        """Project ``vel`` to the nearest div-free field.

        Parameters
        ----------
        vel : (B, 2, H, W)

        Returns
        -------
        (B, 2, H, W) with approximately zero discrete divergence.
        """
        ...


# =========================================================================
# Dependency validation
# =========================================================================

# Maps noise strategy class names → compatible standardizer class names
_NOISE_STANDARDIZER_COMPAT = {
    # Div-free noise strategies REQUIRE unified standardizer
    "SpectralDivFreeNoise": {"UnifiedZScoreStandardizer"},
    "ForwardDiffDivFreeNoise": {"UnifiedZScoreStandardizer"},
    "ForwardDiffEqualizedDivFreeNoise": {"UnifiedZScoreStandardizer"},
    "DivergenceFreeNoise": {"UnifiedZScoreStandardizer"},
    "DivergenceFreeGaussianNoise": {"UnifiedZScoreStandardizer"},
    "HH_Decomp_Div_Free": {"UnifiedZScoreStandardizer"},
    # Gaussian noise works with any standardizer
    "GaussianNoise": {
        "ZScoreStandardizer",
        "UnifiedZScoreStandardizer",
        "MaxMagnitudeStandardizer",
        "UnitVectorNormalizer",
    },
}

# Maps noise strategy class names → compatible projection function names
_NOISE_PROJECTION_COMPAT = {
    "ForwardDiffDivFreeNoise": {"forward_diff_project_div_free"},
    "ForwardDiffEqualizedDivFreeNoise": {"forward_diff_project_div_free"},
    "SpectralDivFreeNoise": {"spectral_project_div_free"},
    "GaussianNoise": {"project_div_free_2d", "spectral_project_div_free",
                       "forward_diff_project_div_free"},
    # Legacy strategies — no strict projection requirement
    "DivergenceFreeNoise": set(),
    "DivergenceFreeGaussianNoise": set(),
    "HH_Decomp_Div_Free": set(),
}

# Maps prediction target → compatible inpainting functions
_PREDICTION_INPAINT_COMPAT = {
    "x0": {"x0_full_reverse_inpaint", "x0_predict_inpaint"},
    "eps": {"repaint_standard", "mask_aware_inpaint",
            "mask_aware_inpaint_cfg", "guided_inpaint",
            "inpaint_generate_new_images"},
}

# Maps unet_type config → expected UNet class name
_UNET_TYPE_CLASS = {
    "film": "MyUNet_FiLM",
    "concat": "MyUNet_Inpaint",
    "standard": "MyUNet",             # 2-channel unconditional (RePaint-style)
    "standard_attn": "MyUNet_Attn",   # 2-channel unconditional with self-attention
}


class ComponentIncompatibilityError(Exception):
    """Raised when two building blocks have incompatible configurations."""
    pass


def validate_noise_standardizer(
    noise_strategy: NoiseStrategyProtocol,
    standardizer: StandardizerProtocol,
) -> None:
    """Verify that the noise strategy and standardizer are compatible.

    Raises ``ComponentIncompatibilityError`` if a div-free noise strategy
    is paired with a per-component standardizer (which would break the
    div-free property).
    """
    noise_cls = type(noise_strategy).__name__
    std_cls = type(standardizer).__name__

    allowed = _NOISE_STANDARDIZER_COMPAT.get(noise_cls)
    if allowed is not None and std_cls not in allowed:
        raise ComponentIncompatibilityError(
            f"Noise strategy '{noise_cls}' requires one of {allowed} "
            f"as standardizer, but got '{std_cls}'.\n"
            f"Reason: div-free noise needs a unified scale to preserve "
            f"the zero-divergence property after standardization."
        )


def validate_noise_projection(
    noise_strategy: NoiseStrategyProtocol,
    projection_fn_name: str,
) -> None:
    """Verify that the projection operator matches the noise construction.

    Raises ``ComponentIncompatibilityError`` if the discrete derivative
    operators are inconsistent (e.g. spectral projection with forward-diff noise).
    """
    noise_cls = type(noise_strategy).__name__
    allowed = _NOISE_PROJECTION_COMPAT.get(noise_cls)
    if allowed and projection_fn_name not in allowed:
        raise ComponentIncompatibilityError(
            f"Noise strategy '{noise_cls}' requires projection from "
            f"{allowed}, but got '{projection_fn_name}'.\n"
            f"Reason: the projection's discrete derivative operator "
            f"must match the noise construction method."
        )


def validate_prediction_inpaint(
    prediction_target: str,
    inpaint_fn_name: str,
) -> None:
    """Verify that the inpainting function matches the prediction target.

    Raises ``ComponentIncompatibilityError`` if an eps-prediction model
    is used with an x0-inpainting algorithm, or vice versa.
    """
    allowed = _PREDICTION_INPAINT_COMPAT.get(prediction_target)
    if allowed is not None and inpaint_fn_name not in allowed:
        raise ComponentIncompatibilityError(
            f"Prediction target '{prediction_target}' is not compatible "
            f"with inpainting function '{inpaint_fn_name}'.\n"
            f"Compatible functions: {allowed}"
        )


def validate_unet_type(
    unet_type: str,
    network: nn.Module,
) -> None:
    """Verify that the UNet instance matches the configured type.

    Raises ``ComponentIncompatibilityError`` if ``unet_type='film'`` but
    the network is actually a ``MyUNet_Inpaint``, etc.
    """
    expected_cls = _UNET_TYPE_CLASS.get(unet_type)
    actual_cls = type(network).__name__
    if expected_cls is not None and actual_cls != expected_cls:
        raise ComponentIncompatibilityError(
            f"Config specifies unet_type='{unet_type}' (expects {expected_cls}) "
            f"but the network is '{actual_cls}'."
        )


def validate_all(
    noise_strategy: NoiseStrategyProtocol,
    standardizer: StandardizerProtocol,
    prediction_target: str = "eps",
    inpaint_fn_name: Optional[str] = None,
    projection_fn_name: Optional[str] = None,
    unet_type: Optional[str] = None,
    network: Optional[nn.Module] = None,
) -> list[str]:
    """Run all compatibility checks and return a list of warnings/errors.

    Raises nothing — collects all issues into the returned list.
    Use this for diagnostics; use individual ``validate_*`` functions
    for hard enforcement.
    """
    issues: list[str] = []

    try:
        validate_noise_standardizer(noise_strategy, standardizer)
    except ComponentIncompatibilityError as e:
        issues.append(str(e))

    if projection_fn_name:
        try:
            validate_noise_projection(noise_strategy, projection_fn_name)
        except ComponentIncompatibilityError as e:
            issues.append(str(e))

    if inpaint_fn_name:
        try:
            validate_prediction_inpaint(prediction_target, inpaint_fn_name)
        except ComponentIncompatibilityError as e:
            issues.append(str(e))

    if unet_type and network:
        try:
            validate_unet_type(unet_type, network)
        except ComponentIncompatibilityError as e:
            issues.append(str(e))

    return issues
