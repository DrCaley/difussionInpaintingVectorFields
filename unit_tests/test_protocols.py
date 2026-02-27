"""Conformance tests for DDPM building-block protocols.

Run with:
    PYTHONPATH=. python -m pytest unit_tests/test_protocols.py -v

These tests verify that every concrete implementation satisfies
its ``typing.Protocol`` contract from ``ddpm.protocols``, and that
the dependency validation functions catch known-incompatible pairings.
"""

import pytest
import torch

# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------
from ddpm.protocols import (
    NoiseStrategyProtocol,
    LossStrategyProtocol,
    StandardizerProtocol,
    MaskGeneratorProtocol,
    DenoiserNetworkProtocol,
    DivFreeProjectionProtocol,
    ComponentIncompatibilityError,
    validate_noise_standardizer,
    validate_noise_projection,
    validate_prediction_inpaint,
    validate_unet_type,
    validate_all,
)

# ---------------------------------------------------------------------------
# Concrete implementations
# ---------------------------------------------------------------------------
from ddpm.utils.noise_utils import (
    GaussianNoise,
    SpectralDivFreeNoise,
    ForwardDiffDivFreeNoise,
    HH_Decomp_Div_Free,
    NOISE_REGISTRY,
)
from ddpm.helper_functions.loss_functions import (
    MSELossStrategy,
    PhysicalLossStrategy,
    LOSS_REGISTRY,
)
from ddpm.helper_functions.standardize_data import (
    ZScoreStandardizer,
    UnifiedZScoreStandardizer,
    MaxMagnitudeStandardizer,
    UnitVectorNormalizer,
    STANDARDIZER_REGISTRY,
)

# =========================================================================
# 1. NoiseStrategy protocol conformance
# =========================================================================


class TestNoiseStrategyProtocol:
    """Every registered noise strategy must satisfy NoiseStrategyProtocol."""

    # Only test strategies that don't need external GP dependencies
    SAFE_STRATEGIES = {
        "gaussian": GaussianNoise,
        "spectral_div_free": SpectralDivFreeNoise,
        "forward_diff_div_free": ForwardDiffDivFreeNoise,
    }

    @pytest.mark.parametrize("name,cls", list(SAFE_STRATEGIES.items()))
    def test_isinstance_check(self, name, cls):
        """runtime_checkable Protocol isinstance check passes."""
        strategy = cls()
        assert isinstance(strategy, NoiseStrategyProtocol), (
            f"{name} ({cls.__name__}) does not satisfy NoiseStrategyProtocol"
        )

    @pytest.mark.parametrize("name,cls", list(SAFE_STRATEGIES.items()))
    def test_generate_shape(self, name, cls):
        """generate() returns the correct (B, 2, H, W) shape."""
        strategy = cls()
        shape = torch.Size([4, 2, 16, 32])
        t = torch.randint(0, 100, (4,))
        out = strategy.generate(shape, t=t, device=torch.device("cpu"))
        assert out.shape == shape, f"{name}: expected {shape}, got {out.shape}"

    @pytest.mark.parametrize("name,cls", list(SAFE_STRATEGIES.items()))
    def test_call_convenience(self, name, cls):
        """__call__(x, t) returns noise matching x's shape."""
        strategy = cls()
        x = torch.randn(2, 2, 16, 32)
        t = torch.randint(0, 100, (2,))
        out = strategy(x, t)
        assert out.shape == x.shape

    @pytest.mark.parametrize("name,cls", list(SAFE_STRATEGIES.items()))
    def test_gaussian_scaling_is_bool(self, name, cls):
        """get_gaussian_scaling() returns a bool."""
        strategy = cls()
        result = strategy.get_gaussian_scaling()
        assert isinstance(result, bool), f"{name}: expected bool, got {type(result)}"

    def test_gaussian_noise_unit_variance(self):
        """GaussianNoise should produce approximately unit-variance output."""
        strategy = GaussianNoise()
        out = strategy.generate(torch.Size([1000, 2, 16, 32]))
        var = out.var().item()
        assert 0.9 < var < 1.1, f"GaussianNoise variance = {var}, expected ~1.0"

    def test_forward_diff_near_unit_variance(self):
        """ForwardDiffDivFreeNoise should rescale to approximately unit variance."""
        strategy = ForwardDiffDivFreeNoise()
        out = strategy.generate(torch.Size([256, 2, 16, 32]))
        var = out.var().item()
        assert 0.7 < var < 1.3, f"ForwardDiffDivFreeNoise variance = {var}, expected ~1.0"

    def test_spectral_near_unit_variance(self):
        """SpectralDivFreeNoise should rescale to approximately unit variance."""
        strategy = SpectralDivFreeNoise()
        out = strategy.generate(torch.Size([256, 2, 16, 32]))
        var = out.var().item()
        assert 0.7 < var < 1.3, f"SpectralDivFreeNoise variance = {var}, expected ~1.0"

    def test_forward_diff_div_free_property(self):
        """ForwardDiffDivFreeNoise output must have near-zero discrete divergence.

        Discrete divergence (forward diff):
            div[i,j] = (u[i,j+1] - u[i,j]) + (v[i+1,j] - v[i,j])
        """
        strategy = ForwardDiffDivFreeNoise()
        out = strategy.generate(torch.Size([4, 2, 16, 32]))
        u, v = out[:, 0], out[:, 1]

        du_dx = u[:, :, 1:] - u[:, :, :-1]   # (B, H, W-1)
        dv_dy = v[:, 1:, :] - v[:, :-1, :]   # (B, H-1, W)
        # Interior divergence: (H-1, W-1) overlap
        div = du_dx[:, :-1, :] + dv_dy[:, :, :-1]
        max_div = div.abs().max().item()
        assert max_div < 1e-5, f"ForwardDiffDivFreeNoise max div = {max_div}, expected ~0"


# =========================================================================
# 2. LossStrategy protocol conformance
# =========================================================================


class TestLossStrategyProtocol:
    """Every registered loss strategy must satisfy LossStrategyProtocol."""

    SAFE_LOSSES = {
        "mse": MSELossStrategy,
        "physical": PhysicalLossStrategy,
        # HotGarbage requires DDInitializer — skip in unit tests
    }

    @pytest.mark.parametrize("name,cls", list(SAFE_LOSSES.items()))
    def test_isinstance_check(self, name, cls):
        strategy = cls()
        assert isinstance(strategy, LossStrategyProtocol), (
            f"{name} ({cls.__name__}) does not satisfy LossStrategyProtocol"
        )

    @pytest.mark.parametrize("name,cls", list(SAFE_LOSSES.items()))
    def test_forward_returns_scalar(self, name, cls):
        """forward() must return a scalar tensor."""
        strategy = cls()
        pred = torch.randn(2, 2, 16, 32)
        target = torch.randn(2, 2, 16, 32)
        noisy = torch.randn(2, 2, 16, 32)
        loss = strategy(pred, target, noisy)
        assert loss.ndim == 0, f"{name}: loss should be scalar, got shape {loss.shape}"
        assert loss.item() >= 0, f"{name}: loss should be non-negative"


# =========================================================================
# 3. Standardizer protocol conformance
# =========================================================================


class TestStandardizerProtocol:
    """Every registered standardizer must satisfy StandardizerProtocol."""

    def _make_standardizer(self, name):
        """Instantiate a standardizer by name with dummy params."""
        cls = STANDARDIZER_REGISTRY[name]
        if name == "zscore":
            return cls(0.0, 1.0, 0.0, 1.0)
        elif name == "zscore_unified":
            return cls(0.0, 1.0)
        else:
            return cls()

    @pytest.mark.parametrize("name", ["zscore", "zscore_unified", "maxmag", "units"])
    def test_isinstance_check(self, name):
        std = self._make_standardizer(name)
        assert isinstance(std, StandardizerProtocol), (
            f"{name} does not satisfy StandardizerProtocol"
        )

    @pytest.mark.parametrize("name", ["zscore", "zscore_unified"])
    def test_roundtrip(self, name):
        """__call__ followed by unstandardize should approximately recover the input."""
        std = self._make_standardizer(name)
        x = torch.randn(2, 16, 32)
        y = std(x)
        x_hat = std.unstandardize(y)
        assert torch.allclose(x, x_hat, atol=1e-5), (
            f"{name}: roundtrip error = {(x - x_hat).abs().max().item()}"
        )


# =========================================================================
# 4. Dependency validation
# =========================================================================


class TestDependencyValidation:
    """Validate that the compatibility functions catch bad pairings."""

    def test_divfree_with_unified_passes(self):
        """ForwardDiffDivFreeNoise + UnifiedZScoreStandardizer should pass."""
        noise = ForwardDiffDivFreeNoise()
        std = UnifiedZScoreStandardizer(0.0, 1.0)
        # Should not raise
        validate_noise_standardizer(noise, std)

    def test_divfree_with_zscore_fails(self):
        """ForwardDiffDivFreeNoise + ZScoreStandardizer should fail."""
        noise = ForwardDiffDivFreeNoise()
        std = ZScoreStandardizer(0.0, 1.0, 0.0, 1.0)
        with pytest.raises(ComponentIncompatibilityError):
            validate_noise_standardizer(noise, std)

    def test_spectral_with_zscore_fails(self):
        """SpectralDivFreeNoise + ZScoreStandardizer should fail."""
        noise = SpectralDivFreeNoise()
        std = ZScoreStandardizer(0.0, 1.0, 0.0, 1.0)
        with pytest.raises(ComponentIncompatibilityError):
            validate_noise_standardizer(noise, std)

    def test_gaussian_with_any_standardizer_passes(self):
        """GaussianNoise is compatible with all standardizers."""
        noise = GaussianNoise()
        for name in ["zscore", "zscore_unified", "maxmag", "units"]:
            cls = STANDARDIZER_REGISTRY[name]
            if name == "zscore":
                std = cls(0.0, 1.0, 0.0, 1.0)
            elif name == "zscore_unified":
                std = cls(0.0, 1.0)
            else:
                std = cls()
            # Should not raise
            validate_noise_standardizer(noise, std)

    def test_noise_projection_compat(self):
        """forward_diff noise requires forward_diff projection."""
        noise = ForwardDiffDivFreeNoise()
        validate_noise_projection(noise, "forward_diff_project_div_free")  # ok
        with pytest.raises(ComponentIncompatibilityError):
            validate_noise_projection(noise, "spectral_project_div_free")

    def test_prediction_inpaint_compat(self):
        """x0 prediction should reject eps-only inpainting functions."""
        validate_prediction_inpaint("x0", "x0_full_reverse_inpaint")  # ok
        with pytest.raises(ComponentIncompatibilityError):
            validate_prediction_inpaint("x0", "repaint_standard")

    def test_prediction_inpaint_eps_compat(self):
        """eps prediction should reject x0-only inpainting functions."""
        validate_prediction_inpaint("eps", "repaint_standard")  # ok
        with pytest.raises(ComponentIncompatibilityError):
            validate_prediction_inpaint("eps", "x0_full_reverse_inpaint")

    def test_validate_all_collects_issues(self):
        """validate_all() should return a list of issues, not raise."""
        noise = ForwardDiffDivFreeNoise()
        std = ZScoreStandardizer(0.0, 1.0, 0.0, 1.0)  # incompatible!
        issues = validate_all(
            noise,
            std,
            prediction_target="x0",
            inpaint_fn_name="repaint_standard",  # also incompatible
            projection_fn_name="spectral_project_div_free",  # also incompatible
        )
        assert len(issues) == 3, f"Expected 3 issues, got {len(issues)}: {issues}"


# =========================================================================
# 5. Registry completeness
# =========================================================================


class TestRegistries:
    """Verify that registries are non-empty and well-formed."""

    def test_noise_registry_not_empty(self):
        assert len(NOISE_REGISTRY) >= 3

    def test_loss_registry_not_empty(self):
        assert len(LOSS_REGISTRY) >= 2

    def test_standardizer_registry_not_empty(self):
        assert len(STANDARDIZER_REGISTRY) >= 3

    def test_noise_registry_values_are_classes(self):
        for name, cls in NOISE_REGISTRY.items():
            assert isinstance(cls, type), f"NOISE_REGISTRY['{name}'] is not a class"

    def test_loss_registry_values_are_classes(self):
        for name, cls in LOSS_REGISTRY.items():
            assert isinstance(cls, type), f"LOSS_REGISTRY['{name}'] is not a class"
