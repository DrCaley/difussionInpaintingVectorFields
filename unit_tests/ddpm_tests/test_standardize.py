import torch
import pytest
from ddpm.helper_functions.standardize_data import (
    ZScoreStandardizer,
    MaxMagnitudeStandardizer,
    UnitVectorNormalizer,
    STANDARDIZER_REGISTRY
)

# Example test tensor: shape [2, 4, 4] (u and v channels)
@pytest.fixture
def test_tensor():
    torch.manual_seed(0)
    return torch.randn(2, 4, 4)

def test_zscore_standardizer_roundtrip(test_tensor):
    u_mean = test_tensor[0].mean()
    u_std = test_tensor[0].std()
    v_mean = test_tensor[1].mean()
    v_std = test_tensor[1].std()

    standardizer = ZScoreStandardizer(u_mean, u_std, v_mean, v_std)
    standardized = standardizer(test_tensor)
    unstandardized = standardizer.unstandardize(standardized)

    assert torch.allclose(unstandardized, test_tensor, atol=1e-5)

def test_maxmag_standardizer_roundtrip(test_tensor):
    standardizer = MaxMagnitudeStandardizer()
    standardized = standardizer(test_tensor.permute(1, 2, 0).unsqueeze(0))  # Shape [1, H, W, 2] → [1, H, W, 2]
    unstandardized = standardizer.unstandardize(standardized)

    assert torch.allclose(unstandardized, test_tensor.permute(1, 2, 0).unsqueeze(0), atol=1e-5)

def test_maxmag_unstandardize_without_call_raises():
    standardizer = MaxMagnitudeStandardizer()
    with pytest.raises(ValueError):
        standardizer.unstandardize(torch.zeros(1, 4, 4, 2))

def test_unitvector_normalizer_roundtrip(test_tensor):
    standardizer = UnitVectorNormalizer()
    standardized = standardizer(test_tensor)
    unstandardized = standardizer.unstandardize(standardized)

    assert torch.allclose(unstandardized, test_tensor, atol=1e-4)

def test_unitvector_unstandardize_without_call_raises():
    standardizer = UnitVectorNormalizer()
    with pytest.raises(ValueError):
        standardizer.unstandardize(torch.zeros(2, 4, 4))

def test_registry_contains_all_keys():
    assert "zscore" in STANDARDIZER_REGISTRY
    assert "maxmag" in STANDARDIZER_REGISTRY
    assert "units" in STANDARDIZER_REGISTRY
    assert issubclass(STANDARDIZER_REGISTRY["zscore"], ZScoreStandardizer.__bases__[0])  # Abstract base
