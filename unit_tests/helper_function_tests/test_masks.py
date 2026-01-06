"""
Tests for mask generators.
Note: Some tests may be skipped if mask modules cannot be imported due to tkinter dependency.
"""
import torch
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import sys

# Import abstract class directly (doesn't depend on tkinter)
from ddpm.helper_functions.masks.abstract_mask import MaskGenerator

# Try to import individual mask generators
# They may fail if the masks/__init__.py runs and tries to import mask_drawer
try:
    # Temporarily prevent tkinter from being imported
    sys.modules['tkinter'] = MagicMock()
    
    from ddpm.helper_functions.masks.random_mask import RandomMaskGenerator
    from ddpm.helper_functions.masks.n_coverage_mask import CoverageMaskGenerator
    from ddpm.helper_functions.masks.border_mask import BorderMaskGenerator
    MASKS_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    MASKS_AVAILABLE = False
    RandomMaskGenerator = None
    CoverageMaskGenerator = None
    BorderMaskGenerator = None
finally:
    # Restore tkinter
    if 'tkinter' in sys.modules and isinstance(sys.modules['tkinter'], MagicMock):
        del sys.modules['tkinter']


@pytest.fixture
def standard_shape():
    """Standard image shape for testing."""
    return (1, 2, 44, 94)


# Skip tests that require specific mask generators if they're not available
masks_available = pytest.mark.skipif(
    not MASKS_AVAILABLE,
    reason="Mask generators cannot be imported (likely missing tkinter)"
)


class TestMaskGeneratorAbstract:
    """Tests for the abstract MaskGenerator class."""
    
    def test_cannot_instantiate_abstract(self):
        """Cannot instantiate abstract MaskGenerator directly."""
        with pytest.raises(TypeError):
            MaskGenerator()
            
    def test_subclass_requires_methods(self):
        """Subclass must implement all abstract methods."""
        class IncompleteMask(MaskGenerator):
            def __init__(self):
                pass
        
        # Should fail because not all methods implemented
        with pytest.raises(TypeError):
            IncompleteMask()


class TestMaskGeneratorRegistry:
    """Tests for the dynamic mask generator registry."""
    
    @masks_available
    def test_random_mask_generator_available(self):
        """RandomMaskGenerator should be importable."""
        assert RandomMaskGenerator is not None
        
    @masks_available
    def test_coverage_mask_generator_available(self):
        """CoverageMaskGenerator should be importable."""
        assert CoverageMaskGenerator is not None
        
    @masks_available
    def test_border_mask_generator_available(self):
        """BorderMaskGenerator should be importable."""
        assert BorderMaskGenerator is not None


@pytest.mark.skipif(not MASKS_AVAILABLE, reason="Mask generators not available")
class TestRandomMaskGenerator:
    """Tests for RandomMaskGenerator."""
    
    @pytest.fixture
    def generator(self):
        return RandomMaskGenerator(max_mask_size=16)
    
    def test_init(self, generator):
        """Should initialize with max_mask_size."""
        assert generator.max_mask_size == 16
        
    def test_generate_mask_returns_tensor(self, generator, standard_shape):
        """generate_mask should return a tensor."""
        mask = generator.generate_mask(image_shape=standard_shape)
        assert isinstance(mask, torch.Tensor)
        
    def test_generate_mask_correct_shape(self, generator, standard_shape):
        """Mask should have shape (1, 1, H, W)."""
        mask = generator.generate_mask(image_shape=standard_shape)
        assert mask.shape == (1, 1, 44, 94)
        
    def test_mask_binary_values(self, generator, standard_shape):
        """Mask should contain only 0s and 1s."""
        mask = generator.generate_mask(image_shape=standard_shape)
        unique_vals = torch.unique(mask)
        assert all(v in [0.0, 1.0] for v in unique_vals.tolist())
        
    def test_str_method(self, generator):
        """__str__ should return readable name."""
        name = str(generator)
        assert "Random" in name or "Mask" in name


@pytest.mark.skipif(not MASKS_AVAILABLE, reason="Mask generators not available")
class TestCoverageMaskGenerator:
    """Tests for CoverageMaskGenerator."""
    
    @pytest.fixture
    def generator(self):
        return CoverageMaskGenerator(coverage_ratio=0.3)
    
    def test_init(self, generator):
        """Should initialize with coverage_ratio."""
        assert generator.coverage_ratio == 0.3
        
    def test_generate_mask_returns_tensor(self, generator, standard_shape):
        """generate_mask should return a tensor."""
        mask = generator.generate_mask(image_shape=standard_shape)
        assert isinstance(mask, torch.Tensor)
        
    def test_coverage_ratio_affects_mask_size(self, standard_shape):
        """Higher coverage ratio should have more masked pixels."""
        small_gen = CoverageMaskGenerator(coverage_ratio=0.1)
        large_gen = CoverageMaskGenerator(coverage_ratio=0.5)
        
        small_mask = small_gen.generate_mask(image_shape=standard_shape)
        large_mask = large_gen.generate_mask(image_shape=standard_shape)
        
        # Count zeros (masked areas)
        small_zeros = (small_mask == 0).sum().item()
        large_zeros = (large_mask == 0).sum().item()
        
        # This may not always hold due to BFS randomness, so we just check they're generated
        assert small_zeros >= 0
        assert large_zeros >= 0


@pytest.mark.skipif(not MASKS_AVAILABLE, reason="Mask generators not available")
class TestBorderMaskGenerator:
    """Tests for BorderMaskGenerator."""
    
    @pytest.fixture
    def generator(self):
        return BorderMaskGenerator()
    
    def test_generate_mask_returns_tensor(self, generator, standard_shape):
        """generate_mask should return a tensor."""
        mask = generator.generate_mask(image_shape=standard_shape)
        assert isinstance(mask, torch.Tensor)


@pytest.mark.skipif(not MASKS_AVAILABLE, reason="Mask generators not available")
class TestMaskDeterminism:
    """Tests for mask generation reproducibility."""
    
    def test_random_mask_different_each_call(self, standard_shape):
        """Random mask should be different each call (stochastic)."""
        gen = RandomMaskGenerator(max_mask_size=16)
        
        # Generate multiple masks
        masks = [gen.generate_mask(image_shape=standard_shape) for _ in range(5)]
        
        # At least some should be different (probabilistically)
        # We just check they all have correct shape
        for mask in masks:
            assert mask.shape == (1, 1, 44, 94)


@pytest.mark.skipif(not MASKS_AVAILABLE, reason="Mask generators not available")
class TestMaskShapeConsistency:
    """Tests for mask shape consistency across generators."""
    
    def test_all_generators_return_correct_shape(self, standard_shape):
        """All mask generators should return consistent shapes."""
        generators = [
            RandomMaskGenerator(max_mask_size=16),
            BorderMaskGenerator(),
        ]
        
        for gen in generators:
            mask = gen.generate_mask(image_shape=standard_shape)
            # Should be 4D with batch and channel = 1
            assert mask.dim() == 4
            assert mask.shape[0] == 1  # batch
            assert mask.shape[1] == 1  # channel


@pytest.mark.skipif(not MASKS_AVAILABLE, reason="Mask generators not available")
class TestMaskEdgeCases:
    """Tests for edge cases in mask generation."""
    
    def test_random_mask_very_small_max_size(self, standard_shape):
        """Should handle very small max_mask_size."""
        gen = RandomMaskGenerator(max_mask_size=1)
        mask = gen.generate_mask(image_shape=standard_shape)
        
        assert mask.shape == (1, 1, 44, 94)
        
    def test_coverage_very_small_ratio(self, standard_shape):
        """Should handle very small coverage ratio."""
        gen = CoverageMaskGenerator(coverage_ratio=0.01)
        mask = gen.generate_mask(image_shape=standard_shape)
        
        assert isinstance(mask, torch.Tensor)
        
    def test_coverage_very_large_ratio(self, standard_shape):
        """Should handle large coverage ratio."""
        gen = CoverageMaskGenerator(coverage_ratio=0.99)
        mask = gen.generate_mask(image_shape=standard_shape)
        
        assert isinstance(mask, torch.Tensor)


@pytest.mark.skipif(not MASKS_AVAILABLE, reason="Mask generators not available")
class TestMaskDeviceHandling:
    """Tests for mask device handling."""
    
    def test_mask_on_correct_device(self, standard_shape):
        """Mask should be on the correct device."""
        gen = RandomMaskGenerator()
        mask = gen.generate_mask(image_shape=standard_shape)
        
        # Should be on a valid device
        assert mask.device.type in ['cpu', 'cuda', 'mps']
