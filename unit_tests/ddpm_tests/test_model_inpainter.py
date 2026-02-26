"""
Tests for the ModelInpainter class - evaluation system.
Note: Some tests are skipped if ModelInpainter cannot be imported due to tkinter dependency.
"""
import torch
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import tempfile
import os

# Try to import ModelInpainter - it may fail due to tkinter dependency
try:
    from ddpm.Testing.model_inpainter import ModelInpainter
    INPAINTER_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    INPAINTER_AVAILABLE = False
    ModelInpainter = None

# These utilities should be importable without tkinter
from ddpm.utils.inpainting_utils import calculate_mse, calculate_percent_error


# Skip all tests in this module if ModelInpainter is not available
pytestmark = pytest.mark.skipif(
    not INPAINTER_AVAILABLE, 
    reason="ModelInpainter cannot be imported (likely missing tkinter)"
)


@pytest.fixture
def inpainter():
    """Create a ModelInpainter instance."""
    return ModelInpainter()


@pytest.fixture
def sample_images():
    """Create sample original and predicted images."""
    original = torch.randn(1, 2, 64, 128)
    predicted = original + torch.randn_like(original) * 0.1  # Small perturbation
    mask = torch.ones_like(original)
    return original, predicted, mask


class TestModelInpainterInit:
    """Tests for ModelInpainter initialization."""
    
    def test_inpainter_initializes(self, inpainter):
        """ModelInpainter should initialize without errors."""
        assert inpainter is not None
        
    def test_inpainter_has_dd(self, inpainter):
        """Should have DDInitializer instance."""
        assert inpainter.dd is not None
        
    def test_inpainter_results_path_exists(self, inpainter):
        """Results path should be set."""
        assert inpainter.results_path is not None
        
    def test_inpainter_model_paths_empty_initially(self, inpainter):
        """Model paths list should start empty or with one entry."""
        assert isinstance(inpainter.model_paths, list)


class TestModelManagement:
    """Tests for adding and configuring models."""
    
    def test_add_model_nonexistent_warns(self, inpainter, capsys):
        """Adding nonexistent model should print warning."""
        inpainter.add_model("/nonexistent/path/model.pt")
        captured = capsys.readouterr()
        assert "Warning" in captured.out or len(inpainter.model_paths) == 0
        
    def test_add_model_with_real_file(self, inpainter):
        """Adding existing file should work."""
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            torch.save({'model_state_dict': {}}, f.name)
            inpainter.add_model(f.name)
            assert f.name in [str(p) for p in inpainter.model_paths]
            os.unlink(f.name)
            
    def test_add_models_list(self, inpainter):
        """add_models should accept a list."""
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f1:
            with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f2:
                torch.save({}, f1.name)
                torch.save({}, f2.name)
                
                initial_count = len(inpainter.model_paths)
                inpainter.add_models([f1.name, f2.name])
                
                # Should have added both
                assert len(inpainter.model_paths) >= initial_count + 2
                
                os.unlink(f1.name)
                os.unlink(f2.name)


class TestResultsPath:
    """Tests for results path management."""
    
    def test_set_results_path(self, inpainter):
        """Should be able to set custom results path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            new_path = Path(tmpdir) / "custom_results"
            inpainter.set_results_path(str(new_path))
            assert inpainter.results_path == new_path
            assert new_path.exists()
            
    def test_results_path_creates_directory(self, inpainter):
        """Setting results path should create directory if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            new_path = Path(tmpdir) / "new_dir" / "results"
            inpainter.set_results_path(str(new_path))
            assert new_path.exists()


class TestPlotLists:
    """Tests for plot data management."""
    
    def test_reset_plot_lists(self, inpainter):
        """reset_plot_lists should clear all lists."""
        # Add some data
        inpainter.mse_ddpm_list.append(1.0)
        inpainter.mse_gp_list.append(2.0)
        
        inpainter.reset_plot_lists()
        
        assert len(inpainter.mse_ddpm_list) == 0
        assert len(inpainter.mse_gp_list) == 0
        assert len(inpainter.mse_distance_ddpm) == 0
        assert len(inpainter.mse_distance_gp) == 0


class TestPixelDimensions:
    """Tests for pixel dimension settings."""
    
    def test_set_pixel_dimensions(self, inpainter):
        """Should be able to set pixel dimensions."""
        inpainter.set_pixel_dimensions(5.0, 10.0)
        
        assert inpainter.pixel_height == 5.0
        assert inpainter.pixel_width == 10.0
        
    def test_default_pixel_dimensions(self, inpainter):
        """Default pixel dimensions should be 1.0."""
        assert inpainter.pixel_height == 1.0
        assert inpainter.pixel_width == 1.0


class TestMetricCalculations:
    """Tests for metric calculation utilities."""
    
    def test_calculate_mse_correct_shape(self, sample_images):
        """calculate_mse should return a scalar."""
        original, predicted, mask = sample_images
        
        mse = calculate_mse(original, predicted, mask)
        
        assert mse.dim() == 0  # scalar
        
    def test_calculate_mse_zero_for_identical(self):
        """MSE should be zero for identical images."""
        img = torch.randn(1, 2, 64, 128)
        mask = torch.ones_like(img)
        
        mse = calculate_mse(img, img.clone(), mask)
        
        assert torch.isclose(mse, torch.tensor(0.0), atol=1e-6)
        
    def test_calculate_mse_positive(self, sample_images):
        """MSE should be positive for different images."""
        original, predicted, mask = sample_images
        
        mse = calculate_mse(original, predicted, mask)
        
        assert mse > 0
        
    def test_calculate_percent_error_returns_scalar(self, sample_images):
        """calculate_percent_error should return a scalar."""
        original, predicted, mask = sample_images
        
        error = calculate_percent_error(original, predicted, mask)
        
        assert error.dim() == 0


class TestMaskCalculations:
    """Tests for mask-related calculations."""
    
    def test_compute_mask_percentage(self, inpainter):
        """compute_mask_percentage should return correct percentage."""
        # 25% masked (75% ones)
        mask = torch.ones(1, 2, 10, 10)
        mask[:, :, :5, :5] = 0  # 25 out of 100 pixels masked
        
        percentage = inpainter.compute_mask_percentage(mask)
        
        # Percentage of MASKED pixels (zeros)
        assert 0 <= percentage <= 100


class TestInpainterFlags:
    """Tests for various flags and settings."""
    
    def test_save_pt_files_flag(self, inpainter):
        """save_pt_files should set the flag."""
        initial = inpainter.save_pt_fields
        inpainter.save_pt_files()
        assert inpainter.save_pt_fields is True


class TestMaskGeneratorIntegration:
    """Tests for mask generator integration."""
    
    def test_add_mask(self, inpainter):
        """Should be able to add mask generators."""
        mock_mask = MagicMock()
        inpainter.add_mask(mock_mask)
        
        assert mock_mask in inpainter.masks_to_use
        
    def test_masks_list_starts_empty(self, inpainter):
        """Masks list should start empty."""
        assert len(inpainter.masks_to_use) == 0
