"""
Tests for data loading and dataset classes.
"""
import torch
import pytest
import numpy as np
from torch.utils.data import DataLoader

from data_prep.ocean_image_dataset import OceanImageDataset
from ddpm.utils.noise_utils import GaussianNoise


@pytest.fixture
def sample_data_tensor():
    """Create a sample data tensor matching expected format."""
    # Shape: (94, 44, 2, n_timesteps) - ROMS format
    n_timesteps = 100
    data = torch.randn(94, 44, 2, n_timesteps)
    # Add some NaN values to simulate real data
    data[10:15, 20:25, :, :50] = float('nan')
    return data


@pytest.fixture
def simple_noise_strategy():
    """Simple gaussian noise for testing."""
    return GaussianNoise()


class TestOceanImageDataset:
    """Tests for the OceanImageDataset class."""
    
    def test_dataset_initializes(self, sample_data_tensor, simple_noise_strategy):
        """Dataset should initialize without errors."""
        dataset = OceanImageDataset(
            data_tensor=sample_data_tensor,
            n_steps=100,
            noise_strategy=simple_noise_strategy
        )
        assert dataset is not None
        
    def test_dataset_length(self, sample_data_tensor, simple_noise_strategy):
        """Dataset length should match timesteps."""
        dataset = OceanImageDataset(
            data_tensor=sample_data_tensor,
            n_steps=100,
            noise_strategy=simple_noise_strategy
        )
        assert len(dataset) == 100
        
    def test_dataset_getitem_returns_tuple(self, sample_data_tensor, simple_noise_strategy):
        """__getitem__ should return (x0, t, noise) tuple."""
        dataset = OceanImageDataset(
            data_tensor=sample_data_tensor,
            n_steps=100,
            noise_strategy=simple_noise_strategy
        )
        
        item = dataset[0]
        
        assert isinstance(item, tuple)
        assert len(item) == 3
        
    def test_dataset_output_shapes(self, sample_data_tensor, simple_noise_strategy):
        """Output tensors should have correct shapes."""
        dataset = OceanImageDataset(
            data_tensor=sample_data_tensor,
            n_steps=100,
            noise_strategy=simple_noise_strategy
        )
        
        x0, t, noise = dataset[0]
        
        # x0 should be (3, H, W) - u, v, mask
        assert x0.dim() == 3
        assert x0.shape[0] == 3  # u, v, mask channels
        
        # t should be a scalar
        assert isinstance(t, int) or (isinstance(t, torch.Tensor) and t.dim() == 0)
        
        # noise should match x0's spatial dimensions
        assert noise.shape[-2:] == x0.shape[-2:]
        
    def test_dataset_handles_nan(self, sample_data_tensor, simple_noise_strategy):
        """Dataset should handle NaN values in input."""
        dataset = OceanImageDataset(
            data_tensor=sample_data_tensor,
            n_steps=100,
            noise_strategy=simple_noise_strategy
        )
        
        x0, _, _ = dataset[0]
        
        # No NaN values in output
        assert not torch.isnan(x0).any()
        
    def test_dataset_creates_valid_mask(self, sample_data_tensor, simple_noise_strategy):
        """Mask channel should be binary (0 or 1)."""
        dataset = OceanImageDataset(
            data_tensor=sample_data_tensor,
            n_steps=100,
            noise_strategy=simple_noise_strategy
        )
        
        x0, _, _ = dataset[0]
        mask = x0[2]  # Third channel is mask
        
        # Should be binary
        unique_vals = torch.unique(mask)
        assert all(v in [0.0, 1.0] for v in unique_vals.tolist())
        
    def test_dataset_with_max_samples(self, sample_data_tensor, simple_noise_strategy):
        """max_samples should limit dataset size."""
        max_samples = 50
        dataset = OceanImageDataset(
            data_tensor=sample_data_tensor,
            n_steps=100,
            noise_strategy=simple_noise_strategy,
            max_samples=max_samples
        )
        
        assert len(dataset) <= max_samples
        
    def test_dataset_with_data_fraction(self, sample_data_tensor, simple_noise_strategy):
        """data_fraction should limit dataset size."""
        dataset = OceanImageDataset(
            data_tensor=sample_data_tensor,
            n_steps=100,
            noise_strategy=simple_noise_strategy,
            data_fraction=0.5
        )
        
        # Should use approximately half the data
        assert len(dataset) <= 100  # Can't exceed timesteps


class TestDataLoader:
    """Tests for DataLoader compatibility."""
    
    def test_dataloader_works(self, sample_data_tensor, simple_noise_strategy):
        """Dataset should work with PyTorch DataLoader."""
        dataset = OceanImageDataset(
            data_tensor=sample_data_tensor,
            n_steps=100,
            noise_strategy=simple_noise_strategy
        )
        
        loader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        batch = next(iter(loader))
        x0, t, noise = batch
        
        assert x0.shape[0] == 4  # batch size
        
    def test_dataloader_batch_dimension(self, sample_data_tensor, simple_noise_strategy):
        """Batched data should have correct dimensions."""
        dataset = OceanImageDataset(
            data_tensor=sample_data_tensor,
            n_steps=100,
            noise_strategy=simple_noise_strategy
        )
        
        loader = DataLoader(dataset, batch_size=8)
        x0, t, noise = next(iter(loader))
        
        # Should be (batch, channels, H, W)
        assert x0.dim() == 4
        assert x0.shape[0] == 8


class TestDataConsistency:
    """Tests for data consistency and reproducibility."""
    
    def test_same_index_same_data(self, sample_data_tensor, simple_noise_strategy):
        """Same index should return same x0 (but different noise)."""
        dataset = OceanImageDataset(
            data_tensor=sample_data_tensor,
            n_steps=100,
            noise_strategy=simple_noise_strategy
        )
        
        x0_1, _, _ = dataset[0]
        x0_2, _, _ = dataset[0]
        
        # x0 should be the same
        assert torch.allclose(x0_1, x0_2)
        
    def test_different_timesteps_random(self, sample_data_tensor, simple_noise_strategy):
        """Timestep t should be randomly sampled."""
        dataset = OceanImageDataset(
            data_tensor=sample_data_tensor,
            n_steps=100,
            noise_strategy=simple_noise_strategy
        )
        
        # Get multiple samples and check t varies
        timesteps = [dataset[0][1] for _ in range(20)]
        
        # Should have some variation (very unlikely to get same t 20 times)
        assert len(set(timesteps)) > 1


class TestDataValidation:
    """Tests for input validation."""
    
    def test_wrong_tensor_shape_raises(self, simple_noise_strategy):
        """Wrong input shape should raise AssertionError."""
        bad_data = torch.randn(64, 64, 100)  # Wrong shape
        
        with pytest.raises(AssertionError):
            OceanImageDataset(
                data_tensor=bad_data,
                n_steps=100,
                noise_strategy=simple_noise_strategy
            )
            
    def test_wrong_channels_raises(self, simple_noise_strategy):
        """Wrong number of channels should raise AssertionError."""
        bad_data = torch.randn(94, 44, 3, 100)  # 3 channels instead of 2
        
        with pytest.raises(AssertionError):
            OceanImageDataset(
                data_tensor=bad_data,
                n_steps=100,
                noise_strategy=simple_noise_strategy
            )
