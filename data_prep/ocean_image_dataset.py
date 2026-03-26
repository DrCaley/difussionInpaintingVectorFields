import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
import yaml
from typing import Optional, Union


class OceanImageDataset(Dataset):
    """
    Loads u and v velocity components from a provided tensor (not a .mat file), applies masking to handle NaNs,
    and returns 3-channel tensors representing u, v, and a binary mask.

    data_tensor can be a torch Tensor or a numpy array (e.g. memory-mapped).
    """

    def __init__(
        self, data_tensor,
        n_steps,
        noise_strategy=None,
        transform=None,
        boundaries: Optional[str] = None,
        data_fraction: Optional[float] = None,
        max_samples: Optional[int] = None,
        max_size = None,
        ocean_masks=None,
        bathymetry=None,
        bathy_stats: Optional[tuple] = None,
    ):
        """
        Initializes the dataset.

        Args:
            data_tensor: Tensor or numpy array with shape (94, 44, 2, n)
            transform (callable, optional): Optional transform to apply to each tensor.
            boundaries (str, optional): Path to YAML file with boundary info.
            data_fraction (float, optional): Fraction of the dataset to use (between 0 and 1).
            max_samples (int, optional): Maximum number of samples to use.
            ocean_masks: Per-sample ocean masks, shape (94, 44, n). 1=ocean, 0=land.
            bathymetry: Per-sample bathymetry, shape (94, 44, n). Raw metres.
            bathy_stats (tuple, optional): (bathy_min, bathy_max) for min-max normalization.
        """
        self.n_steps = n_steps
        self.noise_strategy = noise_strategy
        self._is_numpy = isinstance(data_tensor, np.ndarray)
        assert data_tensor.ndim == 4 and data_tensor.shape[2] == 2, "Expected shape (94, 44, 2, n)"
        total_timesteps = data_tensor.shape[3]

        # Determine how many samples to use
        if data_fraction is not None:
            assert 0 < data_fraction <= 1, "data_fraction must be in (0, 1]"
            used_timesteps = int(total_timesteps * data_fraction)
        elif max_samples is not None:
            used_timesteps = min(max_samples, total_timesteps)
        else:
            used_timesteps = total_timesteps

        if max_size is not None:
            if max_size > used_timesteps:
                print(f"Warning: Requested max_size {max_size} exceeds available timesteps {used_timesteps}. Using all data.")
                max_size = used_timesteps
        else:
            max_size = used_timesteps

        self.raw_tensor = data_tensor[..., :used_timesteps]  # restrict to selected portion
        self.tensor_labels = list(range(max_size))
        self.transform = transform
        self.ocean_masks = ocean_masks[..., :used_timesteps] if ocean_masks is not None else None
        self.bathymetry = bathymetry[..., :used_timesteps] if bathymetry is not None else None
        self.bathy_stats = bathy_stats  # (bathy_min, bathy_max) or None

        # Load boundaries if provided
        self.boundaries = None
        if boundaries:
            with open(boundaries, 'r') as file:
                self.boundaries = yaml.safe_load(file)
            print(f"Loaded {boundaries}")

        # For small torch datasets, precompute all samples for speed.
        # For large datasets or mmap arrays, use lazy loading to avoid OOM.
        self._lazy = self._is_numpy or max_size > 50_000
        if self._lazy:
            self.tensor_arr = None
            print(f"Lazy-loading mode: {max_size} samples (not precomputed).")
        else:
            self.tensor_arr = [self.load_array(n) for n in self.tensor_labels]
            print(f"Loaded {len(self.tensor_arr)} time steps.")

    def __len__(self) -> int:
        return len(self.tensor_labels)

    def __getitem__(self, idx):
        if self._lazy:
            x0 = self.load_array(self.tensor_labels[idx])
        else:
            x0 = self.tensor_arr[idx]
        if self.transform:
            x0 = self.transform(x0)
        t = torch.randint(0, self.n_steps, (1,)).item()

        noise = self.noise_strategy(x0.unsqueeze(0), torch.tensor([t])).squeeze(0)
        return x0, t, noise

    def load_array(self, n: int) -> Tensor:
        """
        Process and return the (u, v, mask) tensor for time index n.
        Handles both torch Tensors and numpy (mmap) arrays.
        """
        if self._is_numpy:
            # numpy mmap path — read slice, convert to torch
            sample = self.raw_tensor[..., n]  # (94, 44, 2)
            u = torch.from_numpy(sample[..., 0].T.copy()).float()  # (44, 94)
            v = torch.from_numpy(sample[..., 1].T.copy()).float()  # (44, 94)
            if self.ocean_masks is not None:
                mask = torch.from_numpy(self.ocean_masks[..., n].T.copy()).float()
            else:
                mask = (~(torch.isnan(u) | torch.isnan(v))).float()
            u = torch.nan_to_num(u, nan=0.0)
            v = torch.nan_to_num(v, nan=0.0)
            return torch.stack((u, v, mask), dim=0)

        # Torch tensor path (original)
        u = self.raw_tensor[..., n][..., 0].T  # shape: (44, 94)
        v = self.raw_tensor[..., n][..., 1].T  # shape: (44, 94)

        # Handle NaNs and build mask
        if self.ocean_masks is not None:
            mask = self.ocean_masks[..., n].T.float()  # (44, 94), 1=ocean
        else:
            mask = (~(u.isnan() | v.isnan())).float()
        u = torch.nan_to_num(u, nan=0.0)
        v = torch.nan_to_num(v, nan=0.0)

        return torch.stack((u, v, mask), dim=0)  # shape: (3, 44, 94)

    def load_bathymetry(self, n: int) -> Optional[Tensor]:
        """Return normalized bathymetry for sample n, or None if unavailable."""
        if self.bathymetry is None:
            return None
        if self._is_numpy:
            bathy = torch.from_numpy(self.bathymetry[..., n].T.copy()).float()
        else:
            bathy = self.bathymetry[..., n].T.float()  # (44, 94)
        if self.bathy_stats is not None:
            bmin, bmax = self.bathy_stats
            if bmax > bmin:
                bathy = (bathy - bmin) / (bmax - bmin)
        return bathy  # (44, 94), normalized to ~[0, 1]
