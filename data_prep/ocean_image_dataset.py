import torch
from torch import Tensor
from torch.utils.data import Dataset
import yaml
from typing import Optional, Union


class OceanImageDataset(Dataset):
    """
    Loads u and v velocity components from a provided tensor (not a .mat file), applies masking to handle NaNs,
    and returns 3-channel tensors representing u, v, and a binary mask.
    """

    def __init__(
        self, data_tensor: Tensor,
        n_steps,
        noise_strategy=None,
        transform=None,
        boundaries: Optional[str] = None,
        data_fraction: Optional[float] = None,
        max_samples: Optional[int] = None,
        max_size = None
    ):
        """
        Initializes the dataset.

        Args:
            data_tensor (Tensor): Tensor with shape (94, 44, 2, n)
            transform (callable, optional): Optional transform to apply to each tensor.
            boundaries (str, optional): Path to YAML file with boundary info.
            data_fraction (float, optional): Fraction of the dataset to use (between 0 and 1).
            max_samples (int, optional): Maximum number of samples to use.
        """
        self.n_steps = n_steps
        self.noise_strategy = noise_strategy
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
            if max_size > total_timesteps:
                print(f"Warning: Requested max_size {max_size} exceeds available timesteps {total_timesteps}. Using all data.")
                max_size = total_timesteps
        else:
            max_size = total_timesteps

        self.raw_tensor = data_tensor[..., :used_timesteps]  # restrict to selected portion
        self.tensor_labels = list(range(max_size))
        self.transform = transform

        # Load boundaries if provided
        self.boundaries = None
        if boundaries:
            with open(boundaries, 'r') as file:
                self.boundaries = yaml.safe_load(file)
            print(f"Loaded {boundaries}")

        # Preprocess all time steps
        self.tensor_arr = [self.load_array(n) for n in self.tensor_labels]
        print(f"Loaded {len(self.tensor_arr)} time steps.")

    def __len__(self) -> int:
        return len(self.tensor_arr)

    def __getitem__(self, idx):
        x0 = self.tensor_arr[idx]
        if self.transform:
            x0 = self.transform(x0)
        t = torch.randint(0, self.n_steps, (1,)).item()

        noise = self.noise_strategy(x0.unsqueeze(0), torch.tensor([t])).squeeze(0)
        return x0, t, noise

    def load_array(self, n: int) -> Tensor:
        """
        Process and return the (u, v, mask) tensor for time index n.
        """
        # Extract single time step
        u = self.raw_tensor[..., n][..., 0].T  # shape: (44, 94)
        v = self.raw_tensor[..., n][..., 1].T  # shape: (44, 94)

        # Handle NaNs and build mask
        mask = (~(u.isnan() | v.isnan())).float()
        u = torch.nan_to_num(u, nan=0.0)
        v = torch.nan_to_num(v, nan=0.0)

        return torch.stack((u, v, mask), dim=0)  # shape: (3, 44, 94)
