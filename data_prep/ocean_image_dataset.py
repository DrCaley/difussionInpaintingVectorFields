import torch
from torch import Tensor
from torch.utils.data import Dataset
import yaml
from typing import Optional


class OceanImageTensorDataset(Dataset):
    """
    Loads u and v velocity components from a provided tensor (not a .mat file), applies masking to handle NaNs,
    and returns 3-channel tensors representing u, v, and a binary mask.
    """

    def __init__(self, data_tensor: Tensor, transform=None, boundaries: Optional[str] = None):
        """
        Initializes the dataset.

        Args:
            data_tensor (Tensor): Tensor with shape (94, 44, 2, n)
            transform (callable, optional): Optional transform to apply to each tensor.
            boundaries (str, optional): Path to YAML file with boundary info.
        """
        assert data_tensor.ndim == 4 and data_tensor.shape[2] == 2, "Expected shape (94, 44, 2, n)"
        self.raw_tensor = data_tensor  # shape: (94, 44, 2, n)
        self.tensor_labels = list(range(data_tensor.shape[3]))
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
        tensor = self.tensor_arr[idx]
        if self.transform:
            tensor = self.transform(tensor)
        return tensor, idx

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
