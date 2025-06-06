import torch
import yaml
from scipy.io import loadmat
from torch import Tensor
from torch.utils.data import Dataset


class OceanImageDataset(Dataset):
    """
    loads 'u' and 'v' velocity components from .mat file, applies masking to handle NaNs and returns 3-channel tensors representing u, v, and a binary mask

    Attributes:
        mat_data (dict): Dictionary loaded from the .mat file.
        tensor_arr (List[Tensor]): List of preprocessed tensors for each time step.
        tensor_labels (List[int]): Index labels corresponding to each tensor.
        boundaries (dict): Optional spatial boundary data loaded from YAML.
        transform (callable, optional): Optional transform to apply to each tensor.
    """

    def __init__(self, data_tensor: Tensor, transform=None, boundaries="../data/rams_head/boundaries.yaml", max_size = None):
        """
        Initializes the dataset.

        Args:
            data_tensor (Tensor): Tensor with shape (94, 44, 2, n)
            transform (callable, optional): Optional transform to apply to each tensor.
            boundaries (str, optional): Path to YAML file with boundary info.
            max_size (int, optional): Max number of time steps to load.
        """
        assert data_tensor.ndim == 4 and data_tensor.shape[2] == 2, "Expected shape (94, 44, 2, n)"
        self.raw_tensor = data_tensor  # shape: (94, 44, 2, n)
        total_timesteps = data_tensor.shape[3]

        if max_size is not None:
            if max_size > total_timesteps:
                print(f"Warning: Requested max_size {max_size} exceeds available timesteps {total_timesteps}. Using all data.")
                max_size = total_timesteps
        else:
            max_size = total_timesteps

        self.tensor_labels = list(range(max_size))
        self.transform = transform

        # Load boundaries if provided
        self.boundaries = None
        if boundaries:
            with open(boundaries, 'r') as file:
                self.boundaries = yaml.safe_load(file)
            print(f"Loaded {boundaries}")

        # Preprocess only selected time steps
        self.tensor_arr = [self.load_array(n) for n in self.tensor_labels]
        print(f"Loaded {len(self.tensor_arr)} of {total_timesteps} available time steps.")

    def __len__(self) -> int:
        """
        Returns the number of tensors in the dataset.

        Returns:
            int: Number of time steps loaded.
        """
        return len(self.tensor_arr)

    def __getitem__(self, idx):
        """
        Retrieves a single tensor from the dataset at index. used for for-loop iterator and similar to
        next() from Iterable in Java.

        Args:
            idx (int): Index of the desired tensor.

        Returns:
            Tuple[Tensor, int]: The processed tensor and its corresponding index.
        """
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
