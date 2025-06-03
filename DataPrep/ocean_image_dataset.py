import torch
import yaml
from scipy.io import loadmat
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data.dataset import _T_co

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

    def __init__(self, num : int, transform=None, mat_file="../data/rams_head/stjohn_hourly_5m_velocity_ramhead_v2.mat", boundaries="../data/rams_head/boundaries.yaml"):
        """
        Initializes the coean_image_dataset
        Args:
            num (int) : Number of time steps to load
            transform (callable, optional): Optional transform to apply to each tensor.
            mat_file: the MATLAB file for which we are getting the 'u', 'v', and 'ocean_time' arrays
            boundaries : path to YAML file that defines our boundaries.
        """
        self.mat_data = loadmat(mat_file)
        self.tensor_labels = []
        self.tensor_arr = []
        self.transform = transform

        with open(boundaries, 'r') as file:
            self.boundaries = yaml.safe_load(file)
        print(f"Loaded {boundaries}")

        for n in range(num):
            self.tensor_arr.append(self.load_array(n))
            self.tensor_labels.append(n)
        print(f"Loaded tensor_arr")

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

    def load_array(self, tensor_num: object) -> Tensor:
        """
            Loads and processes a single timestep from the MATLAB file.

            - Extracts the u and v velocity fields.
            - Reorders dimensions to match PyTorch format.
            - Replaces NaNs with 0s.
            - Builds a binary mask for valid data.
            - Stacks u, v, and mask into a 3-channel tensor.

            Args:
                tensor_num (int): The index of the timestep to load.

            Returns:
                torch.Tensor: A 3xHxW tensor of u, v, and mask components.
        """
        u_tensors = torch.from_numpy(self.mat_data['u'])
        u_tensors = u_tensors.permute(*torch.arange(u_tensors.ndim - 1, -1, -1))[tensor_num]
        v_tensors = torch.from_numpy(self.mat_data['v'])
        v_tensors = v_tensors.permute(*torch.arange(v_tensors.ndim - 1, -1, -1))[tensor_num]
        time = torch.from_numpy(self.mat_data['ocean_time'].squeeze())

        mask = u_tensors.clone().detach()
        for x in range(94):
            for y in range(44):
                if u_tensors[y][x].isnan() or v_tensors[y][x].isnan():
                    mask[y][x] = 0
                    u_tensors[y][x] = 0
                    v_tensors[y][x] = 0
                else:
                    mask[y][x] = 1

        combined_tensor = torch.stack((u_tensors, v_tensors, mask))
        return combined_tensor
