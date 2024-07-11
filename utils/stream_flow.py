import torch
from torch.utils.data import DataLoader
import numpy as np

from dataloaders.dataloader import OceanImageDataset
from utils.tensors_to_png import generate_png

#input should be an any by 2 or 3 by any by any Tensor
#takes really long, not sure why

def calculate_flow(tensor):
    if len(tensor.shape) > 3:
        return torch.stack([calculate_flow(tensor[i]) for i in range(tensor.shape[0])])

    u = tensor[0]
    v = tensor[1]

    # Replace NaNs with 0
    u = torch.nan_to_num(u, nan=0.0)
    v = torch.nan_to_num(v, nan=0.0)

    # Calculate differences
    u_diff = u[:, :-1] - u[:, 1:]
    v_diff = v[:-1, :] - v[1:, :]

    # Pad the differences to match the original tensor shape
    u_diff = torch.nn.functional.pad(u_diff, (0, 1, 0, 0))
    v_diff = torch.nn.functional.pad(v_diff, (0, 0, 0, 1))

    # Sum the differences
    diff_tensor = u_diff + v_diff

    # Expand to match the input tensor shape (for consistency)
    diff_tensor = diff_tensor.unsqueeze(0)

    return diff_tensor

def test_flow():
    data = OceanImageDataset(
        mat_file=".././data/rams_head/stjohn_hourly_5m_velocity_ramhead_v2.mat",
        boundaries=".././data/rams_head/boundaries.yaml",
        num=5
    )
    test_loader = DataLoader(data)
    for i, (tensors, label) in enumerate(test_loader):
        if i == 1:
            for tensor in tensors:
                calculate_flow(tensor)

#test_flow()
