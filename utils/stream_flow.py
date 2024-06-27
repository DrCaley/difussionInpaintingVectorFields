import torch
from torch.utils.data import DataLoader
import numpy as np

from dataloaders.dataloader import OceanImageDataset
from utils.tensors_to_png import generate_png

#input should be an any by 2 or 3 by any by any Tensor
def calculate_flow(tensor):
    diff_tensor = torch.zeros(tensor.shape)
    if len(tensor.shape) > 3:

        for i in range(tensor.shape[0]):
            diff_tensor[i] = calculate_flow(tensor[i])[0]
        return diff_tensor
    #left strategy
    total = 0
    for y in range(tensor.shape[1] - 1):
        for x in range(tensor.shape[2] - 1):
            u_right = tensor[0][y][x+1]
            u_left = tensor[0][y][x]
            v_up = tensor[1][y][x]
            v_down = tensor[1][y+1][x]

            u_right = 0 if torch.isnan(u_right) else u_right
            u_left = 0 if torch.isnan(u_left) else u_left
            v_up = 0 if torch.isnan(v_up) else v_up
            v_down = 0 if torch.isnan(v_down) else v_down

            net_both = u_left - u_right + v_up - v_down
            total += abs(net_both)
            diff_tensor[0][y][x] = net_both
    avg = total / (92 * 42) #.0069,.0059
    for channel in range(tensor.shape[0]):
        diff_tensor[channel] = diff_tensor[0].clone().detach()

    #testing
    diff_arr = diff_tensor.detach().numpy() #between -0.3195 and 0.2726
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
