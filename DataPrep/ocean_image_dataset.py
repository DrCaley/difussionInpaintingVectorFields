import torch
import yaml
from scipy.io import loadmat
from torch.utils.data import Dataset

"""The main dataloader for all data"""
class ocean_image_dataset(Dataset):
    def __init__(self, num, transform=None, mat_file="../data/rams_head/stjohn_hourly_5m_velocity_ramhead_v2.mat", boundaries="../data/rams_head/boundaries.yaml"):
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

    def __len__(self):
        return len(self.tensor_arr)

    def __getitem__(self, idx):
        tensor = self.tensor_arr[idx]
        if self.transform:
            tensor = self.transform(tensor)
        return tensor, idx

    def load_array(self, tensor_num):
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
