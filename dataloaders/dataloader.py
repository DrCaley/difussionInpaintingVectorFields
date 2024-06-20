import torch
import yaml
from scipy.io import loadmat
from torch.utils.data import Dataset


class OceanImageDataset(Dataset):
    def __init__(self, mat_file, boundaries, num):
        self.mat_data = loadmat(mat_file)
        self.tensor_labels = []
        self.tensor_arr = []

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
        #label = self.tensor_labels[idx]
        return tensor, 0

    def load_array(self, tensor_num):
        #load tensors
        u_tensors = torch.from_numpy(self.mat_data['u'])
        u_tensors = u_tensors.permute(*torch.arange(u_tensors.ndim - 1, -1, -1))[tensor_num]
        v_tensors = torch.from_numpy(self.mat_data['v'])
        v_tensors = v_tensors.permute(*torch.arange(v_tensors.ndim - 1, -1, -1))[tensor_num]
        time = torch.from_numpy(self.mat_data['ocean_time'].squeeze())

        # Normalize if wanted
        #minU, maxU = -0.8973235906436031, 1.0859991093945718
        #minV, maxV = -0.6647028130174489, 0.5259408400292674
        #u_tensors = (u_tensors - minU) / (maxU - minU)
        #v_tensors = (v_tensors - minV) / (maxV - minV)

        mask = u_tensors.clone().detach()
        # 1 if land, 0 if water
        for x in range(94):
            for y in range(44):
                if u_tensors[y][x].isnan() or u_tensors[y][x].isnan():
                    mask[y][x] = 0
                    u_tensors[y][x] = 0
                    u_tensors[y][x] = 0
                else:
                    mask[y][x] = 1

        combined_tensor = torch.stack((u_tensors, u_tensors, mask))

        return combined_tensor