import torch
import torchvision.transforms as transforms
import yaml
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader

class OceanImageDataset(Dataset):
    def __init__(self, mat_file, boundaries, tensor_dir):
        self.mat_data = loadmat(mat_file)
        self.tensor_labels = []
        self.tensor_arr = []

        with open(boundaries, 'r') as file:
            self.boundaries = yaml.safe_load(file)

        for x in range(5):
            self.tensor_arr.append(self.load_array(x))

        # for dirpath, dirnames, filenames in os.walk(tensor_dir):
        #     for filename in filenames:
        #         self.tensor_arr.append(torch.load(dirpath + filename))
        #         self.tensor_labels.append(filename)

        print('done')

    def __len__(self):
        return len(self.tensor_arr)

    def __getitem__(self, idx):
        tensor = self.tensor_arr[idx]

        target = self.tensor_labels
        return tensor, target

    def load_array(self, tensor_num):
        u_array = self.mat_data['u']
        v_array = self.mat_data['v']
        time = self.mat_data['ocean_time'].squeeze()

        # Previously found using the normal method
        minU, maxU = -0.8973235906436031, 1.0859991093945718
        minV, maxV = -0.6647028130174489, 0.5259408400292674

        # Normalize and multiply by 255
        adjusted_u_arr = (u_array - minU) / (maxU - minU)
        adjusted_v_arr = (v_array - minV) / (maxV - minV)

        #Convert to tensors
        u_tensors = torch.from_numpy(adjusted_u_arr)
        u_tensors = u_tensors.permute(*torch.arange(u_tensors.ndim - 1, -1, -1))[tensor_num]
        v_tensors = torch.from_numpy(adjusted_v_arr)
        v_tensors = v_tensors.permute(*torch.arange(v_tensors.ndim - 1, -1, -1))[tensor_num]

        land_tensors = u_tensors.clone().detach()

        #1 if land, 0 if water
        for x in range(94):
            for y in range(44):
                if land_tensors[y][x].isnan() or v_tensors[y][x].isnan() or u_tensors[y][x].isnan():
                    land_tensors[y][x] = 1
                    v_tensors[y][x] = 0
                    u_tensors[y][x] = 0
                else:
                    land_tensors[y][x] = 0

        combined_tensor = torch.stack((u_tensors, v_tensors, land_tensors))
        # torch.save(combined_tensor, "./data/tensors/" + str(tensor_num) + ".pt")
        return combined_tensor

