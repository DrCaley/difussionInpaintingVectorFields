import torch
import torchvision.transforms as transforms
import yaml
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader
from image_noiser import generate_noised_images
import numpy as np
import datetime


class OceanImageDataset(Dataset):
    def __init__(self, mat_file, boundaries, img_dir):
        self.mat_data = loadmat(mat_file)
        self.img_labels = []

        with open(boundaries, 'r') as file:
            self.boundaries = yaml.safe_load(file)

        self.images = []

        for x in range(5):
            # change to load_array
            self.generate_image(x)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        img_label = self.img_labels[idx]
        transform = transforms.Compose([transforms.PILToTensor()])

        if isinstance(image, Image.Image):
            vector = transform(image)

        return vector, img_label

    def load_array(self, image_num):
        u_array = self.mat_data['u']
        v_array = self.mat_data['v']
        time = self.mat_data['ocean_time'].squeeze()

        # Previously found using the normal method
        minU, maxU = -0.8973235906436031, 1.0859991093945718
        minV, maxV = -0.6647028130174489, 0.5259408400292674

        # Normalize and multiply by 255
        adjusted_u_arr = 255 * (u_array - minU) / (maxU - minU)
        adjusted_v_arr = 255 * (v_array - minV) / (maxV - minV)

        for y in range(94):
            for x in range(44):
                pass

    def generate_image(self, image_num, img_dir):
        """Generates images"""
        img = Image.new('RGB', (94, 44), color='white')

        u_array = self.mat_data['u']
        v_array = self.mat_data['v']
        time = self.mat_data['ocean_time'].squeeze()
        time_array = [
            datetime.datetime.fromordinal(int(t)) + datetime.timedelta(days=t % 1) - datetime.timedelta(days=366) for t
            in time]

        # Previously found using the normal method
        minU, maxU = -0.8973235906436031, 1.0859991093945718
        minV, maxV = -0.6647028130174489, 0.5259408400292674

        # Normalize and multiply by 255
        adjusted_u_arr = 255 * (u_array - minU) / (maxU - minU)
        adjusted_v_arr = 255 * (v_array - minV) / (maxV - minV)

        for y in range(94):
            for x in range(44):
                isLand = False
                if np.isnan(adjusted_u_arr[y][x][image_num]):  # When there was no u value
                    curr_u = 0
                    isLand = True  # The minimal dataloader displayed it is land
                else:
                    curr_u = int(adjusted_u_arr[y][x][image_num])
                if np.isnan(adjusted_v_arr[y][x][0]):  # When there was no v value
                    curr_v = 0  # The minimal dataloader displayed it as a place in the ocean with no current
                    curr_u = 0
                else:
                    curr_v = int(adjusted_v_arr[y][x][image_num])

                img.putpixel((y, 43 - x), (curr_u, curr_v, isLand * 255))

        self.images.append(img)
        self.img_labels.append(image_num)
        # img.save(os.path.join(img_dir, 'ocean_image' + str(image_num) + '.png'), 'PNG')
        # return img


training_data = OceanImageDataset(
    mat_file="./data/rams_head/stjohn_hourly_5m_velocity_ramhead_v2.mat",
    boundaries="./data/rams_head/boundaries.yaml",
    img_dir="./data/images"
)

train_loader = DataLoader(
    training_data,
    batch_size=1,
    shuffle=True
)

# Iterate over dataset
# Hard-coded to only load first 5 images
for epoch in range(1):
    for i, data in enumerate(train_loader):
        tensor, label = data
        # apply noise to each img
        noised_tensor = torch.from_numpy(generate_noised_images(tensor))
