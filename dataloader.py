import os
import pandas as pd
import yaml
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from image_noiser import generate_noised_images
import numpy as np
import datetime


class OceanImageDataset(Dataset):
    def __init__(self, mat_file, boundaries, img_dir):
        self.mat_data = loadmat(mat_file)
        self.img_labels = pd.DataFrame(self.mat_data['ocean_time'])

        with open(boundaries, 'r') as file:
            self.boundaries = yaml.safe_load(file)

        self.img_dir = img_dir
        self.generate_image()

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name = self.img_labels.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        label = self.img_labels.iloc[idx, 1]

        return image, label

    def __generate_image(self, image_num):
        """ Generates images """
        u_array = self.mat_data['u']
        v_array = self.mat_data['v']
        time = self.mat_data['ocean_time'].squeeze()
        time_array = [
            datetime.datetime.fromordinal(int(t)) + datetime.timedelta(days=t % 1) - datetime.timedelta(days=366) for t
            in time]

        # previously found using the normal method
        minU, maxU = -0.8973235906436031, 1.0859991093945718
        minV, maxV = -0.6647028130174489, 0.5259408400292674

        # normalize and multiply by 255

        adjusted_u_arr = 255 * (u_array - minU) / (maxU - minU)
        adjusted_v_arr = 255 * (v_array - minV) / (maxV - minV)

        for y in range(94):
            for x in range(44):
                isLand = False
                if np.isnan(adjusted_u_arr[y][x][image_num]):  # when there was no u value
                    curr_u = 0
                    isLand = True  # the minimal dataloader displayed it is land
                else:
                    curr_u = int(adjusted_u_arr[y][x][image_num])
                if np.isnan(adjusted_v_arr[y][x][0]):  # when there was no v value
                    curr_v = 0  # the minimal dataloader displayed it as a place in the ocean with no current
                    curr_u = 0
                else:
                    curr_v = int(adjusted_v_arr[y][x][image_num])

                img.putpixel((y, 43 - x), (curr_u, curr_v, isLand * 255))

        img.save()
        return img


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

training_data = OceanImageDataset(
    mat_file="./data/rams_head/stjohn_hourly_5m_velocity_ramhead_v2.mat",
    boundaries="./data/rams_head/boundaries.yaml",
    img_dir="./data/images"
)

train_dataloader = DataLoader(training_data, batch_size=50, shuffle=True)

output_dir = './data/noisy_images'
os.makedirs(output_dir, exist_ok=True)

for batch_idx, (images, labels) in enumerate(train_dataloader):
    for i, img in enumerate(images):
        img = img.permute(1, 2, 0).numpy()  # Convert tensor to numpy array
        noisy_img = generate_noised_images(img, mode="gaussian", total_iterations=1000, display_iterations=7,
                                           save_final=True, output_dir=output_dir)
        # save_noisy_image(noisy_img, "gaussian", batch_idx * len(images) + i, output_dir)
