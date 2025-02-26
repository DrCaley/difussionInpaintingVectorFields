import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from plots.adapt_visualize_data import plot_png, plot_tensor
from dataloaders.dataloader import OceanImageDataset


def normalize_image(img):
    norm_image = (img / 127.5) - 1.0
    return norm_image


def calculate_mse(original_image, reconstructed_image, mask):
    mse = np.mean((original_image - reconstructed_image) ** 2)
    return mse


# image = Image.open('images/input_img_cropped.png').convert('RGB')
# image = np.array(image, dtype=np.float32)
# normalized_image = normalize_image(image)
#
# mask = Image.open('images/mask_cropped.png').convert('L')
# mask = np.array(mask, dtype=np.float32)
# missing_pixel_mask = mask > 0
# missing_pixel_mask = ~missing_pixel_mask
#
# masked_image = normalized_image.copy()
# masked_image[missing_pixel_mask] = np.nan
#
# expanded_mask = np.repeat(missing_pixel_mask[..., np.newaxis], 3, axis=2)

# gp_image = Image.open('images/gp_image.png')
# gp_image = np.array(gp_image, dtype=np.float32)
# normalized_gp_image = normalize_image(gp_image)
# mse_gp = calculate_mse(normalized_image, normalized_gp_image, expanded_mask)
# print(f"GP MSE: {mse_gp}")
#
# ddpm_image = Image.open('images/ddpm_img_cropped.png')
# ddpm_image = np.array(ddpm_image, dtype=np.float32)
# normalized_ddpm_image = normalize_image(ddpm_image)
# mse_ddpm = calculate_mse(normalized_image, normalized_ddpm_image, expanded_mask)
# print(f"DDPM MSE: {mse_ddpm}")

# plot_png('images/0_ddpm_image_0_sample0_resample20.png', './images/land_mask_cropped.png')
# plot_png('./images/gp_image.png', './images/land_mask_cropped.png')
# plot_png('images/input_img_cropped.png', './images/land_mask_cropped.png')


og1 = torch.load("./og1.pt")
pred1 = torch.load("./img1_random_path_thin_resample5.pt")

# og1 = torch.transpose(og1[0],2,1)[:,0:94,0:44]
og1 = og1[0][:,0:44,0:94]
pred1 = pred1[0][:,0:44,0:94]

data = OceanImageDataset(num=1)
train_loader = DataLoader(data, batch_size=1, shuffle=True)
original_tensor = train_loader.dataset[0][0]

plot_tensor(original_tensor)
# 44 x 94
mask = original_tensor[2]

# torch.cat((og1, torch.zeros(1, 94, 44)), 0)

plot_tensor(torch.cat((og1, mask.unsqueeze(0)), 0))
plot_tensor(torch.cat((pred1, mask.unsqueeze(0)), 0))