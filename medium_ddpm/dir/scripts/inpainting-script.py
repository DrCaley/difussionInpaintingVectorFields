import math
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, Lambda
import matplotlib.pyplot as plt

from dataloaders.dataloader import OceanImageDataset
from medium_ddpm.dir.ddpm import MyDDPM
from medium_ddpm.dir.inpainting import inpaint_generate_new_images, calculate_mse, naive_inpaint
from medium_ddpm.dir.masks import generate_squiggly_line_mask, generate_random_mask, generate_straight_line_mask
from medium_ddpm.dir.resize_tensor import ResizeTransform
from medium_ddpm.dir.unets.unet_resized_2_channel_xl import MyUNet
from medium_ddpm.dir.tensors_to_png import generate_png

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

n_steps, min_beta, max_beta = 1000, 1e-4, 0.02
store_path = "./ddpm_ocean_xl.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(store_path, map_location=device)

if 'model_state_dict' in checkpoint:
    model_state_dict = checkpoint['model_state_dict']
else:
    model_state_dict = checkpoint

best_model = MyDDPM(MyUNet(n_steps), n_steps=n_steps, device=device)
best_model.load_state_dict(model_state_dict)
best_model.eval()
print("Model loaded")

transform = Compose([
    Lambda(lambda x: (x - 0.5) * 2),  # Normalize to range [-1, 1]
    ResizeTransform((2, 64, 128))  # Resized to (2, 64, 128)
])

data = OceanImageDataset(
    mat_file="../../../data/rams_head/stjohn_hourly_5m_velocity_ramhead_v2.mat",
    boundaries="../../../data/rams_head/boundaries.yaml",
    num=10,
    transform=transform
)

train_len = int(math.floor(len(data) * 0.7))
test_len = int(math.floor(len(data) * 0.15))
val_len = len(data) - train_len - test_len

training_data, test_data, validation_data = random_split(data, [train_len, test_len, val_len])

batch_size = 1
loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size)
val_loader = DataLoader(validation_data, batch_size=batch_size)

for batch in test_loader:
    input_image = batch[0].to(device)
    break

def reverse_normalization(tensor):
    return (tensor + 1) / 2

input_image_original = reverse_normalization(input_image)

# land is 0, ocean is 1
land_mask = (input_image_original != 0).float()

# Change to 'square', 'squiggly', or 'straight_line'
mask_type = 'squiggly'

if mask_type == 'square':
    mask = generate_random_mask(input_image.shape, input_image_original)
elif mask_type == 'squiggly':
    mask = generate_squiggly_line_mask(input_image.shape, input_image_original)
elif mask_type == 'straight_line':
    mask = generate_straight_line_mask(input_image.shape, input_image_original, orientation='horizontal')

mask = mask.to(device)

final_image_ddpm = inpaint_generate_new_images(
    best_model,
    input_image,
    mask,
    n_samples=1,
    device=device,
    gif_name="ocean_inpainting.gif"
)

naive_inpainted_image = naive_inpaint(input_image, mask)

mse_naive = calculate_mse(input_image, naive_inpainted_image, mask)
print(f"MSE (Naive Inpainting) on {mask_type} mask: {mse_naive.item()}")

mse_ddpm = calculate_mse(input_image, final_image_ddpm, mask)
print(f"MSE (DDPM Inpainting) on {mask_type} mask: {mse_ddpm.item()}")

generate_png(input_image, filename='input_image.png')
generate_png(final_image_ddpm, filename='final_image_ddpm.png')
generate_png(naive_inpainted_image, filename='naive_inpainted_image.png')
generate_png(mask, filename=f'{mask_type}_mask.png')
