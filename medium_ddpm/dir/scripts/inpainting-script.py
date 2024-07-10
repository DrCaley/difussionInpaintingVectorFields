import math
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, Lambda

from dataloaders.dataloader import OceanImageDataset
from medium_ddpm.dir.ddpm import MyDDPM
from medium_ddpm.dir.inpainting import inpaint_generate_new_images, calculate_mse, avg_pixel_value, naive_inpaint
from medium_ddpm.dir.resize_tensor import ResizeTransform
from medium_ddpm.dir.unets.unet_resized_2_channel_xl import MyUNet
from medium_ddpm.dir.utils import display_side_by_side

def generate_random_mask(image_shape, max_mask_size=32):
    h, w = image_shape
    mask = torch.zeros((1, h, w), dtype=torch.float32)
    mask_h = random.randint(1, max_mask_size)
    mask_w = random.randint(1, max_mask_size)
    start_h = random.randint(0, h - mask_h)
    start_w = random.randint(0, w - mask_w)
    mask[:, start_h:start_h + mask_h, start_w:start_w + mask_w] = 1
    return mask

# Setting reproducibility
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

n_steps, min_beta, max_beta = 1000, 1e-4, 0.02
store_path = "./ddpm_ocean_xl.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(store_path, map_location=device)
print("Checkpoint keys:", checkpoint.keys())

if 'model_state_dict' in checkpoint:
    model_state_dict = checkpoint['model_state_dict']
else:
    model_state_dict = checkpoint

best_model = MyDDPM(MyUNet(n_steps), n_steps=n_steps, device=device)
best_model.load_state_dict(torch.load(store_path, map_location=device)['model_state_dict'])
best_model.eval()
print("Model loaded")

transform = Compose([
    Lambda(lambda x: (x - 0.5) * 2),  # Normalize to range [-1, 1]
    ResizeTransform((2, 64, 128))  # Resized to (1, 64, 128)
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

# Fill with 0's size of img, then place 1s (missing regions)
mask = torch.zeros_like(input_image)
mask[:, :, 10:20, 50:60] = 1

final_image = inpaint_generate_new_images(
    best_model,
    input_image,
    mask,
    n_samples=1,
    device=device,
    gif_name="ocean_inpainting.gif"
)

mask = generate_random_mask((64, 128)).to(device)

naive_inpainted_image = naive_inpaint(input_image, mask)

# Ensure mask shape is compatible for display
mask = mask.unsqueeze(0)

mse_naive = calculate_mse(input_image, naive_inpainted_image, mask)
display_side_by_side(input_image, mask, naive_inpainted_image, title="Naive Inpainted Image")

print(f"MSE (Naive Inpainting): {mse_naive.item()}")

mse_ddpm = calculate_mse(input_image, final_image, mask)
print(f"MSE (DDPM Inpainting): {mse_ddpm.item()}")
display_side_by_side(input_image, mask, final_image, title="Inpainted Image")
