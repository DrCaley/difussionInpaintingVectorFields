import math
import random
import imageio
import numpy as np
import torch
from einops import rearrange
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, Lambda
import torch.nn.functional as F

from dataloaders.dataloader import OceanImageDataset
from medium_ddpm.dir.ddpm import MyDDPM
from medium_ddpm.dir.resize_tensor import ResizeTransform
from medium_ddpm.dir.unets.unet_resized_2_channel_xl import MyUNet
from medium_ddpm.dir.utils import display_side_by_side

# Setting reproducibility
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

def inpaint_generate_new_images(ddpm, input_image, mask, n_samples=16, device=None, frames_per_gif=100,
                                gif_name="sampling.gif", c=1, h=64, w=128):
    """Given a DDPM model, an input image, and a mask, generates inpainted samples"""
    frame_idxs = np.linspace(0, ddpm.n_steps - 1, frames_per_gif).astype(np.uint)
    frames = []
    noised_imgs = [None] * (ddpm.n_steps + 1)

    with torch.no_grad():
        if device is None:
            device = ddpm.device

        input_img = input_image.clone().to(device)
        noise = torch.randn_like(input_img).to(device)

        # Adding noise step by step
        noised_imgs[0] = input_img
        for t in range(ddpm.n_steps):
            eta = torch.randn_like(input_img).to(device)
            noised_imgs[t+1] = ddpm(noised_imgs[t], t, eta, one_step=True)

        x = noised_imgs[ddpm.n_steps] * (1 - mask) + (noise * mask)

        for idx, t in enumerate(range(ddpm.n_steps - 1, -1, -1)):
            time_tensor = (torch.ones(n_samples, 1) * t).to(device).long()
            eta_theta = ddpm.backward(x, time_tensor)

            alpha_t = ddpm.alphas[t]
            alpha_t_bar = ddpm.alpha_bars[t]

            # Partially denoising the image
            less_noised_img = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta)

            if t > 0:
                z = torch.randn(n_samples, c, h, w).to(device)
                beta_t = ddpm.betas[t]
                sigma_t = beta_t.sqrt()
                less_noised_img = less_noised_img + sigma_t * z

            x = noised_imgs[t] * (1 - mask) + (less_noised_img * mask)

            # Adding frames to the GIF
            if idx in frame_idxs or t == 0:
                # Normalize and prepare the frame for the GIF
                normalized = x.clone()
                for i in range(len(normalized)):
                    normalized[i] -= torch.min(normalized[i])
                    normalized[i] *= 255 / torch.max(normalized[i])

                frame = rearrange(normalized, "(b1 b2) c h w -> (b1 h) (b2 w) c", b1=int(n_samples ** 0.5))
                frame = frame.cpu().numpy().astype(np.uint8)

                frames.append(frame)

    # Storing the gif
    with imageio.get_writer(gif_name, mode="I") as writer:
        for idx, frame in enumerate(frames):
            rgb_frame = np.repeat(frame, 3, axis=2)
            writer.append_data(rgb_frame)

            # Show the last frame for a longer time
            if idx == len(frames) - 1:
                last_rgb_frame = np.repeat(frames[-1], 3, axis=2)
                for _ in range(frames_per_gif // 3):
                    writer.append_data(last_rgb_frame)
    return x

def calculate_mse(original_image, predicted_image, mask):
    """Calculate Mean Squared Error between the original and predicted images for the masked area only."""
    masked_original = original_image * mask
    masked_predicted = predicted_image * mask
    mse = F.mse_loss(masked_predicted, masked_original, reduction='sum') / mask.sum()
    return mse

def avg_pixel_value(original_image, predicted_image, mask):
    avg_pixel_value = torch.sum(torch.abs(original_image * mask)) / mask.sum()

    avg_diff = (torch.sum(torch.abs((predicted_image * mask) - (original_image * mask))) / mask.sum())

    return avg_diff * (100 / avg_pixel_value)


n_steps, min_beta, max_beta = 1000, 1e-4, 0.02
store_path = "./ddpm_ocean_xl.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the checkpoint to inspect its keys
checkpoint = torch.load(store_path, map_location=device)
print("Checkpoint keys:", checkpoint.keys())

# Assuming 'model_state_dict' is not found, we check for other possible keys
if 'model_state_dict' in checkpoint:
    model_state_dict = checkpoint['model_state_dict']
else:
    model_state_dict = checkpoint  # Assuming the checkpoint directly contains the state dict

best_model = MyDDPM(MyUNet(n_steps), n_steps=n_steps, device=device)
best_model.load_state_dict(torch.load(store_path, map_location=device)['model_state_dict'])  # Correctly load the model state dict
best_model.eval()
print("Model loaded")

transform = Compose([
    Lambda(lambda x: (x - 0.5) * 2),  # Normalize to range [-1, 1]
    ResizeTransform((2, 64, 128))  # Resized to (1, 64, 128)
])

data = OceanImageDataset(
    mat_file="../../data/rams_head/stjohn_hourly_5m_velocity_ramhead_v2.mat",
    boundaries="../../data/rams_head/boundaries.yaml",
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

mse = calculate_mse(input_image, final_image, mask)
print("Mean Squared Error:", mse.item())

avg = avg_pixel_value(input_image, final_image, mask)
print(f"Avg. Pixel Value: %{avg.item()}")

display_side_by_side(input_image, mask, final_image, title="Inpainting Example")
