import math
import random
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, Lambda
import logging

from dataloaders.dataloader import OceanImageDataset
from medium_ddpm.dir.ddpm import MyDDPM
from medium_ddpm.dir.inpainting_utils import inpaint_generate_new_images, calculate_mse, naive_inpaint
from medium_ddpm.dir.masks import generate_straight_line_mask, generate_robot_path_mask
from medium_ddpm.dir.resize_tensor import ResizeTransform
from medium_ddpm.dir.unets.unet_xl import MyUNet
from utils.tensors_to_png import generate_png

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

n_steps, min_beta, max_beta = 1000, 1e-4, 0.02
store_path = "../../../models/ddpm_ocean_xl.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(store_path, map_location=device)
model_state_dict = checkpoint.get('model_state_dict', checkpoint)

best_model = MyDDPM(MyUNet(n_steps), n_steps=n_steps, device=device)
try:
    logging.info("Loading model")
    best_model.load_state_dict(model_state_dict)
    best_model.eval()
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    exit(1)

transform = Compose([
    Lambda(lambda x: (x - 0.5) * 2),  # Normalize to range [-1, 1]
    ResizeTransform((2, 64, 128))  # Resized to (2, 64, 128)
])

try:
    logging.info("Preparing data")
    data = OceanImageDataset(
        mat_file="../../../data/rams_head/stjohn_hourly_5m_velocity_ramhead_v2.mat",
        boundaries="../../../data/rams_head/boundaries.yaml",
        num=8,
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
    logging.info("Data prepared successfully")
except Exception as e:
    logging.error(f"Error loading data: {e}")
    exit(1)

def reverse_normalization(tensor):
    return (tensor + 1) / 2

line_numbers = [50, 75, 90]

mse_naive_list = []
mse_ddpm_list = []

try:
    logging.info("Processing data")
    image_counter = 0
    num_images_to_process = 1
    n_samples = 1

    for batch in loader:
        if image_counter >= num_images_to_process:
            break

        input_image = batch[0].to(device)
        input_image_original = reverse_normalization(input_image)
        land_mask = (input_image_original != 0).float()

        for num_lines in line_numbers:
            mask = generate_robot_path_mask(input_image.shape, land_mask, num_squares=num_lines)
            mask = mask.to(device)

            mse_ddpm_samples = []
            for i in range(n_samples):
                final_image_ddpm = inpaint_generate_new_images(
                    best_model,
                    input_image,
                    mask,
                    n_samples=1,
                    device=device
                )

                mse_ddpm = calculate_mse(input_image, final_image_ddpm, mask)
                mse_ddpm_samples.append(mse_ddpm.item())
                logging.info(
                    f"MSE (DDPM Inpainting) with {num_lines} lines for image {image_counter}, sample {i}: {mse_ddpm.item()}")

                naive_inpainted_image = naive_inpaint(input_image, mask)
                mse_naive = calculate_mse(input_image, naive_inpainted_image, mask)
                mse_naive_list.append(mse_naive.item())
                logging.info(
                    f"MSE (Naive Inpainting) with {num_lines} lines for image {image_counter}: {mse_naive.item()}")

                generate_png(input_image, filename=f'input_image.png')
                generate_png(final_image_ddpm, filename=f'ddpm_lines_{num_lines}_sample_{i}.png')
                generate_png(naive_inpainted_image, filename=f'naive_lines_{num_lines}_sample_{i}.png')
                generate_png(mask, filename=f'mask_lines_{num_lines}_sample_{i}.png')
                generate_png(land_mask, filename=f'land_mask_lines_{num_lines}_sample_{i}.png')

                del final_image_ddpm
                del naive_inpainted_image

                torch.cuda.empty_cache()

            mean_mse_ddpm_samples = np.mean(mse_ddpm_samples)
            mse_ddpm_list.append(mean_mse_ddpm_samples)

        image_counter += 1

    mean_mse_naive = np.mean(mse_naive_list)
    mean_mse_ddpm = np.mean(mse_ddpm_list)
    logging.info(f"Mean MSE (Naive Inpainting): {mean_mse_naive}")
    logging.info(f"Mean MSE (DDPM Inpainting): {mean_mse_ddpm}")

    plt.figure(figsize=(10, 6))
    plt.plot(line_numbers, mse_naive_list, label='Naive Inpainting MSE', marker='o')
    plt.plot(line_numbers, mse_ddpm_list, label='DDPM Inpainting MSE', marker='x')
    plt.xlabel('Number of Lines')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('Comparison of MSE between Naive Inpainting and DDPM Inpainting')
    plt.legend()
    plt.grid(True)
    plt.show()

except Exception as e:
    logging.error(f"Error during processing: {e}")
    exit(1)