import math
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, Lambda
import logging
import csv

from dataloaders.dataloader import OceanImageDataset
from medium_ddpm.ddpm import MyDDPM
from medium_ddpm.inpainting_utils import inpaint_generate_new_images, calculate_mse
from medium_ddpm.masks import (generate_random_path_mask)
from medium_ddpm.resize_tensor import ResizeTransform
from medium_ddpm.unets.unet_xl import MyUNet
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename="../DDPM/Testing/inpainting_model_test_log.txt")

"""Inpaints, records data about how well the model is doing"""

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

n_steps, min_beta, max_beta = 1000, 1e-4, 0.02
store_path = "../DDPM/Trained_Models/ddpm_ocean_good_normalized.pt"
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
        mat_file="../data/rams_head/stjohn_hourly_5m_velocity_ramhead_v2.mat",
        boundaries="../../../data/rams_head/boundaries.yaml",
        num=17040,
        transform=transform
    )

    train_len = int(math.floor(len(data) * 0.7))
    test_len = int(math.floor(len(data) * 0.15))
    val_len = len(data) - train_len - test_len

    training_data, test_data, validation_data = random_split(data, [train_len, test_len, val_len])

    batch_size = 1
    train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    val_loader = DataLoader(validation_data, batch_size=batch_size)
    logging.info("Data prepared successfully")
except Exception as e:
    logging.error(f"Error loading data: {e}")
    exit(1)


def reverse_normalization(tensor):
    return (tensor + 1) / 2


line_numbers = [40]
resample_nums = [5]
masks_to_test = ["random_path_thick"]
mse_ddpm_list = []

with open("../DDPM/Testing/inpainting-xl-data.csv", "w", newline="") as file:
    try:
        writer = csv.writer(file)
        header = ["image_num", "mask_type", "num_lines" "resample_steps", "mse"]
        writer.writerow(header)
        logging.info("Processing data")
        image_counter = 0
        num_images_to_process = 1
        n_samples = 1

        loader = val_loader

        for batch in loader:
            if image_counter >= num_images_to_process:
                break

            input_image = batch[0].to(device)
            torch.save(input_image, "../DDPM/Testing/results/myInput.pt")
            print("i saved my input")
            input_image_original = reverse_normalization(input_image)
            land_mask = (input_image_original != 0).float()

            for mask_type in masks_to_test:
                for resample in resample_nums:
                    for num_lines in line_numbers:
                        mask = None
                        if mask_type == "random_path_thin":
                            mask = generate_random_path_mask(input_image.shape, land_mask, num_lines=num_lines, line_thickness=1)
                        elif mask_type == "random_path_thick":
                            mask = generate_random_path_mask(input_image.shape, land_mask, num_lines=num_lines, line_thickness=5)
                        mask = mask.to(device)
                        torch.save(mask, "../DDPM/Testing/results/myMask.pt")

                        mse_ddpm_samples = []
                        for i in range(n_samples):
                            print("hello?")
                            final_image_ddpm = inpaint_generate_new_images(
                                best_model,
                                input_image,
                                mask,
                                n_samples=1,
                                device=device,
                                resample_steps=resample
                            )

                            torch.save(final_image_ddpm,
                                       f"../Testing/results/predicted:img{batch[1].item()}_{mask_type}_resample{resample}.pt")

                            mse_ddpm = calculate_mse(input_image, final_image_ddpm, mask)
                            mse_ddpm_samples.append(mse_ddpm.item())
                            logging.info(
                                f"MSE (DDPM Inpainting) with {num_lines} lines for image {image_counter}, sample {i}: {mse_ddpm.item()}")

                            output = [image_counter, mask_type, num_lines, resample, mse_ddpm.item()]
                            writer.writerow(output)

                            del final_image_ddpm
                            torch.cuda.empty_cache()

                    mean_mse_ddpm_samples = np.mean(mse_ddpm_samples)
                    mse_ddpm_list.append(mean_mse_ddpm_samples)

            image_counter += 1

        mean_mse_ddpm = np.mean(mse_ddpm_list)
        logging.info(f"Mean MSE (DDPM Inpainting): {mean_mse_ddpm}")


    except Exception as e:
        logging.error(f"Error during processing: {e}")
        exit(1)
