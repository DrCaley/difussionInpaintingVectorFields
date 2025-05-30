import math
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, Lambda, Normalize
import logging
import csv
import yaml


from DataPrep.dataloader import OceanImageDataset
from DDPM.Neural_Networks.ddpm import MyDDPM
from DDPM.Helper_Functions.inpainting_utils import inpaint_generate_new_images, calculate_mse
from DDPM.Helper_Functions.masks import (generate_random_path_mask)
from DDPM.Helper_Functions.resize_tensor import ResizeTransform
from DDPM.Helper_Functions.standardize_data import StandardizeData, reverseStandardization
from DDPM.Neural_Networks.unets.unet_xl import MyUNet

#output goes to file, not console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename="inpainting_model_test_log.txt")

"""Inpaints, records data about how well the model is doing"""
"""This is the main file to test the model"""


# Load the YAML file
with open('../../data.yaml', 'r') as file:
    config = yaml.safe_load(file)

#Set seed
SEED = config['testSeed']
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

n_steps, min_beta, max_beta = 1000, 1e-4, 0.02
#change to the path to the model you want to test
store_path = "../Trained_Models/ddpm_ocean_good_normalized.pt"
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
    ResizeTransform((2, 64, 128)),
    StandardizeData(config['u_training_mean'],config['u_training_std'],config['v_training_mean'],config['v_training_std'])# Resized to (2, 64, 128)
])

try:
    logging.info("Preparing data")
    data = OceanImageDataset(
        mat_file="../../data/rams_head/stjohn_hourly_5m_velocity_ramhead_v2.mat",
        boundaries="../../data/rams_head/boundaries.yaml",
        num=10, #number of ocean snapshots to load
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




line_numbers = [10, 20, 40] #parameters of mask to test
resample_nums = [5] #number of times to sample/resample
masks_to_test = ["random_path_thin", "random_path_thick"]
mse_ddpm_list = []

with open("inpainting-xl-data.csv", "w", newline="") as file:
    try:
        #writes data to csv file
        writer = csv.writer(file)
        header = ["image_num", "mask_type", "num_lines", "resample_steps", "mse"]
        writer.writerow(header)
        logging.info("Processing data")
        image_counter = 0
        num_images_to_process = 5
        n_samples = 1

        loader = train_loader #change to test_loader, val_loader depending on what you want to test

        for batch in loader:
            if image_counter >= num_images_to_process:
                break

            input_image = batch[0].to(device)
            input_image_original = reverseStandardization(input_image)
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
                        torch.save(mask, f"results/predicted/{mask_type}_{num_lines}.pt")
                        mse_ddpm_samples = []
                        for i in range(n_samples):
                            final_image_ddpm = inpaint_generate_new_images(
                                best_model,
                                input_image,
                                mask,
                                n_samples=1, #number of samples to generate. I think it doesn't work, not sure
                                device=device,
                                resample_steps=resample
                            )

                            #saves tensor and mask
                            torch.save(final_image_ddpm, f"results/predicted/img{batch[1].item()}_{mask_type}_resample{resample}_num_lines_{num_lines}.pt")
                            torch.save(mask, f"results/predicted/mask{batch[1].item()}_{mask_type}_resample{resample}_num_lines_{num_lines}.pt");
                            #find mse
                            mse_ddpm = calculate_mse(input_image, final_image_ddpm, mask)
                            mse_ddpm_samples.append(mse_ddpm.item())
                            #print mse to
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
