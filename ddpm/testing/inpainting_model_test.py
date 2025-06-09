import math
import os.path
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose
import logging
import csv
import yaml
import sys
import pickle

from data_prep.data_initializer import DDInitializer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from data_prep.ocean_image_dataset import OceanImageDataset
from ddpm.neural_networks.ddpm_gaussian import MyDDPMGaussian
from ddpm.helper_functions.inpainting_utils import inpaint_generate_new_images, calculate_mse
from ddpm.helper_functions.masks import generate_random_path_mask
from ddpm.helper_functions.resize_tensor import resize_transform
from ddpm.helper_functions.standardize_data import standardize_data
from ddpm.neural_networks.unets.unet_xl import MyUNet

data_initializer = DDInitializer()

# Output goes to file, not console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename="inpainting_model_test_log.txt")

# Load the YAML file
if not os.path.exists('../../data.yaml') :
    using_pycharm = False
    with open('data.yaml', 'r') as file:
        config = yaml.safe_load(file)
else :
    using_pycharm = True
    with open('../../data.yaml', 'r') as file:
        config = yaml.safe_load(file)
print("loaded yaml file.")

using_dumb_pycharm = True
# Load the pickle
if using_dumb_pycharm :
    with open('../../data.pickle', 'rb') as f:
        training_data_np, validation_data_np, test_data_np = pickle.load(f)
else:
    with open('data.pickle', 'rb') as f:
        training_data_np, validation_data_np, test_data_np = pickle.load(f)

training_tensor = torch.from_numpy(training_data_np).float()
validation_tensor = torch.from_numpy(validation_data_np).float()
test_tensor = torch.from_numpy(test_data_np).float()

# ======== Random Seed Initialization ========
SEED = config['testSeed']
snapshots = config['snapshots']
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ======== Model Configuration ========
n_steps, min_beta, max_beta = config['n_steps'], config['min_beta'], config['max_beta']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if len(sys.argv) < 2 :
    print("Usage: python3 inpainting_model_test.py <model file ending with .pt>")
    store_path = config['model_path']
else :
    store_path = sys.argv[1]

# ======== Load DDPM Checkpoint ========
checkpoint = torch.load(store_path, map_location=device)
model_state_dict = checkpoint.get('model_state_dict', checkpoint)

# Create and load the DDPM model with UNet backbone
best_model = MyDDPMGaussian(MyUNet(n_steps), n_steps=n_steps, device=device)
try:
    logging.info("Loading model")
    best_model.load_state_dict(model_state_dict)
    best_model.eval()
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    exit(1)

# ======== Data Transformation Setup ========
standardizer = standardize_data(config['u_training_mean'], config['u_training_std'], config['v_training_mean'], config['v_training_std'])

# Compose resize and standardization transforms
transform = Compose([
    resize_transform((2, 64, 128)),
    standardizer
])

# ======== Dataset Loading & Splitting ========
try:
    logging.info("Preparing data")
    boundaries_file = "../../data/rams_head/boundaries.yaml" if using_pycharm else "./data/rams_head/boundaries.yaml"

    training_tensor = torch.from_numpy(training_data_np).float()
    validation_tensor = torch.from_numpy(validation_data_np).float()
    test_tensor = torch.from_numpy(test_data_np).float()

    batch_size = 1

    training_data = OceanImageDataset(
        data_tensor=training_tensor,
        boundaries=boundaries_file,
        transform=transform
    )
    test_data = OceanImageDataset(
        data_tensor=test_tensor,
        boundaries=boundaries_file,
        transform=transform
    )
    validation_data = OceanImageDataset(
        data_tensor=validation_tensor,
        boundaries=boundaries_file,
        transform=transform
    )

    # Set up dataloaders
    batch_size = config['batch_size']
    train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    val_loader = DataLoader(validation_data, batch_size=batch_size)

    logging.info("Data prepared successfully")
except Exception as e:
    logging.error(f"Error loading data: {e}")
    exit(1)


# ======== Inpainting Evaluation Parameters ========
line_numbers = [10, 20, 40]          # Number of lines in the mask
resample_nums = [5]                 # Number of resampling steps
masks_to_test = ["random_path_thin", "random_path_thick"]  # Mask types
mse_ddpm_list = []                 # To store average MSEs per image


def Testing():
    # writes data to csv file
    writer = csv.writer(file)
    header = ["image_num", "mask_type", "num_lines", "resample_steps", "mse"]
    writer.writerow(header)
    logging.info("Processing data")
    image_counter = 0
    num_images_to_process = 5
    n_samples = 1  # Number of samples per mask config

    loader = train_loader  # change to test_loader, val_loader depending on what you want to test

    # ======== Loop Through Batches ========
    for batch in loader:
        if image_counter >= num_images_to_process:
            break

        input_image = batch[0].to(device)  # (Batch size, Channels, Height, Width)

        # Convert back to unstandardized form for land masking
        # TODO: Fix all this/make it nicer/sanity check
        input_image_original = standardizer.unstandardize(input_image)
        land_mask = (input_image_original != 0).float()

        # ======== Masking and Inpainting Loops ========
        for mask_type in masks_to_test:
            for resample in resample_nums:
                for num_lines in line_numbers:

                    # Generate a mask with different parameters
                    if mask_type == "random_path_thin":
                        mask = generate_random_path_mask(input_image.shape, land_mask, num_lines=num_lines,
                                                         line_thickness=1)
                    elif mask_type == "random_path_thick":
                        mask = generate_random_path_mask(input_image.shape, land_mask, num_lines=num_lines,
                                                         line_thickness=5)

                    mask = mask.to(device)
                    torch.save(mask, f"results/predicted/{mask_type}_{num_lines}.pt")

                    mse_ddpm_samples = []

                    # ======== Generate Samples ========
                    for i in range(n_samples):
                        final_image_ddpm = inpaint_generate_new_images(
                            best_model,
                            input_image,
                            mask,
                            n_samples=1,  # number of samples to generate. I think it doesn't work, not sure
                            device=device,
                            resample_steps=resample
                        )

                        # Save inpainted result and mask
                        torch.save(final_image_ddpm,
                                   f"results/predicted/img{batch[1].item()}_{mask_type}_resample{resample}_num_lines_{num_lines}.pt")
                        torch.save(mask,
                                   f"results/predicted/mask{batch[1].item()}_{mask_type}_resample{resample}_num_lines_{num_lines}.pt");

                        # Calculate MSE for masked region
                        mse_ddpm = calculate_mse(input_image, final_image_ddpm, mask)
                        mse_ddpm_samples.append(mse_ddpm.item())

                        logging.info(
                            f"MSE (DDPM Inpainting) with {num_lines} lines for image {image_counter}, sample {i}: {mse_ddpm.item()}")

                        # Write result to CSV
                        output = [image_counter, mask_type, num_lines, resample, mse_ddpm.item()]
                        writer.writerow(output)

                        del final_image_ddpm
                        torch.cuda.empty_cache()

                # Compute average MSE over all samples for this mask config
                mean_mse_ddpm_samples = np.mean(mse_ddpm_samples)
                mse_ddpm_list.append(mean_mse_ddpm_samples)

        image_counter += 1

    # ======== Log Final Averages ========
    mean_mse_ddpm = np.mean(mse_ddpm_list)
    logging.info(f"Mean MSE (DDPM Inpainting): {mean_mse_ddpm}")




# ======== CSV Output File for Results ========
with open("inpainting-xl-data.csv", "w", newline="") as file:
    try:
        Testing()
    except Exception as e:
        logging.error(f"Error during processing: {e}")
        exit(1)
