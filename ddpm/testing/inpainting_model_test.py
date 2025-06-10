import os.path
import numpy as np
import torch
from torch.utils.data import DataLoader
import logging
import csv
import sys

from data_prep.data_initializer import DDInitializer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from ddpm.neural_networks.ddpm import MyDDPMGaussian
from ddpm.helper_functions.inpainting_utils import inpaint_generate_new_images, calculate_mse
from ddpm.helper_functions.masks import generate_random_path_mask
from ddpm.neural_networks.unets.unet_xl import MyUNet

dd = DDInitializer()

(training_tensor, validation_tensor, test_tensor) = dd.get_tensors()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename="inpainting_model_test_log.txt")

# ======== Model Configuration ========
n_steps = dd.get_attribute("n_steps")
min_beta = dd.get_attribute("min_beta")
max_beta = dd.get_attribute("max_beta")

if len(sys.argv) < 2 :
    print("Usage: python3 inpainting_model_test.py <model file ending with .pt>")
    store_path = dd.get_attribute("model_path")
else :
    store_path = sys.argv[1]

# ======== Load DDPM Checkpoint ========
checkpoint = torch.load(store_path, map_location=dd.get_device())
model_state_dict = checkpoint.get('model_state_dict', checkpoint)

# Create and load the DDPM model with UNet backbone
best_model = MyDDPMGaussian(MyUNet(n_steps), n_steps=n_steps, device=dd.get_device())
try:
    logging.info("Loading model")
    best_model.load_state_dict(model_state_dict)
    best_model.eval()
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    exit(1)

# ======== Dataset Loading & Splitting ========
try:
    logging.info("Preparing data")

    batch_size = dd.get_attribute("inpainting_batch_size")

    training_data = dd.get_training_data()
    test_data = dd.get_test_data()
    validation_data = dd.get_validation_data()

    # Set up dataloaders
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

    loader = train_loader  # TODO: Add this being changeable to yaml

    # ======== Loop Through Batches ========
    for batch in loader:
        if image_counter >= num_images_to_process:
            break

        input_image = batch[0].to(dd.get_device()) # (Batch size, Channels, Height, Width)

        # Convert back to unstandardized form for land masking
            #TODO: Fix all this/make it nicer/sanity check
        input_image_original = dd.get_standardizer().unstandardize(input_image)
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

                        mask = mask.to(dd.get_device())
                    torch.save(mask, f"results/predicted/{mask_type}_{num_lines}.pt")

                    mse_ddpm_samples = []

                    # ======== Generate Samples ========
                    for i in range(n_samples):
                        final_image_ddpm = inpaint_generate_new_images(
                            best_model,
                            input_image,
                            mask,
                                n_samples=1, #number of samples to generate. I think it doesn't work, not sure
                                device=dd.get_device(),
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
