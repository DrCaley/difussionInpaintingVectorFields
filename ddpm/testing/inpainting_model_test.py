import os.path
import numpy as np
import torch
from torch.utils.data import DataLoader
import logging
import csv
import sys

from tqdm import tqdm

from ddpm.helper_functions.interpolation_tool import interpolate_masked_velocity_field

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from ddpm.helper_functions.mask_factory.masks.abstract_mask import MaskGenerator
from ddpm.helper_functions.mask_factory.masks.straigth_line import StraightLineMaskGenerator
from data_prep.data_initializer import DDInitializer
from ddpm.neural_networks.ddpm import MyDDPMGaussian
from ddpm.utils.inpainting_utils import inpaint_generate_new_images, calculate_mse, top_left_crop
from ddpm.neural_networks.unets.unet_xl import MyUNet

dd = DDInitializer()
results_path = "./results/"

if not os.path.exists(results_path):
    os.makedirs(results_path, exist_ok=True)

(training_tensor, validation_tensor, test_tensor) = dd.get_tensors()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename=f"{results_path}inpainting_model_test_log.txt")

# ======== Model Configuration ========
n_steps = dd.get_attribute("n_steps")
min_beta = dd.get_attribute("min_beta")
max_beta = dd.get_attribute("max_beta")

store_path = dd.get_attribute("store_path")

if len(sys.argv) < 2:
    print("Usage: python3 inpainting_model_test.py <model file ending with .pt>")
    store_path = dd.get_attribute("model_path")
else:
    if os.path.exists(sys.argv[1]):
        store_path = sys.argv[1]
    else:
        print(sys.argv[1], "not found, using:", store_path)

# ======== Load DDPM Checkpoint ========
checkpoint = torch.load(store_path, map_location=dd.get_device())
model_state_dict = checkpoint.get('model_state_dict', checkpoint)

best_model = MyDDPMGaussian(MyUNet(n_steps), n_steps=n_steps, device=dd.get_device())
try:
    logging.info("Loading model")
    best_model.load_state_dict(model_state_dict)
    best_model.eval()
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    exit(1)

# ======== Dataset Loading ========
try:
    logging.info("Preparing data")
    batch_size = dd.get_attribute("inpainting_batch_size")
    training_data = dd.get_training_data()
    test_data = dd.get_test_data()
    validation_data = dd.get_validation_data()

    train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    val_loader = DataLoader(validation_data, batch_size=batch_size)

    logging.info("Data prepared successfully")
except Exception as e:
    logging.error(f"Error loading data: {e}")
    exit(1)

# ======== Inpainting Evaluation Parameters ========
resample_nums = dd.get_attribute("resample_nums")
mse_ddpm_list = []

# =========== Initializing Masks ==================
robot_mask = StraightLineMaskGenerator(1,1)

masks_to_test = [robot_mask]

def inpaint_testing(mask_generator: MaskGenerator, image_counter: int) -> int:
    writer = csv.writer(file)
    header = ["image_num", "num_lines", "resample_steps", "mse"]
    writer.writerow(header)

    logging.info("Processing data")
    num_images_to_process = dd.get_attribute('num_images_to_process')
    n_samples = dd.get_attribute('n_samples')

    # ======== Loop Through Batches ========
    loader = val_loader
    total_images = min(len(loader.dataset), num_images_to_process)

    with tqdm(total=total_images, desc=f"Mask: {mask_generator}", colour="#00ffff") as main_pbar:
        for step, batch in enumerate(loader):
            logging.info("Processing batch")
            if image_counter >= num_images_to_process:
                break

            device = dd.get_device()
            input_image = batch[0].to(device) # (Batch size, Channels, Height, Width)

            # Convert back to unstandardized form for land masking
            input_image_original = dd.get_standardizer().unstandardize(input_image).to(device)
            land_mask = (input_image_original != 0).float().to(device)

            mask = mask_generator.generate_mask(input_image.shape, land_mask)
            num_lines = mask_generator.get_num_lines()

            # ======== Masking and Inpainting Loops ========
            for resample in resample_nums:

                torch.save(mask, f"{results_path}{mask_generator}_{num_lines}.pt")

                mse_ddpm_samples = []
                # ======== Generate Samples ========
                for i in tqdm(range(n_samples), leave=False, desc="Samples", colour="#006666"):
                    final_image_ddpm = inpaint_generate_new_images(
                        best_model,
                        input_image,
                        mask,
                        n_samples=1, #number of samples to generate. I think it doesn't work, not sure
                        device=device,
                        resample_steps=resample
                    )

                    standardizer = dd.get_standardizer()
                    final_image_ddpm = standardizer.unstandardize(final_image_ddpm).to(device)
                    interpolated_field = interpolate_masked_velocity_field(input_image_original[0], mask[0,0:1],).unsqueeze(0).to(device)

                    input_image_original = top_left_crop(input_image_original, 44, 94).to(device)
                    final_image_ddpm = top_left_crop(final_image_ddpm, 44, 94).to(device)
                    interpolated_field = top_left_crop(interpolated_field, 44, 94).to(device)
                    mask = top_left_crop(mask, 44, 94).to(device)

                    # Save inpainted result and mask.py
                    torch.save(final_image_ddpm,
                               f"{results_path}ddpm{batch[1].item()}_{mask_generator}_resample{resample}_num_lines_{num_lines}.pt")
                    torch.save(mask,
                               f"{results_path}mask{batch[1].item()}_{mask_generator}_resample{resample}_num_lines_{num_lines}.pt")
                    torch.save(input_image_original,
                               f"{results_path}initial{batch[1].item()}_{mask_generator}_resample{resample}_num_lines_{num_lines}.pt")
                    torch.save(interpolated_field,
                               f"{results_path}interpolated{batch[1].item()}_{mask_generator}_resample{resample}_num_lines_{num_lines}.pt")

                    # Calculate MSE for masked region
                    mse_ddpm = calculate_mse(input_image_original, final_image_ddpm, mask)
                    mse_ddpm_samples.append(mse_ddpm.item())

                    logging.info(
                        f"MSE (DDPM Inpainting) with {num_lines} lines for image {image_counter}, sample {i}: {mse_ddpm.item()}")

                    # Write result to CSV
                    output = [image_counter, num_lines, resample, mse_ddpm.item()]
                    writer.writerow(output)

                    del final_image_ddpm
                    torch.cuda.empty_cache()
                    logging.info("finished resampling")

            mean_mse_ddpm_samples = np.mean(mse_ddpm_samples)
            mse_ddpm_list.append(mean_mse_ddpm_samples)

            image_counter += 1
            main_pbar.update(1)
            logging.info("Finished processing batch:")

    return image_counter

# ======== CSV Output File for Results ========
with open(f"{results_path}inpainting_xl_data.csv", "w", newline="") as file:
    try:
        image_counter = dd.get_attribute("image_counter")
        for mask in masks_to_test:
            logging.info(f"Running next mask: {mask}")
            image_counter = inpaint_testing(mask, image_counter)
    except Exception as e:
        logging.error(f"Error during processing: {e}")
        logging.exception(f"Stack trace:")
        exit(1)

mean_mse_ddpm = np.mean(mse_ddpm_list)
logging.info(f"Mean MSE (DDPM Inpainting): {mean_mse_ddpm}")
