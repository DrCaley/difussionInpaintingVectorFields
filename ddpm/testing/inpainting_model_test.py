import os.path
import numpy as np
import torch
from torch.utils.data import DataLoader
import logging
import csv
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from ddpm.helper_functions.mask_factory.masks.abstract_mask import MaskGenerator
from ddpm.helper_functions.mask_factory.masks.random_path import RandomPathMaskGenerator
from data_prep.data_initializer import DDInitializer
from ddpm.neural_networks.ddpm import MyDDPMGaussian
from ddpm.helper_functions.inpainting_utils import inpaint_generate_new_images, calculate_mse
from ddpm.neural_networks.unets.unet_xl import MyUNet

dd = DDInitializer()

(training_tensor, validation_tensor, test_tensor) = dd.get_tensors()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename="inpainting_model_test_log.txt")

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
line_numbers = [10, 20, 40]          # Number of lines in the abstract_mask.py
resample_nums = [5]                  # Number of resampling steps
mse_ddpm_list = []                   # To store average MSEs per image

# =========== Initializing Masks ==================
masks_to_test = []
for line in line_numbers:
    random_mask_thin = RandomPathMaskGenerator(num_lines=line, line_thickness=1, line_length=5)
    random_mask_thick = RandomPathMaskGenerator(num_lines=line, line_thickness=5, line_length=5)

    masks_to_test.append(random_mask_thin)
    masks_to_test.append(random_mask_thick)

def inpaint_testing(mask_generator: MaskGenerator, image_counter: int) -> int:
    writer = csv.writer(file)
    header = ["image_num", "num_lines", "resample_steps", "mse"]
    writer.writerow(header)

    logging.info("Processing data")
    num_images_to_process = dd.get_attribute('num_images_to_process')
    n_samples = dd.get_attribute('n_samples')

    # ======== Loop Through Batches ========
    batch_num = 1
    loader = train_loader

    for batch in loader:
        logging.info("Processing batch:", batch_num)
        if image_counter >= num_images_to_process:
            break

        input_image = batch[0].to(dd.get_device()) # (Batch size, Channels, Height, Width)

        # Convert back to unstandardized form for land masking
        input_image_original = dd.get_standardizer().unstandardize(input_image)
        land_mask = (input_image_original != 0).float()

        mask = mask_generator.generate_mask(input_image.shape, land_mask)
        num_lines = mask_generator.num_lines

        mask = mask.to(dd.get_device())

        # ======== Masking and Inpainting Loops ========
        for resample in resample_nums:

            torch.save(mask, f"results/predicted/{mask_generator}_{num_lines}.pt")

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

                # Save inpainted result and mask.py
                torch.save(final_image_ddpm,
                           f"results/predicted/img{batch[1].item()}_{mask_generator}_resample{resample}_num_lines_{num_lines}.pt")
                torch.save(mask_generator,
                           f"results/predicted/mask{batch[1].item()}_{mask_generator}_resample{resample}_num_lines_{num_lines}.pt")

                # Calculate MSE for masked region
                mse_ddpm = calculate_mse(input_image, final_image_ddpm, mask)
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
        logging.info("Finished processing batch:", batch_num)
        batch_num += 1


    return image_counter

# ======== CSV Output File for Results ========
with open("inpainting-xl-data.csv", "w", newline="") as file:
    try:
        image_counter = dd.get_attribute("image_counter")
        for mask in masks_to_test:
            image_counter = inpaint_testing(mask, image_counter)
    except Exception as e:
        logging.error(f"Error during processing: {e}")
        exit(1)

mean_mse_ddpm = np.mean(mse_ddpm_list)
logging.info(f"Mean MSE (DDPM Inpainting): {mean_mse_ddpm}")
