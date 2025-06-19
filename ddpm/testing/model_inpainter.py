import os.path
import numpy as np
import torch
from torch.utils.data import DataLoader
import logging
import csv
import sys

from tqdm import tqdm

from ddpm.helper_functions.masks.n_coverage_mask import CoverageMaskGenerator

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from ddpm.helper_functions.masks.robot_path import RobotPathMaskGenerator
from ddpm.helper_functions.interpolation_tool import interpolate_masked_velocity_field, gp_fill
# https://genius.com/22643703/Dream-mask/Thats-what-the-mask-is-thats-what-the-point-of-the-mask-is
from ddpm.helper_functions.masks.better_robot_path import BetterRobotPathGenerator
from ddpm.helper_functions.masks.abstract_mask import MaskGenerator
from ddpm.helper_functions.masks.straigth_line import StraightLineMaskGenerator
from ddpm.helper_functions.masks.gaussian_mask import GaussianNoiseBinaryMaskGenerator
from ddpm.helper_functions.masks.squiggly_line import SquigglyLineMaskGenerator
from ddpm.helper_functions.masks.mask_drawer import ManualMaskDrawer
from data_prep.data_initializer import DDInitializer
from ddpm.neural_networks.ddpm import MyDDPMGaussian
from ddpm.utils.inpainting_utils import inpaint_generate_new_images, calculate_mse, top_left_crop
from ddpm.neural_networks.unets.unet_xl import MyUNet


class ModelInpainter:
    def __init__(self):
        self.dd = DDInitializer()
        self.set_results_path()
        self.use_this_model()
        self.masks_to_use = []
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename=f"{self.results_path}inpainting_model_test_log.txt")
        self.resample_nums = self.dd.get_attribute("resample_nums")
        self.mse_ddpm_list = []
        self._configure_model()
        self._load_checkpoint()
        self._load_dataset()

    def set_results_path(self, results_path = "./results/"):
        self.results_path = results_path
        os.makedirs(results_path, exist_ok=True)

    def use_this_model(self, store_path=None):
        self.store_path = self.dd.get_attribute("model_path")
        if store_path is None:
            self.store_path = self.dd.get_attribute("model_path")
        else :
            if os.path.exists(store_path):
                self.store_path = store_path
            else :
                print(f"{store_path} does not exist, using {self.store_path}")

    def _configure_model(self):
        dd = self.dd
        checkpoint = torch.load(self.store_path, map_location=self.dd.get_device(), weights_only=False)
        self.model_state_dict = checkpoint.get('model_state_dict', checkpoint)

        self.n_steps = checkpoint.get('n_steps', dd.get_attribute("n_steps"))
        self.min_beta = checkpoint.get('min_beta', dd.get_attribute("min_beta"))
        self.max_beta = checkpoint.get('max_beta', dd.get_attribute("max_beta"))
        self.noise_strategy = checkpoint.get('noise_strategy', dd.get_noise_strategy())

    def _load_checkpoint(self):
        self.best_model = MyDDPMGaussian(MyUNet(self.n_steps), n_steps=self.n_steps, device=self.dd.get_device())
        try:
            logging.info("Loading model")
            self.best_model.load_state_dict(self.model_state_dict)
            self.best_model.eval()
            logging.info("Model loaded successfully")
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            exit(1)

    def _load_dataset(self):
        dd = self.dd
        try:
            logging.info("Preparing data")
            batch_size = dd.get_attribute("inpainting_batch_size")
            training_data = dd.get_training_data()
            test_data = dd.get_test_data()
            validation_data = dd.get_validation_data()

            self.train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
            self.test_loader = DataLoader(test_data, batch_size=batch_size)
            self.val_loader = DataLoader(validation_data, batch_size=batch_size)
            logging.info("Data prepared successfully")
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            exit(1)

    def add_mask(self, mask : MaskGenerator):
        self.masks_to_use.append(mask)

    def _inpaint_testing(self, mask_generator : MaskGenerator, image_counter : int, file : str):
        dd = self.dd
        writer = csv.writer(file)
        header = ["image_num", "mask", "num_lines", "resample_steps", "mse"]
        writer.writerow(header)

        logging.info("Processing data")
        num_images_to_process = dd.get_attribute('num_images_to_process')
        n_samples = dd.get_attribute('n_samples')

        loader = self.val_loader
        total_images = min(len(loader.dataset), num_images_to_process)
        with tqdm(total=total_images, desc=f"Mask: {mask_generator}", colour="#00ffff") as main_pbar:
            for step, batch in enumerate(loader):
                logging.info("Processing batch")
                if image_counter >= num_images_to_process:
                    break

                device = dd.get_device()
                input_image = batch[0].to(device)  # (Batch size, Channels, Height, Width)

                # Convert back to unstandardized form for land masking
                input_image_original = dd.get_standardizer().unstandardize(input_image).to(device)
                land_mask = (input_image_original != 0).float().to(device)

                mask = mask_generator.generate_mask(input_image.shape, land_mask)
                num_lines = mask_generator.get_num_lines()

                # ======== Masking and Inpainting Loops ========
                for resample in self.resample_nums:

                    torch.save(mask, f"{self.results_path}{mask_generator}_{num_lines}.pt")

                    mse_ddpm_samples = []
                    # ======== Generate Samples ========
                    for i in tqdm(range(n_samples), leave=False, desc="Samples", colour="#006666"):
                        final_image_ddpm = inpaint_generate_new_images(
                            self.best_model,
                            input_image,
                            mask,
                            n_samples=1,  # number of samples to generate. I think it doesn't work, not sure
                            device=device,
                            resample_steps=resample,
                            noise_strategy=self.noise_strategy
                            )

                        standardizer = dd.get_standardizer()
                        final_image_ddpm = standardizer.unstandardize(final_image_ddpm).to(device)
                        gp_field = gp_fill(input_image_original, mask)

                        input_image_original_cropped = top_left_crop(input_image_original, 44, 94).to(device)
                        final_image_ddpm_cropped = top_left_crop(final_image_ddpm, 44, 94).to(device)
                        mask_cropped = top_left_crop(mask, 44, 94).to(device)
                        gp_field_cropped = top_left_crop(gp_field, 44, 94).to(device)

                        # Save inpainted result and mask.py
                        torch.save(final_image_ddpm_cropped,
                                   f"{self.results_path}ddpm{batch[1].item()}_{mask_generator}_resample{resample}_num_lines_{num_lines}.pt")
                        torch.save(mask_cropped,
                                   f"{self.results_path}mask{batch[1].item()}_{mask_generator}_resample{resample}_num_lines_{num_lines}.pt")
                        torch.save(input_image_original_cropped,
                                   f"{self.results_path}initial{batch[1].item()}_{mask_generator}_resample{resample}_num_lines_{num_lines}.pt")
                        torch.save(gp_field_cropped,
                                   f"{self.results_path}gp_field{batch[1].item()}_{mask_generator}_resample{resample}_num_lines_{num_lines}.pt")

                        # Calculate MSE for masked region
                        mse_ddpm = calculate_mse(input_image_original_cropped, final_image_ddpm_cropped, mask_cropped)
                        mse_ddpm_samples.append(mse_ddpm.item())

                        logging.info(
                            f"MSE (DDPM Inpainting) with {num_lines} lines for image {image_counter}, sample {i}: {mse_ddpm.item()}")

                        # Write result to CSV
                        output = [image_counter, mask_generator, num_lines, resample, mse_ddpm.item()]
                        writer.writerow(output)

                        del final_image_ddpm
                        torch.cuda.empty_cache()
                        logging.info("finished resampling")

                mean_mse_ddpm_samples = np.mean(mse_ddpm_samples)
                self.mse_ddpm_list.append(mean_mse_ddpm_samples)

                image_counter += 1
                main_pbar.update(1)
                logging.info("Finished processing batch:")

        return image_counter

    def begin_inpainting(self):
        if len(self.masks_to_use) == 0:
            raise Exception('No masks available! make sure to add some with:'
                            '\n mi = ModelInpainter()'
                            '\n mi.add_mask(MaskGenerator())')

        with open(f"{self.results_path}inpainting_xl_data.csv", 'w', newline="") as file:
            try:
                for mask in self.masks_to_use:
                    image_counter = 0
                    logging.info(f"Running next mask: {mask}")
                    self._inpaint_testing(mask, image_counter, file)
            except Exception as e:
                logging.error(f"Error during processing: {e}")
                exit(1)
            mean_mse_ddpm = np.mean(self.mse_ddpm_list)
            logging.info(f"Mean MSE (DDPM Inpainting): {mean_mse_ddpm}")

if __name__ == '__main__':
    mi = ModelInpainter()
    mi.add_mask(CoverageMaskGenerator(coverage_ratio=0.25))
    mi.begin_inpainting()