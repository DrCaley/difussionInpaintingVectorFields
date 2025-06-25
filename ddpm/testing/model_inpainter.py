import csv
import sys
import torch
import logging
import os.path
import numpy as np
from tensorflow.python.ops.numpy_ops.np_math_ops import linspace
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from ddpm.helper_functions.masks.n_coverage_mask import CoverageMaskGenerator
from ddpm.helper_functions.masks.random_mask import RandomMaskGenerator

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from ddpm.helper_functions.masks import MaskGenerator
from ddpm.helper_functions.masks import *
from data_prep.data_initializer import DDInitializer
from ddpm.neural_networks.ddpm import MyDDPMGaussian
from ddpm.neural_networks.unets.unet_xl import MyUNet
from plots.visualization_tools.pt_visualizer_plus import PTVisualizer
from ddpm.helper_functions.interpolation_tool import interpolate_masked_velocity_field, gp_fill
from ddpm.utils.inpainting_utils import inpaint_generate_new_images, calculate_mse, top_left_crop

os.chdir(CURRENT_DIR)



class ModelInpainter:
    def __init__(self):
        self.dd = DDInitializer()
        self.set_results_path()
        self.model_paths = []
        self.masks_to_use = []
        self.resamples = self.dd.get_attribute("resample_nums")
        self.mse_ddpm_list = []
        self.visualizer = False
        self.compute_coverage_plot = False
        self.model_name = "default"

        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            filename=f"{self.results_path}inpainting_model_test_log.txt")

    def set_results_path(self, results_path="."):
        self.results_path = results_path + "/results/"
        os.makedirs(self.results_path, exist_ok=True)

    def add_model(self, model_path : str):
        if not os.path.exists(model_path):
            print(f"Warning: {model_path} does not exist and will be skipped.")
            return
        self.model_paths.append(model_path)

    def add_models(self, model_path_list : list):
        for path in model_path_list:
            self.add_model(path)

    def _configure_model(self):
        checkpoint = torch.load(self.store_path, map_location=self.dd.get_device(), weights_only=False)
        self.model_state_dict = checkpoint.get('model_state_dict', checkpoint)

        self.n_steps = checkpoint.get('n_steps', self.dd.get_attribute("noise_steps"))
        self.min_beta = checkpoint.get('min_beta', self.dd.get_attribute("min_beta"))
        self.max_beta = checkpoint.get('max_beta', self.dd.get_attribute("max_beta"))

        self.noise_strategy = checkpoint.get('noise_strategy', self.dd.get_noise_strategy())
        self.standardizer_strategy = checkpoint.get('standardizer_strategy', self.dd.get_standardizer())

        self.dd.reinitialize(self.min_beta, self.max_beta, self.n_steps, self.standardizer_strategy)

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
        try:
            logging.info("Preparing data")
            batch_size = self.dd.get_attribute("inpainting_batch_size")
            self.train_loader = DataLoader(self.dd.get_training_data(), batch_size=batch_size, shuffle=True)
            self.test_loader = DataLoader(self.dd.get_test_data(), batch_size=batch_size)
            self.val_loader = DataLoader(self.dd.get_validation_data(), batch_size=batch_size)
            logging.info("Data prepared successfully")
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            exit(1)

    def add_mask(self, mask: MaskGenerator):
        self.masks_to_use.append(mask)

    def compute_mask_percentage(self, mask_tensor):
        cropped_mask = top_left_crop(mask_tensor, 44, 94)
        total_pixels = cropped_mask.numel()
        masked_pixels = cropped_mask.sum().item()
        return 100 * masked_pixels / total_pixels

    def plot_mse_vs_mask_percentage(self):
        if not self.mse_ddpm_list:
            logging.warning("No MSE data to plot.")
            return

        percentages, mses = zip(*self.mse_ddpm_list)
        plt.figure(figsize=(8, 5))
        plt.scatter(percentages, mses, alpha=0.7, color='blue')
        plt.title(f"MSE vs. Mask Coverage Percentage (Model: {self.model_name})")
        plt.xlabel("Percentage of Masked Pixels")
        plt.ylabel("MSE")
        plt.grid(True)
        plt.tight_layout()
        plot_path = os.path.join(self.results_path, f"mse_vs_mask_percentage_{self.model_name}.png")
        plt.savefig(plot_path)
        plt.close()
        logging.info(f"Saved MSE vs. mask percentage plot to {plot_path}")

    def _inpaint_testing(self, mask_generator: MaskGenerator, image_counter: int, file: csv.writer):
        writer = csv.writer(file)
        writer.writerow(["model", "image_num", "mask", "num_lines", "resample_steps", "mse", "mask_percent"])

        num_images_to_process = self.dd.get_attribute('num_images_to_process')
        n_samples = self.dd.get_attribute('n_samples')
        loader = self.val_loader

        with tqdm(total=min(len(loader.dataset), num_images_to_process),
                  desc=f"[{self.model_name}] Mask: {mask_generator}", colour="#00ffff") as main_pbar:

            for step, batch in enumerate(loader):
                if image_counter >= num_images_to_process:
                    break

                device = self.dd.get_device()
                input_image = batch[0].to(device)
                input_image_original = self.dd.get_standardizer().unstandardize(input_image).to(device)
                land_mask = (input_image_original != 0).float().to(device)

                raw_mask = mask_generator.generate_mask(input_image.shape)
                mask = raw_mask * land_mask
                num_lines = mask_generator.get_num_lines()

                for resample in self.resamples:
                    for i in tqdm(range(n_samples), leave=False, desc="Samples", colour="#006666"):
                        final_image_ddpm = inpaint_generate_new_images(
                            self.best_model, input_image, mask, n_samples=1,
                            device=device, resample_steps=resample, noise_strategy=self.noise_strategy
                        )

                        standardizer = self.dd.get_standardizer()
                        final_image_ddpm = standardizer.unstandardize(final_image_ddpm).to(device)
                        gp_field = gp_fill(input_image_original, mask)

                        # Cropping
                        input_image_original_cropped = top_left_crop(input_image_original, 44, 94).to(device)
                        final_image_ddpm_cropped = top_left_crop(final_image_ddpm, 44, 94).to(device)
                        mask_cropped = top_left_crop(mask, 44, 94).to(device)
                        gp_field_cropped = top_left_crop(gp_field, 44, 94).to(device)

                        # MSE + Coverage
                        mse_ddpm = calculate_mse(input_image_original_cropped, final_image_ddpm_cropped, mask_cropped)
                        mask_percentage = self.compute_mask_percentage(mask)

                        base_id = f"{batch[1].item()}_{mask_generator}_resample{resample}_num_lines_{num_lines}"
                        torch.save(final_image_ddpm_cropped, f"{self.results_path}ddpm{base_id}.pt")
                        torch.save(mask_cropped, f"{self.results_path}mask{base_id}.pt")
                        torch.save(input_image_original_cropped, f"{self.results_path}initial{base_id}.pt")
                        torch.save(gp_field_cropped, f"{self.results_path}gp_field{base_id}.pt")

                        writer.writerow([self.model_name, image_counter, mask_generator, num_lines, resample, mse_ddpm.item(), mask_percentage])

                        if self.compute_coverage_plot:
                            self.mse_ddpm_list.append((mask_percentage, mse_ddpm.item()))

                        if self.visualizer:
                            ptv = PTVisualizer(mask_type=mask_generator, sample_num=batch[1].item(),
                                               vector_scale=self.vector_scale, num_lines=num_lines, resamples=resample, results_dir = self.results_path)
                            ptv.visualize()
                            ptv.calc()

                        torch.cuda.empty_cache()

                image_counter += 1
                main_pbar.update(1)

        return image_counter

    def begin_inpainting(self):
        if len(self.masks_to_use) == 0:
            raise Exception('No masks available! Use `add_mask(...)` before running.')

        for model_path in self.model_paths:
            try:
                self.store_path = model_path
                self.model_name = os.path.splitext(os.path.basename(model_path))[0]
                self.set_results_path(f"./results/{self.model_name}/")

                self._configure_model()
                self._load_checkpoint()
                self._load_dataset()

                with open(f"{self.results_path}inpainting_xl_data.csv", 'w', newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(
                        ["model", "image_num", "mask", "num_lines", "resample_steps", "mse", "mask_percent"])
                    for mask in self.masks_to_use:
                        image_counter = 0
                        logging.info(f"Running mask {mask} with model {self.model_name}")
                        image_counter = self._inpaint_testing(mask, image_counter, file)

                if self.compute_coverage_plot:
                    self.plot_mse_vs_mask_percentage()

            except Exception as e:
                logging.error(f"Error processing model {model_path}: {e}")
                continue

    def visualize_images(self, vector_scale=0.15):
        self.visualizer = True
        self.vector_scale = vector_scale

    def find_coverage(self):
        self.compute_coverage_plot = True


# === USAGE EXAMPLE ===
if __name__ == '__main__':
    mi = ModelInpainter()

    # mi.add_model("../trained_models/ddpm_ocean_model_0624.pt")
    # mi.add_model("../trained_models/ddpm_ocean_model_best_checkpoint.pt")
    mi.add_model("../trained_models/weekend_ddpm_ocean_model.pt")

    for val in linspace(0,1,110):
        mi.add_mask(CoverageMaskGenerator(val))

    mi.visualize_images()
    mi.find_coverage()
    mi.begin_inpainting()
