import csv
import sys

import torch
import logging
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from scipy.ndimage import distance_transform_edt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR.parent.parent))

from ddpm.helper_functions.masks import MaskGenerator
from ddpm.helper_functions.masks import *
from data_prep.data_initializer import DDInitializer
from ddpm.neural_networks.ddpm import GaussianDDPM
from ddpm.neural_networks.unets.unet_xl import MyUNet
from plots.visualization_tools.pt_visualizer_plus import PTVisualizer
from ddpm.helper_functions.interpolation_tool import interpolate_masked_velocity_field, gp_fill
from ddpm.utils.inpainting_utils import inpaint_generate_new_images, calculate_mse, top_left_crop


class ModelInpainter:
    def __init__(self, config_path = None, model_file = None):
        if config_path is None:
            self.dd = DDInitializer()
        else:
            self.dd = DDInitializer(config_path=config_path)
        self.set_results_path("./results")
        self.csv_file = self.results_path / "inpainting_xl_data.csv"
        self.write_header()
        self.model_paths = []
        if model_file is not None:
            self.model_paths.append(model_file)

        self.masks_to_use = []
        self.resamples = self.dd.get_attribute("resample_nums")
        self.reset_plot_lists()
        self.pixel_height = 1.0
        self.pixel_width = 1.0
        self.visualizer = False
        self.compute_coverage_plot = False
        self.save_pt_fields = self.dd.get_attribute("save_pt_fields")
        self.model_name = None

        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            filename=self.results_path / "inpainting_model_test_log.txt")

    def set_results_path(self, results_path="."):
        self.results_path = Path(results_path)
        self.results_path.mkdir(exist_ok=True)

    def reset_plot_lists(self):
        self.mse_ddpm_list = []
        self.mse_gp_list = []
        self.mse_distance_ddpm = []
        self.mse_distance_gp = []

    def set_pixel_dimensions(self, pixel_height, pixel_width):
        self.pixel_height = pixel_height
        self.pixel_width = pixel_width

    def save_pt_files(self):
        self.save_pt_fields = True

    def add_model(self, model_path: str):
        path = Path(model_path)
        if not path.exists():
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
        self.best_model = GaussianDDPM(MyUNet(self.n_steps), n_steps=self.n_steps, device=self.dd.get_device())
        try:
            logging.info("Loading model")
            self.best_model.load_state_dict(self.model_state_dict)
            self.best_model.eval()
            logging.info("Model loaded successfully")
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise Exception(f"{e}")

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
            raise Exception(f"{e}")

    def add_mask(self, mask: MaskGenerator):
        self.masks_to_use.append(mask)

    def compute_mask_percentage(self, mask_tensor):
        cropped_mask = top_left_crop(mask_tensor, 44, 94)
        total_pixels = cropped_mask.numel()
        masked_pixels = cropped_mask.sum().item()
        return 100 * masked_pixels / total_pixels

    def compute_avg_distance_to_seen(self, mask_tensor):
        cropped_mask = top_left_crop(mask_tensor, 44, 94).cpu()

        if cropped_mask.ndim == 4:
            cropped_mask = cropped_mask[0, 0]  # From shape [1, 2, H, W] to [H, W]

        cropped_mask = cropped_mask.numpy()
        distances = distance_transform_edt(cropped_mask, sampling=[self.pixel_height, self.pixel_width])
        return float(np.sum(distances * cropped_mask) / np.sum(cropped_mask))

    def plot_mse_vs_mask_percentage(self):
        if not self.mse_ddpm_list:
            logging.warning("No DDPM MSE data to plot.")
            return
        percentages, mses = zip(*self.mse_ddpm_list)
        plt.figure(figsize=(8, 5))
        plt.scatter(percentages, mses, alpha=0.7, color='blue')
        plt.title(f"MSE vs. Mask Coverage Percentage (Model: {self.model_name})")
        plt.xlabel("Percentage of Model Predicts")
        plt.ylabel("MSE (DDPM)")
        plt.grid(True)
        plt.tight_layout()
        plot_path = self.results_path / f"mse_vs_mask_percentage_{self.model_name}.png"
        plt.savefig(plot_path)
        plt.close()
        logging.info(f"Saved DDPM MSE vs. mask percentage plot to {plot_path}")

    def plot_mse_vs_mask_percentage_gp(self):
        if not self.mse_gp_list:
            logging.warning("No GP MSE data to plot.")
            return
        percentages, mses = zip(*self.mse_gp_list)
        plt.figure(figsize=(8, 5))
        plt.scatter(percentages, mses, alpha=0.7, color='green')
        plt.title(f"GP Fill MSE vs. Mask Coverage Percentage (Model: {self.model_name})")
        plt.xlabel("Percentage of Model Predicts")
        plt.ylabel("MSE (GP Fill)")
        plt.grid(True)
        plt.tight_layout()
        plot_path = self.results_path / f"gp_mse_vs_mask_percentage_{self.model_name}.png"
        plt.savefig(plot_path)
        plt.close()
        logging.info(f"Saved GP Fill MSE vs. mask percentage plot to {plot_path}")

    def plot_mse_vs_distance(self):
        if not self.mse_distance_ddpm:
            return
        x_ddpm, y_ddpm = zip(*self.mse_distance_ddpm)
        x_gp, y_gp = zip(*self.mse_distance_gp)

        plt.figure(figsize=(8, 5))
        plt.scatter(x_ddpm, y_ddpm, color='blue', alpha=0.6, label="DDPM")
        plt.scatter(x_gp, y_gp, color='green', alpha=0.6, label="GP Fill")
        plt.xlabel("Average Distance to Seen Pixel")
        plt.ylabel("MSE")
        plt.title(f"MSE vs. Avg Distance to Seen Pixel (Model: {self.model_name})")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        path = self.results_path / f"mse_vs_distance_{self.model_name}.png"
        plt.savefig(path)
        plt.close()
        logging.info(f"Saved MSE vs. distance plot to {path}")

    def _inpaint_testing(self, mask_generator: MaskGenerator, image_counter: int, file: csv.writer):
        writer = csv.writer(file)
        num_images_to_process = self.dd.get_attribute('num_images_to_process')
        n_samples = self.dd.get_attribute('n_samples')
        loader = self.val_loader

        with tqdm(total=min(len(loader.dataset), num_images_to_process),
                  desc=f"[{self.model_name}] Mask: {mask_generator}({mask_generator.get_num_lines()})", colour="#00ffff") as main_pbar:

            for step, batch in enumerate(loader):
                if image_counter >= num_images_to_process:
                    break

                device = self.dd.get_device()
                input_image = batch[0].to(device)
                input_image_original = self.dd.get_standardizer().unstandardize(torch.squeeze(input_image, 0)).to(device)
                input_image_original = torch.unsqueeze(input_image_original, 0)
                land_mask = (input_image_original.abs() > 1e-5).float().to(device)
                raw_mask = mask_generator.generate_mask(input_image.shape)
                mask = raw_mask * land_mask
                num_lines = mask_generator.get_num_lines()

                with torch.no_grad():
                    for resample in self.resamples:
                        for i in tqdm(range(n_samples), leave=False, desc="Samples", colour="#006666"):
                            final_image_ddpm = inpaint_generate_new_images(
                                self.best_model, input_image, mask, n_samples=1,
                                device=device, resample_steps=resample, noise_strategy=self.noise_strategy
                            )

                            standardizer = self.dd.get_standardizer()
                            final_image_ddpm = torch.unsqueeze(standardizer.unstandardize(torch.squeeze(final_image_ddpm, 0)).to(device), 0)
                            gp_field = gp_fill(input_image_original, mask)

                            input_image_original_cropped = top_left_crop(input_image_original, 44, 94).to(device)
                            final_image_ddpm_cropped = top_left_crop(final_image_ddpm, 44, 94).to(device)
                            mask_cropped = top_left_crop(mask, 44, 94).to(device)
                            gp_field_cropped = top_left_crop(gp_field, 44, 94).to(device)

                            mse_ddpm = calculate_mse(input_image_original_cropped, final_image_ddpm_cropped, mask_cropped, normalize=True)
                            mse_gp = calculate_mse(input_image_original_cropped, gp_field_cropped, mask_cropped, normalize=True)
                            mask_percentage = self.compute_mask_percentage(mask)
                            avg_dist = self.compute_avg_distance_to_seen(mask_cropped)

                            base_id = f"{batch[1].item()}_{mask_generator}_resample{resample}_num_lines_{num_lines}"

                            torch.save(final_image_ddpm_cropped, self.results_path / f"ddpm{base_id}.pt")
                            torch.save(mask_cropped, self.results_path / f"mask{base_id}.pt")
                            torch.save(input_image_original_cropped, self.results_path / f"initial{base_id}.pt")
                            torch.save(gp_field_cropped, self.results_path / f"gp_field{base_id}.pt")


                            writer.writerow([self.model_name, image_counter, mask_generator, num_lines, resample, mse_ddpm.item(), mse_gp.item(), mask_percentage, avg_dist])

                            if self.compute_coverage_plot:
                                self.mse_ddpm_list.append((mask_percentage, mse_ddpm.item()))
                                self.mse_gp_list.append((mask_percentage, mse_gp.item()))
                                self.mse_distance_ddpm.append((avg_dist, mse_ddpm.item()))
                                self.mse_distance_gp.append((avg_dist, mse_gp.item()))

                            if self.visualizer:
                                ptv = PTVisualizer(mask_type=mask_generator, sample_num=batch[1].item(),
                                                   vector_scale=self.vector_scale, num_lines=num_lines, resamples=resample, results_dir=self.results_path)
                                ptv.visualize()
                                ptv.calc()

                            if not self.save_pt_fields:
                                (self.results_path / f"ddpm{base_id}.pt").unlink()
                                (self.results_path / f"mask{base_id}.pt").unlink()
                                (self.results_path / f"initial{base_id}.pt").unlink()
                                (self.results_path / f"gp_field{base_id}.pt").unlink()

                            del final_image_ddpm, input_image_original_cropped, mask_cropped, gp_field

                    del input_image, input_image_original, land_mask, mask
                    torch.cuda.empty_cache()

                image_counter += 1
                main_pbar.update(1)

        return image_counter

    def write_header(self):
        with open(self.csv_file, 'w', newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["model", "image_num", "mask", "num_lines", "resample_steps", "ddp_mse", "gp_mse", "mask_percent", "average_pixel_distance"])

    def _set_up_model(self, model_path):
        self.store_path = Path(model_path)
        if self.model_name is None:
            self.model_name = self.store_path.stem
        self.set_results_path(f"./results/{self.model_name}")

        try:
            import yaml
            config_path = self.results_path / "config.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(self.dd.get_full_config(), f)
            logging.info(f"Saved config to {config_path}")
        except Exception as e:
            logging.warning(f"Failed to save config: {e}")

        self._configure_model()
        self._load_checkpoint()
        self._load_dataset()

    def begin_inpainting(self):
        if len(self.masks_to_use) == 0:
            raise Exception('No masks available! Use `add_mask(...)` before running.')

        model_bar = tqdm(self.model_paths, desc="🧠 Models", colour="magenta")

        for model_path in model_bar:
            try:
                self._set_up_model(model_path)

                with open(self.csv_file, 'a', newline="") as file:
                    mask_bar = tqdm(self.masks_to_use, desc=f"🎭 Masks ({self.model_name})", leave=False, colour="cyan")
                    for mask in mask_bar:
                        mask_bar.set_postfix(model=self.model_name, mask=str(mask))
                        logging.info(f"Running mask {mask} with model {self.model_name}")
                        image_counter = self._inpaint_testing(mask, 0, file)

                if self.compute_coverage_plot:
                    self.plot_mse_vs_mask_percentage()
                    self.plot_mse_vs_mask_percentage_gp()
                    self.plot_mse_vs_distance()

                self.reset_plot_lists()

            except Exception as e:
                logging.error(f"Error inpainting model {model_path}: {e}", stack_info=True)
                continue

    def visualize_images(self, vector_scale=0.15):
        self.visualizer = True
        self.vector_scale = vector_scale

    def find_coverage(self):
        self.compute_coverage_plot = True

    def load_models_from_yaml(self):
        models = self.dd.get_attribute('model_paths')
        print("adding models from data.yaml")
        self.add_models(models)
        if len(self.model_paths) == 0:
            print("no models in model_paths attribute in data.yaml")

    def set_model_name(self, model_name):
        self.model_name = model_name

# === USAGE EXAMPLE ===
if __name__ == '__main__':
    mi = ModelInpainter()
    mi.load_models_from_yaml()

    mi.add_mask(ManualMaskDrawer())

    mi.visualize_images()
    mi.find_coverage()
    mi.begin_inpainting()