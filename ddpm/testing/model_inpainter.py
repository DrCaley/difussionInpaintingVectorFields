# === FULLY INTEGRATED CODE ===
import csv
import sys
import torch
import logging
import pygame
import os.path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.ndimage
from torch.utils.data import DataLoader

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
        self.mse_gp_list = []
        self.mse_ddpm_distance_list = []
        self.mse_gp_distance_list = []
        self.visualizer = False
        self.compute_coverage_plot = False
        self.model_name = "default"

        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            filename=f"{self.results_path}inpainting_model_test_log.txt")

    def set_results_path(self, results_path="."):
        self.results_path = results_path + "/results/"
        os.makedirs(self.results_path, exist_ok=True)

    def set_pixel_dimensions(self, pixel_height_m: float, pixel_width_m: float):
        self.pixel_height_m = pixel_height_m
        self.pixel_width_m = pixel_width_m

    def compute_mask_percentage(self, mask_tensor):
        cropped_mask = top_left_crop(mask_tensor, 44, 94)
        total_pixels = cropped_mask.numel()
        masked_pixels = cropped_mask.sum().item()
        return 100 * masked_pixels / total_pixels

    def compute_average_mask_distance(self, mask: torch.Tensor, pixel_height: float, pixel_width: float) -> float:
        mask_np = mask.squeeze().cpu().numpy().astype(bool)
        distance_map = scipy.ndimage.distance_transform_edt(mask_np, sampling=(pixel_height, pixel_width))
        return float(distance_map[mask_np].mean()) if mask_np.any() else 0.0

    def plot_mse_vs_mask_percentage(self):
        if not self.mse_ddpm_list:
            return
        percentages, mses = zip(*self.mse_ddpm_list)
        plt.figure()
        plt.scatter(percentages, mses, alpha=0.7, color='blue')
        plt.title(f"MSE vs. Mask Coverage (DDPM): {self.model_name}")
        plt.xlabel("Mask Coverage %")
        plt.ylabel("MSE")
        plt.grid(True)
        plt.savefig(os.path.join(self.results_path, f"mse_vs_mask_ddpm_{self.model_name}.png"))
        plt.close()

    def plot_mse_vs_mask_percentage_gp(self):
        if not self.mse_gp_list:
            return
        percentages, mses = zip(*self.mse_gp_list)
        plt.figure()
        plt.scatter(percentages, mses, alpha=0.7, color='green')
        plt.title(f"MSE vs. Mask Coverage (GP): {self.model_name}")
        plt.xlabel("Mask Coverage %")
        plt.ylabel("MSE")
        plt.grid(True)
        plt.savefig(os.path.join(self.results_path, f"mse_vs_mask_gp_{self.model_name}.png"))
        plt.close()

    def plot_mse_vs_avg_distance_ddpm(self):
        if not self.mse_ddpm_distance_list:
            return
        distances, mses = zip(*self.mse_ddpm_distance_list)
        plt.figure()
        plt.scatter(distances, mses, alpha=0.7, color='blue')
        plt.title(f"MSE vs. Avg Distance (DDPM): {self.model_name}")
        plt.xlabel("Average Distance (m)")
        plt.ylabel("MSE")
        plt.grid(True)
        plt.savefig(os.path.join(self.results_path, f"mse_vs_distance_ddpm_{self.model_name}.png"))
        plt.close()

    def plot_mse_vs_avg_distance_gp(self):
        if not self.mse_gp_distance_list:
            return
        distances, mses = zip(*self.mse_gp_distance_list)
        plt.figure()
        plt.scatter(distances, mses, alpha=0.7, color='green')
        plt.title(f"MSE vs. Avg Distance (GP): {self.model_name}")
        plt.xlabel("Average Distance (m)")
        plt.ylabel("MSE")
        plt.grid(True)
        plt.savefig(os.path.join(self.results_path, f"mse_vs_distance_gp_{self.model_name}.png"))
        plt.close()

    def export_mse_vs_mask_coverage_csv(self):
        path = os.path.join(self.results_path, f"mse_vs_mask_coverage_{self.model_name}.csv")
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Mask %", "MSE_DDPM", "MSE_GP"])
            for (pct, mse_d), (_, mse_g) in zip(self.mse_ddpm_list, self.mse_gp_list):
                writer.writerow([pct, mse_d, mse_g])

    def export_mse_vs_avg_distance_csv(self):
        path = os.path.join(self.results_path, f"mse_vs_distance_{self.model_name}.csv")
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Avg_Distance", "MSE_DDPM", "MSE_GP"])
            for (dist, mse_d), (_, mse_g) in zip(self.mse_ddpm_distance_list, self.mse_gp_distance_list):
                writer.writerow([dist, mse_d, mse_g])

    def _inpaint_testing(self, mask_generator: MaskGenerator, image_counter: int, file: csv.writer):
        writer = csv.writer(file)
        writer.writerow(["model", "image_num", "mask", "num_lines", "resample_steps", "mse", "mask_percent"])
        num_images = self.dd.get_attribute('num_images_to_process')
        loader = self.val_loader

        with tqdm(total=min(len(loader.dataset), num_images)) as main_pbar:
            for step, batch in enumerate(loader):
                if image_counter >= num_images:
                    break

                input_image = batch[0].to(self.dd.get_device())
                input_image_original = self.dd.get_standardizer().unstandardize(input_image.squeeze(0)).unsqueeze(0)
                land_mask = (input_image_original.abs() > 1e-5).float()
                mask = mask_generator.generate_mask(input_image.shape) * land_mask

                for resample in self.resamples:
                    ddpm_out = inpaint_generate_new_images(self.best_model, input_image, mask, 1, device=self.dd.get_device(), resample_steps=resample, noise_strategy=self.noise_strategy)
                    ddpm_out = self.dd.get_standardizer().unstandardize(ddpm_out.squeeze(0)).unsqueeze(0)
                    gp_out = gp_fill(input_image_original, mask)

                    input_crop = top_left_crop(input_image_original, 44, 94)
                    ddpm_crop = top_left_crop(ddpm_out, 44, 94)
                    gp_crop = top_left_crop(gp_out, 44, 94)
                    mask_crop = top_left_crop(mask, 44, 94)

                    mse_ddpm = calculate_mse(input_crop, ddpm_crop, mask_crop)
                    mse_gp = calculate_mse(input_crop, gp_crop, mask_crop)
                    pct = self.compute_mask_percentage(mask)
                    dist = self.compute_average_mask_distance(mask_crop, self.pixel_height_m, self.pixel_width_m)

                    self.mse_ddpm_list.append((pct, mse_ddpm.item()))
                    self.mse_gp_list.append((pct, mse_gp.item()))
                    self.mse_ddpm_distance_list.append((dist, mse_ddpm.item()))
                    self.mse_gp_distance_list.append((dist, mse_gp.item()))

                    writer.writerow([self.model_name, image_counter, mask_generator, mask_generator.get_num_lines(), resample, mse_ddpm.item(), pct])

                image_counter += 1
                main_pbar.update(1)
        return image_counter

    def begin_inpainting(self):
        if not self.masks_to_use:
            raise Exception("No masks added.")

        for model_path in self.model_paths:
            self.store_path = model_path
            self.model_name = os.path.splitext(os.path.basename(model_path))[0]
            self.set_results_path(f"./results/{self.model_name}/")
            self._configure_model()
            self._load_checkpoint()
            self._load_dataset()

            with open(f"{self.results_path}inpainting_xl_data.csv", 'w', newline="") as file:
                for mask in self.masks_to_use:
                    self._inpaint_testing(mask, 0, file)

            if self.compute_coverage_plot:
                self.plot_mse_vs_mask_percentage()
                self.plot_mse_vs_mask_percentage_gp()
                self.export_mse_vs_mask_coverage_csv()
                self.plot_mse_vs_avg_distance_ddpm()
                self.plot_mse_vs_avg_distance_gp()
                self.export_mse_vs_avg_distance_csv()

    def add_model(self, path):
        if os.path.exists(path):
            self.model_paths.append(path)

    def add_models(self, paths):
        for p in paths:
            self.add_model(p)

    def visualize_images(self, vector_scale=0.15):
        self.visualizer = True
        self.vector_scale = vector_scale

    def find_coverage(self):
        self.compute_coverage_plot = True

    def load_models_from_yaml(self):
        self.add_models(self.dd.get_attribute('model_paths'))

# === USAGE ===
if __name__ == '__main__':
    mi = ModelInpainter()
    mi.load_models_from_yaml()
    mi.add_mask(CoverageMaskGenerator(0.8))
    mi.visualize_images()
    mi.find_coverage()
    mi.set_pixel_dimensions(1.0, 1.0)
    mi.begin_inpainting()
