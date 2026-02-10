import csv
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from data_prep.data_initializer import DDInitializer
from ddpm.Testing.model_inpainter import ModelInpainter
from ddpm.helper_functions.masks.straight_line_path import StraightLinePathGenerator
from ddpm.helper_functions.interpolation_tool import gp_fill
from ddpm.utils.inpainting_utils import inpaint_generate_new_images, top_left_crop
from scripts.run_comparison import compute_all_metrics


def main(num_runs=10, seed=0, line_width=1):
    random.seed(seed)
    torch.manual_seed(seed)

    dd = DDInitializer()
    device = dd.get_device()

    mi = ModelInpainter()
    mi.load_models_from_yaml()
    if not mi.model_paths:
        raise RuntimeError("No model paths loaded.")
    mi._set_up_model(mi.model_paths[0])

    loader = DataLoader(
        dd.get_validation_data(),
        batch_size=1,
        shuffle=True,
        num_workers=dd.get_attribute("num_workers") or 0,
    )
    iterator = iter(loader)

    resample_nums = dd.get_attribute("resample_nums") or [5]
    resample_steps = resample_nums[0]

    gp_lengthscale = dd.get_attribute("gp_lengthscale") or 1.5
    gp_variance = dd.get_attribute("gp_variance") or 1.0
    gp_noise = dd.get_attribute("gp_noise") or 1e-5
    gp_kernel_type = dd.get_attribute("gp_kernel_type") or "rbf"
    gp_coord_system = dd.get_attribute("gp_coord_system") or "pixels"
    gp_use_double = dd.get_attribute("gp_use_double")
    gp_use_double = True if gp_use_double is None else gp_use_double
    gp_max_points = dd.get_attribute("gp_max_points")
    if gp_max_points is not None:
        gp_max_points = int(gp_max_points)
    gp_sample_posterior = dd.get_attribute("gp_sample_posterior") or False

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    csv_path = results_dir / "ddpm_vs_gp_straight_line_bulk.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "run_idx",
            "sample_id",
            "mask_type",
            "mask_percent",
            "ddpm_mse",
            "gp_mse",
            "ddpm_percent_error",
            "gp_percent_error",
            "ddpm_angular_error",
            "gp_angular_error",
            "ddpm_scaled_error_mag",
            "gp_scaled_error_mag",
            "ddpm_norm_mag_diff",
            "gp_norm_mag_diff",
            "ddpm_minus_gp_mse",
            "ddpm_minus_gp_percent_error",
            "ddpm_minus_gp_angular_error",
            "ddpm_minus_gp_scaled_error_mag",
            "ddpm_minus_gp_norm_mag_diff",
        ])

        for run_idx in range(num_runs):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)

            input_image = batch[0].to(device)
            sample_id = int(batch[1].item()) if len(batch) > 1 else -1

            input_image_original = dd.get_standardizer().unstandardize(torch.squeeze(input_image, 0)).to(device)
            input_image_original = torch.unsqueeze(input_image_original, 0)

            land_mask = (input_image_original.abs() > 1e-5).float().to(device)
            raw_mask = StraightLinePathGenerator(line_width=line_width).generate_mask(input_image.shape).to(device)
            missing_mask = raw_mask * land_mask

            ddpm_output = inpaint_generate_new_images(
                mi.best_model,
                input_image,
                missing_mask,
                n_samples=1,
                device=device,
                resample_steps=resample_steps,
                noise_strategy=dd.get_noise_strategy(),
            )
            ddpm_output = torch.unsqueeze(
                dd.get_standardizer().unstandardize(torch.squeeze(ddpm_output, 0)).to(device),
                0,
            )

            gp_output = gp_fill(
                input_image_original,
                missing_mask,
                lengthscale=gp_lengthscale,
                variance=gp_variance,
                noise=gp_noise,
                use_double=gp_use_double,
                max_points=gp_max_points,
                sample_posterior=gp_sample_posterior,
                kernel_type=gp_kernel_type,
                coord_system=gp_coord_system,
            )

            ddpm_metrics = compute_all_metrics(ddpm_output, input_image_original, missing_mask, device)
            gp_metrics = compute_all_metrics(gp_output, input_image_original, missing_mask, device)

            mask_percent = 100.0 * top_left_crop(missing_mask, 44, 94).mean().item()

            writer.writerow([
                run_idx,
                sample_id,
                "StraightLinePath",
                mask_percent,
                ddpm_metrics["mse"],
                gp_metrics["mse"],
                ddpm_metrics["percent_error"],
                gp_metrics["percent_error"],
                ddpm_metrics["angular_error"],
                gp_metrics["angular_error"],
                ddpm_metrics["scaled_error_mag"],
                gp_metrics["scaled_error_mag"],
                ddpm_metrics["norm_mag_diff"],
                gp_metrics["norm_mag_diff"],
                ddpm_metrics["mse"] - gp_metrics["mse"],
                ddpm_metrics["percent_error"] - gp_metrics["percent_error"],
                ddpm_metrics["angular_error"] - gp_metrics["angular_error"],
                ddpm_metrics["scaled_error_mag"] - gp_metrics["scaled_error_mag"],
                ddpm_metrics["norm_mag_diff"] - gp_metrics["norm_mag_diff"],
            ])

    print(f"Saved results to {csv_path}")


if __name__ == "__main__":
    main()
