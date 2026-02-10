import csv
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from data_prep.data_initializer import DDInitializer
from ddpm.helper_functions.interpolation_tool import gp_fill
from ddpm.helper_functions.masks.robot_path import RobotPathGenerator
from ddpm.helper_functions.masks.straigth_line import StraightLineMaskGenerator
from ddpm.utils.inpainting_utils import calculate_mse, top_left_crop


def _as_list(value, default):
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _build_masks(config):
    mask_names = _as_list(config.get("gp_tuning_masks"), ["straight_line"])
    masks = []
    for name in mask_names:
        if name == "straight_line":
            masks.append(StraightLineMaskGenerator(1))
        elif name == "robot_path":
            masks.append(RobotPathGenerator())
        else:
            raise ValueError(f"Unknown gp_tuning mask: {name}")
    return masks


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    dd = DDInitializer()
    device = dd.get_device()

    train_loader = DataLoader(
        dd.get_training_data(),
        batch_size=dd.get_attribute("inpainting_batch_size") or 1,
        shuffle=False,
        num_workers=dd.get_attribute("num_workers") or 0,
    )

    config = {
        "gp_tuning_num_images": dd.get_attribute("gp_tuning_num_images") or 10,
        "gp_tuning_lengthscales": dd.get_attribute("gp_tuning_lengthscales"),
        "gp_tuning_variances": dd.get_attribute("gp_tuning_variances"),
        "gp_tuning_noises": dd.get_attribute("gp_tuning_noises"),
        "gp_tuning_masks": dd.get_attribute("gp_tuning_masks"),
        "gp_tuning_coverage": dd.get_attribute("gp_tuning_coverage"),
    }

    lengthscales = _as_list(config.get("gp_tuning_lengthscales"), [0.5, 1.0, 1.5, 2.0, 3.0])
    variances = _as_list(config.get("gp_tuning_variances"), [0.5, 1.0, 2.0])
    noises = _as_list(config.get("gp_tuning_noises"), [1e-6, 1e-5, 1e-4])

    gp_use_double = dd.get_attribute("gp_use_double")
    gp_use_double = True if gp_use_double is None else gp_use_double
    gp_max_points = dd.get_attribute("gp_max_points")
    if gp_max_points is not None:
        gp_max_points = int(gp_max_points)
    gp_sample_posterior = dd.get_attribute("gp_sample_posterior") or False
    gp_kernel_type = dd.get_attribute("gp_kernel_type") or "rbf"
    gp_coord_system = dd.get_attribute("gp_coord_system") or "pixels"

    masks = _build_masks(config)

    results_dir = Path("./results")
    results_dir.mkdir(exist_ok=True)
    results_path = results_dir / "gp_tuning_results.csv"

    with open(results_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "lengthscale",
            "variance",
            "noise",
            "mask",
            "num_images",
            "mean_mse",
        ])

        best = {"mse": float("inf"), "params": None}

        for lengthscale in lengthscales:
            for variance in variances:
                for noise in noises:
                    for mask_generator in masks:
                        mses = []
                        image_counter = 0

                        for batch in train_loader:
                            if image_counter >= config["gp_tuning_num_images"]:
                                break

                            input_image = batch[0].to(device)
                            input_image_original = dd.get_standardizer().unstandardize(
                                torch.squeeze(input_image, 0)
                            ).to(device)
                            input_image_original = torch.unsqueeze(input_image_original, 0)

                            land_mask = (input_image_original.abs() > 1e-5).float().to(device)
                            raw_mask = mask_generator.generate_mask(input_image.shape).to(device)
                            missing_mask = raw_mask * land_mask

                            gp_field = gp_fill(
                                input_image_original,
                                missing_mask,
                                lengthscale=lengthscale,
                                variance=variance,
                                noise=noise,
                                use_double=gp_use_double,
                                max_points=gp_max_points,
                                sample_posterior=gp_sample_posterior,
                                kernel_type=gp_kernel_type,
                                coord_system=gp_coord_system,
                            )

                            input_image_original_cropped = top_left_crop(input_image_original, 44, 94).to(device)
                            gp_field_cropped = top_left_crop(gp_field, 44, 94).to(device)
                            mask_cropped = top_left_crop(missing_mask, 44, 94).to(device)

                            mse_gp = calculate_mse(
                                input_image_original_cropped,
                                gp_field_cropped,
                                mask_cropped,
                                normalize=True,
                            )

                            if not torch.isnan(mse_gp):
                                mses.append(mse_gp.item())

                            image_counter += 1

                        if mses:
                            mean_mse = float(sum(mses) / len(mses))
                        else:
                            mean_mse = float("nan")

                        writer.writerow([
                            lengthscale,
                            variance,
                            noise,
                            mask_generator,
                            image_counter,
                            mean_mse,
                        ])

                        logging.info(
                            "GP tuning: lengthscale=%s variance=%s noise=%s mask=%s mean_mse=%.6f",
                            lengthscale,
                            variance,
                            noise,
                            mask_generator,
                            mean_mse,
                        )

                        if mean_mse == mean_mse and mean_mse < best["mse"]:
                            best["mse"] = mean_mse
                            best["params"] = {
                                "lengthscale": lengthscale,
                                "variance": variance,
                                "noise": noise,
                                "mask": str(mask_generator),
                            }

    if best["params"] is not None:
        logging.info("Best GP params: %s (mean_mse=%.6f)", best["params"], best["mse"])
    else:
        logging.warning("No valid GP results found.")


if __name__ == "__main__":
    main()
