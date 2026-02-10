import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from data_prep.data_initializer import DDInitializer
from ddpm.helper_functions.interpolation_tool import gp_fill
from ddpm.helper_functions.masks.robot_path import RobotPathGenerator
from ddpm.utils.inpainting_utils import top_left_crop
from plots.visualization_tools.error_visualization import save_magnitude_difference_heatmap


def summary_stats(arr):
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def save_quiver(tensor, save_path, title, vector_scale=0.15, step=2):
    t = tensor.squeeze()
    u, v = t[0].cpu(), t[1].cpu()
    h, w = u.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x = x[::step, ::step]
    y = y[::step, ::step]
    u = u[::step, ::step]
    v = v[::step, ::step]

    max_dim = 8
    figsize = (max_dim, max_dim * h / w) if w > h else (max_dim * w / h, max_dim)

    plt.figure(figsize=figsize)
    plt.quiver(
        x,
        y,
        u,
        v,
        scale=1.0 / vector_scale,
        width=0.002,
        headwidth=2,
        headlength=2,
        headaxislength=2,
        alpha=0.9,
    )
    plt.title(title)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_mask(mask_tensor, save_path, title):
    m = mask_tensor.squeeze()[0].cpu()
    plt.figure(figsize=(6, 4))
    plt.imshow(m, cmap="gray")
    plt.title(title)
    plt.colorbar()
    plt.savefig(save_path, dpi=200)
    plt.close()


def main():
    dd = DDInitializer()
    device = dd.get_device()

    loader = DataLoader(
        dd.get_validation_data(),
        batch_size=1,
        shuffle=False,
        num_workers=dd.get_attribute("num_workers") or 0,
    )

    right_slice = slice(-20, None)
    mask_generator = RobotPathGenerator()

    results = []
    for batch in loader:
        input_image = batch[0].to(device)
        sample_num = int(batch[1].item()) if len(batch) > 1 else 0

        input_image_original = dd.get_standardizer().unstandardize(torch.squeeze(input_image, 0)).to(device)
        input_image_original = torch.unsqueeze(input_image_original, 0)

        land_mask = (input_image_original.abs() > 1e-5).float().to(device)
        raw_mask = mask_generator.generate_mask(input_image.shape).to(device)
        missing_mask = raw_mask * land_mask

        missing_mask_cropped = top_left_crop(missing_mask, 44, 94).to(device)
        mask_vals = missing_mask_cropped[0, 0, :, right_slice].detach().cpu().numpy()

        if np.sum(mask_vals > 0.5) == 0:
            continue

        gp_field = gp_fill(
            input_image_original,
            missing_mask,
            lengthscale=dd.get_attribute("gp_lengthscale"),
            variance=dd.get_attribute("gp_variance"),
            noise=dd.get_attribute("gp_noise"),
            use_double=dd.get_attribute("gp_use_double"),
            max_points=dd.get_attribute("gp_max_points"),
            sample_posterior=dd.get_attribute("gp_sample_posterior"),
            kernel_type=dd.get_attribute("gp_kernel_type") or "rbf",
            coord_system=dd.get_attribute("gp_coord_system") or "pixels",
        )

        input_image_original_cropped = top_left_crop(input_image_original, 44, 94).to(device)
        gp_field_cropped = top_left_crop(gp_field, 44, 94).to(device)

        # Save full-field magnitude heatmap for first sample with right-side mask
        if len(results) == 0:
            out_dir = Path("results/gp_preview")
            out_dir.mkdir(parents=True, exist_ok=True)
            full_mask = torch.ones_like(missing_mask_cropped)
            save_magnitude_difference_heatmap(
                input_image_original_cropped,
                gp_field_cropped,
                full_mask,
                avg_magnitude=dd.get_attribute(attr="mag_mean"),
                title="gp_full_field",
                save_path=str(out_dir / f"gp_full_mag_diff_{sample_num}.png"),
            )
            save_quiver(
                gp_field_cropped,
                out_dir / f"gp_vector_field_{sample_num}.png",
                title=f"GP Vector Field (sample {sample_num})",
            )
            save_quiver(
                input_image_original_cropped,
                out_dir / f"initial_vector_field_{sample_num}.png",
                title=f"Initial Vector Field (sample {sample_num})",
            )
            save_mask(
                missing_mask_cropped,
                out_dir / f"mask_{sample_num}.png",
                title=f"Mask (sample {sample_num})",
            )

        for ch, name in enumerate(["u", "v"]):
            gp_vals = gp_field_cropped[0, ch, :, right_slice].detach().cpu().numpy()
            gt_vals = input_image_original_cropped[0, ch, :, right_slice].detach().cpu().numpy()

            gp_masked = gp_vals[mask_vals > 0.5]
            gt_masked = gt_vals[mask_vals > 0.5]

            if gp_masked.size == 0:
                continue

            mean_abs_diff = float(np.mean(np.abs(gp_masked - gt_masked)))
            results.append((mean_abs_diff, sample_num, name, summary_stats(gp_masked), summary_stats(gt_masked)))

        if len(results) >= 5:
            break

    if not results:
        print("No samples with masked values in right-side region found.")
        return

    results.sort(reverse=True, key=lambda x: x[0])
    for mean_abs_diff, sample_num, name, gp_stats, gt_stats in results:
        print(f"Sample {sample_num} - {name} channel right-side stats:")
        print("  GP masked stats:", gp_stats)
        print("  GT masked stats:", gt_stats)
        print("  Mean abs diff:", mean_abs_diff)


if __name__ == "__main__":
    main()
