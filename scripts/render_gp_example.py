import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from data_prep.data_initializer import DDInitializer
from ddpm.helper_functions.interpolation_tool import gp_fill
from ddpm.helper_functions.masks.robot_path import RobotPathGenerator
from ddpm.utils.inpainting_utils import top_left_crop


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


def save_mask(mask, save_path, title):
    m = mask.squeeze()[0].cpu()
    plt.figure(figsize=(6, 4))
    plt.imshow(m, cmap="gray")
    plt.title(title)
    plt.colorbar()
    plt.savefig(save_path, dpi=200)
    plt.close()


def main(sample_id=96):
    dd = DDInitializer()
    device = dd.get_device()

    loader = DataLoader(
        dd.get_validation_data(),
        batch_size=1,
        shuffle=False,
        num_workers=dd.get_attribute("num_workers") or 0,
    )

    batch = None
    for b in loader:
        if len(b) > 1 and int(b[1].item()) == sample_id:
            batch = b
            break
    if batch is None:
        batch = next(iter(loader))
    input_image = batch[0].to(device)
    sample_num = int(batch[1].item()) if len(batch) > 1 else 0

    input_image_original = dd.get_standardizer().unstandardize(torch.squeeze(input_image, 0)).to(device)
    input_image_original = torch.unsqueeze(input_image_original, 0)

    land_mask = (input_image_original.abs() > 1e-5).float().to(device)
    mask_generator = RobotPathGenerator()
    raw_mask = mask_generator.generate_mask(input_image.shape).to(device)
    missing_mask = raw_mask * land_mask

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
    mask_cropped = top_left_crop(missing_mask, 44, 94).to(device)

    results_dir = Path("results/gp_preview")
    results_dir.mkdir(parents=True, exist_ok=True)

    save_quiver(
        input_image_original_cropped,
        results_dir / f"initial_sample_{sample_num}.png",
        title=f"Initial (sample {sample_num})",
    )
    save_quiver(
        gp_field_cropped,
        results_dir / f"gp_fill_sample_{sample_num}.png",
        title=f"GP Fill (sample {sample_num})",
    )
    save_mask(
        mask_cropped,
        results_dir / f"mask_sample_{sample_num}.png",
        title=f"Mask (sample {sample_num})",
    )


if __name__ == "__main__":
    sample_id = os.environ.get("SAMPLE_ID")
    sample_id = int(sample_id) if sample_id is not None else 96
    main(sample_id=sample_id)
