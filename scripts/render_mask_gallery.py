import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from data_prep.data_initializer import DDInitializer
from ddpm.helper_functions.masks.straigth_line import StraightLineMaskGenerator
from ddpm.helper_functions.masks.gaussian_mask import GaussianNoiseBinaryMaskGenerator
from ddpm.helper_functions.masks.robot_path import RobotPathGenerator


def _to_2d(mask_tensor: torch.Tensor) -> np.ndarray:
    if isinstance(mask_tensor, np.ndarray):
        arr = mask_tensor
    else:
        arr = mask_tensor.detach().cpu().numpy()

    while arr.ndim > 2:
        arr = arr[0]
    return arr


def save_mask_image(mask_tensor, save_path, title):
    arr = _to_2d(mask_tensor)
    plt.figure(figsize=(6, 4))
    plt.imshow(arr, cmap="gray")
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
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

    batch = next(iter(loader))
    input_image = batch[0].to(device)
    image_shape = input_image.shape

    masks = [
        ("straight_line", StraightLineMaskGenerator()),
        ("gaussian", GaussianNoiseBinaryMaskGenerator()),
        ("robot_path", RobotPathGenerator()),
    ]

    results_dir = Path("results/mask_gallery")
    results_dir.mkdir(parents=True, exist_ok=True)

    for name, mask_gen in masks:
        try:
            mask = mask_gen.generate_mask(image_shape=image_shape)
            save_mask_image(mask, results_dir / f"{name}.png", title=str(mask_gen))
        except Exception as exc:
            print(f"Failed to render {name}: {exc}")


if __name__ == "__main__":
    main()
