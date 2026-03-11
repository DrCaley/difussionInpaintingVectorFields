#!/usr/bin/env python3
"""Pre-compute GP fields for all training and validation samples.

For each sample, applies the fixed row-22 observation mask, runs GP
regression in raw (unstandardized) space, and stores the result.

Output: data/rams_head/gp_precomputed.pt containing:
    - gp_train:   (N_train, 2, 64, 128) GP posterior means (raw space)
    - gp_test:    (N_test, 2, 64, 128)  GP posterior means (raw space)
    - var_train:  (N_train, 2, 64, 128) GP posterior variance (raw space)
    - var_test:   (N_test, 2, 64, 128)  GP posterior variance (raw space)
    - dist_map:   (1, 1, 64, 128)       distance to nearest observation (pixels)
    - mask:       (1, 1, 64, 128)       the observation mask used
    - gp_params:  dict with lengthscale, variance, noise, kernel_type

Usage:
    PYTHONPATH=. python3 scripts/precompute_gp.py
    PYTHONPATH=. python3 scripts/precompute_gp.py --quick 100  # first 100 only
"""

import argparse
import sys
import time
import torch
import numpy as np
from pathlib import Path
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from data_prep.data_initializer import DDInitializer
from ddpm.helper_functions.interpolation_tool import gp_fill
from ddpm.helper_functions.masks.border_mask import BorderMaskGenerator


def make_row22_mask(h=64, w=128):
    """Create the fixed row-22 observation mask (1=missing, 0=known)."""
    area_height, area_width = 44, 94
    mid_row = area_height // 2  # row 22
    mask = np.ones((h, w), dtype=np.float32)
    mask[mid_row:mid_row + 1, 0:area_width] = 0.0  # row 22 = known
    mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    border = BorderMaskGenerator().generate_mask((1, 2, h, w))
    mask = mask * border.cpu()
    return mask  # (1, 1, H, W)


def compute_gp_for_dataset(dataset, standardizer, mask_2ch, gp_params, n_max=None):
    """Compute GP posterior mean and variance for each sample in dataset.

    Args:
        dataset: base OceanImageDataset (returns (x0, t, noise))
        standardizer: to unstandardize x0 for GP
        mask_2ch: (1, 2, H, W) observation mask
        gp_params: dict of GP hyperparameters
        n_max: optional limit on number of samples

    Returns:
        gp_fields: (N, 2, 64, 128) tensor of GP posterior means (raw space)
        var_fields: (N, 2, 64, 128) tensor of GP posterior variances (raw space)
    """
    n = len(dataset) if n_max is None else min(n_max, len(dataset))
    gp_fields = torch.zeros(n, 2, 64, 128, dtype=torch.float32)
    var_fields = torch.zeros(n, 2, 64, 128, dtype=torch.float32)

    for i in tqdm(range(n), desc="Computing GP"):
        x0, _t, _noise = dataset[i]
        # x0 is standardized (2, 64, 128)
        x0_raw = standardizer.unstandardize(x0)  # (2, 64, 128)
        x0_raw_4d = x0_raw.unsqueeze(0)  # (1, 2, H, W)

        gp_mean, gp_var = gp_fill(
            x0_raw_4d,
            mask_2ch,
            lengthscale=gp_params["lengthscale"],
            variance=gp_params["variance"],
            noise=gp_params["noise"],
            use_double=True,
            kernel_type=gp_params["kernel_type"],
            coord_system=gp_params["coord_system"],
            return_variance=True,
        )
        gp_fields[i] = gp_mean.squeeze(0)
        var_fields[i] = gp_var.squeeze(0)

    return gp_fields, var_fields


def main():
    parser = argparse.ArgumentParser(description="Pre-compute GP fields")
    parser.add_argument("--quick", type=int, default=0,
                        help="Limit to first N samples per split (0=all)")
    args = parser.parse_args()

    dd = DDInitializer()
    standardizer = dd.get_standardizer()

    gp_params = {
        "lengthscale": float(dd.get_attribute("gp_lengthscale") or 14.1),
        "variance": float(dd.get_attribute("gp_variance") or 0.0103420345),
        "noise": float(dd.get_attribute("gp_noise") or 1e-8),
        "kernel_type": dd.get_attribute("gp_kernel_type") or "rbf_legacy",
        "coord_system": dd.get_attribute("gp_coord_system") or "pixels",
    }
    print(f"GP params: {gp_params}")

    # Build mask
    mask_1ch = make_row22_mask()  # (1, 1, 64, 128)
    mask_2ch = mask_1ch.expand(-1, 2, -1, -1)  # (1, 2, 64, 128)

    n_known_ocean = (mask_1ch[:, :, :44, :94] == 0).sum().item()
    n_ocean = 44 * 94
    print(f"Mask: {n_known_ocean} known ocean pixels out of {n_ocean} "
          f"({n_known_ocean/n_ocean*100:.1f}% coverage, "
          f"{(n_ocean - n_known_ocean)/n_ocean*100:.1f}% missing)")

    # Get base datasets (before OceanInpaintDataset wrapping)
    train_data = dd.get_training_data()
    test_data = dd.get_test_data()
    n_max = args.quick if args.quick > 0 else None

    # Compute distance-to-sensor map (same for all samples since mask is fixed)
    # mask_1ch: (1, 1, H, W), 1=missing, 0=known
    known_binary = (1.0 - mask_1ch.squeeze().numpy())  # 1=known, 0=missing
    dist_map_np = distance_transform_edt(1.0 - known_binary)  # distance from each pixel to nearest known
    # Normalize distance to [0, 1] range
    dist_max = dist_map_np.max()
    if dist_max > 0:
        dist_map_np = dist_map_np / dist_max
    dist_map = torch.tensor(dist_map_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    print(f"Distance map: shape={dist_map.shape}, range=[{dist_map.min():.3f}, {dist_map.max():.3f}]")

    print(f"\n=== Training set ({len(train_data)} samples) ===")
    t0 = time.time()
    gp_train, var_train = compute_gp_for_dataset(
        train_data, standardizer, mask_2ch, gp_params, n_max
    )
    t_train = time.time() - t0
    print(f"  Done in {t_train:.1f}s ({t_train/len(gp_train):.3f}s/sample)")

    print(f"\n=== Test/validation set ({len(test_data)} samples) ===")
    t0 = time.time()
    gp_test, var_test = compute_gp_for_dataset(
        test_data, standardizer, mask_2ch, gp_params, n_max
    )
    t_test = time.time() - t0
    print(f"  Done in {t_test:.1f}s ({t_test/len(gp_test):.3f}s/sample)")

    # Save
    out_path = BASE_DIR / "data" / "rams_head" / "gp_precomputed.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "gp_train": gp_train,
        "gp_test": gp_test,
        "var_train": var_train,
        "var_test": var_test,
        "dist_map": dist_map,
        "mask_1ch": mask_1ch,
        "gp_params": gp_params,
        "n_train": len(gp_train),
        "n_test": len(gp_test),
    }, out_path)

    size_mb = out_path.stat().st_size / 1e6
    print(f"\nSaved to {out_path} ({size_mb:.1f} MB)")
    print(f"  gp_train:  {gp_train.shape}")
    print(f"  var_train: {var_train.shape}")
    print(f"  gp_test:   {gp_test.shape}")
    print(f"  var_test:  {var_test.shape}")
    print(f"  dist_map:  {dist_map.shape}")


if __name__ == "__main__":
    main()
