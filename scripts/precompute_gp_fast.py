#!/usr/bin/env python3
"""Fast multiprocessing GP precompute.

Uses all CPU cores with OMP_NUM_THREADS=1 per worker for maximum throughput
on the small 94×94 GP systems. ~10 min vs ~130 min single-process.

Output: identical to precompute_gp.py → data/rams_head/gp_precomputed.pt
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import time
import torch
import numpy as np
from pathlib import Path
from scipy.ndimage import distance_transform_edt
from multiprocessing import Pool, cpu_count

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from data_prep.data_initializer import DDInitializer
from ddpm.helper_functions.interpolation_tool import gp_fill
from ddpm.helper_functions.masks.border_mask import BorderMaskGenerator


def make_row22_mask(h=64, w=128):
    area_height, area_width = 44, 94
    mid_row = area_height // 2
    mask = np.ones((h, w), dtype=np.float32)
    mask[mid_row:mid_row + 1, 0:area_width] = 0.0
    mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    border = BorderMaskGenerator().generate_mask((1, 2, h, w))
    mask = mask * border.cpu()
    return mask


# Module-level globals set by worker_init
_worker_data = {}


def worker_init(raw_tensor_np, mask_2ch_np, gp_params):
    """Initialize each worker with shared data (called once per process)."""
    import torch
    _worker_data["raw_tensor"] = torch.from_numpy(raw_tensor_np)
    _worker_data["mask_2ch"] = torch.from_numpy(mask_2ch_np)
    _worker_data["gp_params"] = gp_params


def process_sample(idx):
    """Process a single sample — runs in worker process."""
    x0_raw = _worker_data["raw_tensor"][idx]  # (2, 64, 128) already unstandardized
    mask_2ch = _worker_data["mask_2ch"]
    gp_params = _worker_data["gp_params"]

    x0_raw_4d = x0_raw.unsqueeze(0)
    mask_4d = mask_2ch.clone()
    if mask_4d.dim() == 3:
        mask_4d = mask_4d.unsqueeze(0)

    gp_mean, gp_var = gp_fill(
        x0_raw_4d,
        mask_4d,
        lengthscale=gp_params["lengthscale"],
        variance=gp_params["variance"],
        noise=gp_params["noise"],
        use_double=True,
        kernel_type=gp_params["kernel_type"],
        coord_system=gp_params["coord_system"],
        return_variance=True,
    )
    return idx, gp_mean.squeeze(0).numpy(), gp_var.squeeze(0).numpy()


def process_dataset_mp(dataset, standardizer, mask_2ch, gp_params, label, n_workers):
    """Process entire dataset using multiprocessing pool."""
    n = len(dataset)
    print(f"\n=== {label} ({n} samples, {n_workers} workers) ===")

    # Pre-extract all tensors and unstandardize in main process
    print(f"  Pre-extracting and unstandardizing tensors...", flush=True)
    all_x0_raw = torch.zeros(n, 2, 64, 128, dtype=torch.float32)
    for i in range(n):
        x0, _t, _noise = dataset[i]
        all_x0_raw[i] = standardizer.unstandardize(x0)

    # Prepare data for workers
    all_x0_np = all_x0_raw.numpy()
    mask_2ch_np = mask_2ch.numpy()

    # Standardizer state no longer needed in workers

    gp_fields = torch.zeros(n, 2, 64, 128, dtype=torch.float32)
    var_fields = torch.zeros(n, 2, 64, 128, dtype=torch.float32)

    t0 = time.time()
    completed = 0

    with Pool(
        processes=n_workers,
        initializer=worker_init,
        initargs=(all_x0_np, mask_2ch_np, gp_params),
    ) as pool:
        # Use imap_unordered for progress reporting
        tasks = list(range(n))
        for idx, mean_np, var_np in pool.imap_unordered(process_sample, tasks, chunksize=32):
            gp_fields[idx] = torch.from_numpy(mean_np)
            var_fields[idx] = torch.from_numpy(var_np)
            completed += 1
            if completed % 500 == 0 or completed == n:
                elapsed = time.time() - t0
                rate = completed / elapsed
                eta = (n - completed) / rate if rate > 0 else 0
                print(f"  {completed}/{n} ({rate:.1f} samples/s, ETA {eta:.0f}s)", flush=True)

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({elapsed/n:.3f}s/sample wall)")
    return gp_fields, var_fields


def main():
    n_workers = max(1, cpu_count() - 1)  # Leave 1 core for main process
    print(f"Using {n_workers} worker processes (OMP_NUM_THREADS=1)")

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

    mask_1ch = make_row22_mask()
    mask_2ch = mask_1ch.expand(-1, 2, -1, -1)

    # Distance map
    known_binary = (1.0 - mask_1ch.squeeze().numpy())
    dist_map_np = distance_transform_edt(1.0 - known_binary)
    dist_max = dist_map_np.max()
    if dist_max > 0:
        dist_map_np = dist_map_np / dist_max
    dist_map = torch.tensor(dist_map_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    print(f"Distance map: shape={dist_map.shape}, range=[{dist_map.min():.3f}, {dist_map.max():.3f}]")

    train_data = dd.get_training_data()
    test_data = dd.get_test_data()

    gp_train, var_train = process_dataset_mp(
        train_data, standardizer, mask_2ch, gp_params, "Training set", n_workers
    )
    gp_test, var_test = process_dataset_mp(
        test_data, standardizer, mask_2ch, gp_params, "Test/val set", n_workers
    )

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
