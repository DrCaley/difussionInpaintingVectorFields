#!/usr/bin/env python3
"""Multi-mask GP precomputation.

Generates GP posterior fields for multiple sparse Gaussian masks at different
known-pixel fractions (e.g. 0.1%, 1%, 5%, 10%).  Each mask gets its own GP
posterior mean/variance for every train+test sample.

Output: data/rams_head/gp_multimask.pt
  {
    "mask_configs": [
      {"known_frac": 0.001, "mask_1ch": tensor, "dist_map": tensor, "n_known": int},
      ...
    ],
    "gp_train": [tensor, tensor, ...],   # one (N_train, 2, H, W) per mask
    "gp_test":  [tensor, tensor, ...],   # one (N_test,  2, H, W) per mask
    "var_train": [tensor, tensor, ...],
    "var_test":  [tensor, tensor, ...],
    "gp_params": dict,
    "n_train": int,
    "n_test":  int,
  }

Usage:
    PYTHONPATH=. python scripts/precompute_gp_multimask.py
    PYTHONPATH=. python scripts/precompute_gp_multimask.py --fracs 0.001 0.01 0.05 0.10
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import time
import argparse
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

# ── Mask generation ───────────────────────────────────────────────────────

OCEAN_H, OCEAN_W = 44, 94
GRID_H, GRID_W = 64, 128


def make_sparse_gaussian_mask(known_frac, seed=42):
    """Generate sparse random mask with exact known fraction in ocean area.

    Args:
        known_frac: fraction of ocean pixels to mark as known (0-1)
        seed: random seed for reproducibility

    Returns:
        mask_1ch: (1, 1, H, W) float tensor. 1=missing, 0=known.
    """
    rng = np.random.RandomState(seed)

    n_ocean = OCEAN_H * OCEAN_W  # 4136
    n_known = max(1, int(round(known_frac * n_ocean)))

    # Random positions within ocean area (row-major indexing)
    ocean_indices = np.arange(n_ocean)
    chosen = rng.choice(ocean_indices, size=n_known, replace=False)
    rows = chosen // OCEAN_W
    cols = chosen % OCEAN_W

    # Start all-missing, set chosen points to known
    mask = np.ones((GRID_H, GRID_W), dtype=np.float32)
    mask[rows, cols] = 0.0

    # Convert to tensor, apply border mask (ensures padding stays masked)
    mask_t = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    border = BorderMaskGenerator().generate_mask((1, 2, GRID_H, GRID_W))
    mask_t = mask_t * border.cpu()

    actual_known = int((mask_t == 0).sum().item())
    print(f"  Mask: known_frac={known_frac:.4f}, requested={n_known}, "
          f"actual_known={actual_known} pixels "
          f"({100*actual_known/n_ocean:.2f}% of ocean)")

    return mask_t


def compute_distance_map(mask_1ch):
    """Compute normalized distance-to-nearest-known-pixel map."""
    known_binary = (1.0 - mask_1ch.squeeze().numpy())
    dist_np = distance_transform_edt(1.0 - known_binary)
    dist_max = dist_np.max()
    if dist_max > 0:
        dist_np = dist_np / dist_max
    dist_map = torch.tensor(dist_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return dist_map


# ── Worker functions (module-level for pickling) ──────────────────────────

_worker_data = {}


def worker_init(raw_tensor_np, mask_2ch_np, gp_params):
    """Initialize each worker with shared data (called once per process)."""
    import torch
    _worker_data["raw_tensor"] = torch.from_numpy(raw_tensor_np)
    _worker_data["mask_2ch"] = torch.from_numpy(mask_2ch_np)
    _worker_data["gp_params"] = gp_params


def process_sample(idx):
    """Process a single sample in a worker process."""
    x0_raw = _worker_data["raw_tensor"][idx]
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

    # Pre-extract and unstandardize in main process
    print(f"  Pre-extracting and unstandardizing tensors...", flush=True)
    all_x0_raw = torch.zeros(n, 2, GRID_H, GRID_W, dtype=torch.float32)
    for i in range(n):
        x0, _t, _noise = dataset[i]
        all_x0_raw[i] = standardizer.unstandardize(x0)

    all_x0_np = all_x0_raw.numpy()
    mask_2ch_np = mask_2ch.numpy()

    gp_fields = torch.zeros(n, 2, GRID_H, GRID_W, dtype=torch.float32)
    var_fields = torch.zeros(n, 2, GRID_H, GRID_W, dtype=torch.float32)

    t0 = time.time()
    completed = 0

    with Pool(
        processes=n_workers,
        initializer=worker_init,
        initargs=(all_x0_np, mask_2ch_np, gp_params),
    ) as pool:
        tasks = list(range(n))
        for idx, mean_np, var_np in pool.imap_unordered(process_sample, tasks, chunksize=32):
            gp_fields[idx] = torch.from_numpy(mean_np)
            var_fields[idx] = torch.from_numpy(var_np)
            completed += 1
            if completed % 500 == 0 or completed == n:
                elapsed = time.time() - t0
                rate = completed / elapsed
                eta = (n - completed) / rate if rate > 0 else 0
                print(f"  {completed}/{n} ({rate:.1f} samples/s, ETA {eta:.0f}s)",
                      flush=True)

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({elapsed/n:.3f}s/sample wall)")
    return gp_fields, var_fields


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Multi-mask GP precomputation")
    parser.add_argument("--fracs", nargs="+", type=float,
                        default=[0.001, 0.01, 0.05, 0.10],
                        help="Known fractions (default: 0.001 0.01 0.05 0.10)")
    parser.add_argument("--seed-base", type=int, default=42,
                        help="Base seed; each mask gets seed_base + index")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: data/rams_head/gp_multimask.pt)")
    args = parser.parse_args()

    n_workers = max(1, cpu_count() - 1)
    print(f"Using {n_workers} worker processes (OMP_NUM_THREADS=1)")
    print(f"Known fractions: {args.fracs}")

    # Load data
    dd = DDInitializer()
    standardizer = dd.get_standardizer()
    train_data = dd.get_training_data()
    test_data = dd.get_test_data()

    gp_params = {
        "lengthscale": float(dd.get_attribute("gp_lengthscale") or 14.1),
        "variance": float(dd.get_attribute("gp_variance") or 0.0103420345),
        "noise": float(dd.get_attribute("gp_noise") or 1e-8),
        "kernel_type": dd.get_attribute("gp_kernel_type") or "rbf_legacy",
        "coord_system": dd.get_attribute("gp_coord_system") or "pixels",
    }
    print(f"GP params: {gp_params}")

    # Generate masks and precompute GP for each
    mask_configs = []
    all_gp_train = []
    all_gp_test = []
    all_var_train = []
    all_var_test = []

    total_t0 = time.time()

    for i, frac in enumerate(args.fracs):
        seed = args.seed_base + i
        print(f"\n{'='*60}")
        print(f"MASK {i+1}/{len(args.fracs)}: {frac*100:.1f}% known (seed={seed})")
        print(f"{'='*60}")

        mask_1ch = make_sparse_gaussian_mask(frac, seed=seed)
        mask_2ch = mask_1ch.expand(-1, 2, -1, -1)
        dist_map = compute_distance_map(mask_1ch)

        print(f"  Distance map: range=[{dist_map.min():.3f}, {dist_map.max():.3f}]")

        mask_configs.append({
            "known_frac": frac,
            "mask_1ch": mask_1ch,
            "dist_map": dist_map,
            "n_known": int((mask_1ch == 0).sum().item()),
        })

        gp_train, var_train = process_dataset_mp(
            train_data, standardizer, mask_2ch, gp_params,
            f"Training (mask {i+1}: {frac*100:.1f}%)", n_workers
        )
        gp_test, var_test = process_dataset_mp(
            test_data, standardizer, mask_2ch, gp_params,
            f"Test (mask {i+1}: {frac*100:.1f}%)", n_workers
        )

        all_gp_train.append(gp_train)
        all_gp_test.append(gp_test)
        all_var_train.append(var_train)
        all_var_test.append(var_test)

    total_elapsed = time.time() - total_t0
    print(f"\n{'='*60}")
    print(f"All masks done in {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")

    # Save
    out_path = Path(args.output) if args.output else (
        BASE_DIR / "data" / "rams_head" / "gp_multimask.pt"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    save_data = {
        "mask_configs": mask_configs,
        "gp_train": all_gp_train,
        "gp_test": all_gp_test,
        "var_train": all_var_train,
        "var_test": all_var_test,
        "gp_params": gp_params,
        "n_train": len(all_gp_train[0]),
        "n_test": len(all_gp_test[0]),
        "n_masks": len(args.fracs),
        "known_fracs": args.fracs,
    }

    torch.save(save_data, out_path)
    size_mb = out_path.stat().st_size / 1e6
    print(f"\nSaved to {out_path} ({size_mb:.1f} MB)")
    for i, cfg in enumerate(mask_configs):
        print(f"  Mask {i+1}: {cfg['known_frac']*100:.1f}% known, "
              f"{cfg['n_known']} pixels, "
              f"gp_train={all_gp_train[i].shape}")


if __name__ == "__main__":
    main()
