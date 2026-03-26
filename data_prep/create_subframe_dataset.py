#!/usr/bin/env python3
"""Extract random 94×44 subframes from the large 242×329 ROMS grid.

Produces a pickle compatible with DDInitializer (extended format with
per-sample ocean masks, bathymetry, and statistics).

Usage:
    PYTHONPATH=. python data_prep/create_subframe_dataset.py [--crops-train 25] [--crops-eval 5] [--output data_subframes.pickle]
"""

import argparse
import datetime
import os
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np

# ── Constants ────────────────────────────────────────────────────────────────
MAT_FILE = "data/rams_head/stjohn_hourly_surface_velocity_20250718.mat"
CROP_H, CROP_W = 44, 94          # Same as old dataset (H, W)
MIN_OCEAN_FRAC = 0.75            # Reject crops with <75% ocean
MAX_NAN_FRAC_IN_OCEAN = 0.50     # Reject if >50% NaN within ocean cells
CHUNK_SIZE = 130                  # Temporal chunking (matches old pipeline)


def resolve_path(path: str) -> Path:
    """Try ../path first (PyCharm), then path as-is."""
    alt = Path("..") / path
    if alt.exists():
        return alt
    return Path(path)


def find_valid_positions(mask: np.ndarray) -> list[tuple[int, int]]:
    """Pre-compute (row, col) top-left positions where ocean fraction >= threshold.

    Args:
        mask: ROMS mask (242, 329), 1=ocean, 0=land.
    Returns:
        List of (row, col) tuples.
    """
    H, W = mask.shape
    max_row = H - CROP_H + 1
    max_col = W - CROP_W + 1

    # Use integral image for fast area sums
    integral = np.cumsum(np.cumsum(mask, axis=0), axis=1)
    total_cells = CROP_H * CROP_W

    valid = []
    for r in range(max_row):
        for c in range(max_col):
            # Sum of mask in [r:r+CROP_H, c:c+CROP_W]
            r2, c2 = r + CROP_H - 1, c + CROP_W - 1
            s = integral[r2, c2]
            if r > 0:
                s -= integral[r - 1, c2]
            if c > 0:
                s -= integral[r2, c - 1]
            if r > 0 and c > 0:
                s += integral[r - 1, c - 1]
            if s / total_cells >= MIN_OCEAN_FRAC:
                valid.append((r, c))

    return valid


def assign_splits(n_timesteps: int) -> dict[str, list[int]]:
    """Replicate the 130-frame chunk splitting from spliting_data_sets.py.

    Per chunk of 130 frames:
      train:  i .. i+70   (70 frames)
      gap:    10 frames
      val:    i+80 .. i+95  (15 frames)
      gap:    10 frames
      test:   i+105 .. i+120   (15 frames)
      gap:    10 frames
    """
    splits = {"train": [], "val": [], "test": []}
    for i in range(0, n_timesteps, CHUNK_SIZE):
        splits["train"].extend(range(i, min(i + 70, n_timesteps)))
        splits["val"].extend(range(i + 80, min(i + 95, n_timesteps)))
        splits["test"].extend(range(i + 105, min(i + 120, n_timesteps)))
    return splits


def extract_subframes(mat_path: Path, crops_train: int, crops_eval: int,
                      seed: int) -> dict:
    """Main extraction loop.

    Returns dict with keys: train/val/test each containing
    {velocity, bathymetry, ocean_mask} numpy arrays, plus stats.
    """
    rng = np.random.RandomState(seed)

    with h5py.File(mat_path, "r") as f:
        roms_mask = f["mask"][:]    # (H=242, W=329), 1=ocean, 0=land
        bathy_full = f["h"][:]      # (H=242, W=329), depth in metres
        us_ds = f["us"]              # (N, H=242, W=329) via lazy access
        vs_ds = f["vs"]

        n_timesteps = us_ds.shape[0]
        print(f"Source: {n_timesteps} timesteps, grid {roms_mask.shape}")

        # Step 1: valid crop positions
        valid_positions = find_valid_positions(roms_mask)
        print(f"Valid crop positions (≥{MIN_OCEAN_FRAC*100:.0f}% ocean): "
              f"{len(valid_positions)} / "
              f"{(roms_mask.shape[0]-CROP_H+1)*(roms_mask.shape[1]-CROP_W+1)} total")

        if len(valid_positions) == 0:
            raise RuntimeError("No valid crop positions found!")

        # Step 2: temporal splits
        splits = assign_splits(n_timesteps)
        crops_per_split = {"train": crops_train, "val": crops_eval, "test": crops_eval}
        print(f"Timesteps — train: {len(splits['train'])}, "
              f"val: {len(splits['val'])}, test: {len(splits['test'])}")

        # Step 3: extract
        results = {}
        for split_name in ("train", "val", "test"):
            timesteps = splits[split_name]
            n_crops = crops_per_split[split_name]

            vel_list = []   # each: (CROP_W, CROP_H, 2) = (94, 44, 2) to match old convention
            bathy_list = [] # each: (CROP_W, CROP_H) = (94, 44)
            mask_list = []  # each: (CROP_W, CROP_H) = (94, 44)

            skipped_nan = 0
            for ti, t in enumerate(timesteps):
                if ti % 500 == 0:
                    print(f"  {split_name}: timestep {ti}/{len(timesteps)} "
                          f"({len(vel_list)} crops so far)")

                # Read one timestep lazily — h5py shape is (T, H, W) = (T, lat, lon)
                u_frame = us_ds[t, :, :]  # (H=242, W=329)
                v_frame = vs_ds[t, :, :]  # (H=242, W=329)

                # Sample random crop positions
                chosen = rng.choice(len(valid_positions), size=n_crops, replace=False)

                for idx in chosen:
                    r, c = valid_positions[idx]
                    u_crop = u_frame[r:r+CROP_H, c:c+CROP_W]   # (H=44, W=94)
                    v_crop = v_frame[r:r+CROP_H, c:c+CROP_W]
                    mask_crop = roms_mask[r:r+CROP_H, c:c+CROP_W]
                    bathy_crop = bathy_full[r:r+CROP_H, c:c+CROP_W]

                    # NaN check within ocean cells
                    ocean_cells = mask_crop > 0.5
                    n_ocean = ocean_cells.sum()
                    if n_ocean > 0:
                        nan_in_ocean = (np.isnan(u_crop[ocean_cells]) |
                                        np.isnan(v_crop[ocean_cells])).sum()
                        if nan_in_ocean / n_ocean > MAX_NAN_FRAC_IN_OCEAN:
                            skipped_nan += 1
                            continue

                    # NaN → 0 (same as old pipeline)
                    u_crop = np.nan_to_num(u_crop, nan=0.0)
                    v_crop = np.nan_to_num(v_crop, nan=0.0)

                    # Store in old convention: (W=94, H=44, 2)
                    vel = np.stack([u_crop.T, v_crop.T], axis=-1)  # (94, 44, 2)
                    vel_list.append(vel)
                    bathy_list.append(bathy_crop.T)     # (94, 44)
                    mask_list.append(mask_crop.T)        # (94, 44)

            # Stack along last axis: (94, 44, 2, N)
            velocity = np.stack(vel_list, axis=-1).astype(np.float32)
            bathymetry = np.stack(bathy_list, axis=-1).astype(np.float32)
            ocean_mask = np.stack(mask_list, axis=-1).astype(np.float32)

            print(f"  {split_name}: {velocity.shape[-1]} crops "
                  f"(skipped {skipped_nan} for NaN)")

            results[split_name] = {
                "velocity": velocity,
                "bathymetry": bathymetry,
                "ocean_mask": ocean_mask,
            }

    return results


def compute_statistics(results: dict) -> dict:
    """Compute training-set statistics for standardization."""
    vel = results["train"]["velocity"]  # (94, 44, 2, N)
    u = vel[:, :, 0, :]
    v = vel[:, :, 1, :]
    mask = results["train"]["ocean_mask"]  # (94, 44, N)

    # Mask out land cells for statistics
    mask_3d = mask > 0.5
    u_ocean = u[np.broadcast_to(mask_3d, u.shape)]
    v_ocean = v[np.broadcast_to(mask_3d, v.shape)]

    u_mean = float(np.mean(u_ocean))
    u_std = float(np.std(u_ocean))
    v_mean = float(np.mean(v_ocean))
    v_std = float(np.std(v_ocean))

    shared_mean = (u_mean + v_mean) / 2
    shared_std = float(np.sqrt((u_std**2 + v_std**2) / 2))

    magnitudes = np.sqrt(u_ocean**2 + v_ocean**2)
    mag_mean = float(np.mean(magnitudes))

    bathy = results["train"]["bathymetry"]
    bathy_ocean = bathy[mask_3d]
    bathy_min = float(np.min(bathy_ocean))
    bathy_max = float(np.max(bathy_ocean))

    stats = {
        "u_training_mean": u_mean,
        "u_training_std": u_std,
        "v_training_mean": v_mean,
        "v_training_std": v_std,
        "shared_mean": shared_mean,
        "shared_std": shared_std,
        "mag_mean": mag_mean,
        "bathy_min": bathy_min,
        "bathy_max": bathy_max,
    }

    print("\n=== Training Statistics ===")
    for k, v_val in stats.items():
        print(f"  {k}: {v_val:.8f}")
    return stats


def save_pickle(results: dict, stats: dict, output_path: Path):
    """Save in extended pickle format (10 elements).

    Format: [train_vel, val_vel, test_vel,
             train_bathy, val_bathy, test_bathy,
             train_mask, val_mask, test_mask,
             stats_dict]
    """
    data = [
        results["train"]["velocity"],
        results["val"]["velocity"],
        results["test"]["velocity"],
        results["train"]["bathymetry"],
        results["val"]["bathymetry"],
        results["test"]["bathymetry"],
        results["train"]["ocean_mask"],
        results["val"]["ocean_mask"],
        results["test"]["ocean_mask"],
        stats,
    ]

    with open(output_path, "wb") as f:
        pickle.dump(data, f)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nSaved to {output_path} ({size_mb:.1f} MB)")
    print(f"  train: {results['train']['velocity'].shape[-1]} samples")
    print(f"  val:   {results['val']['velocity'].shape[-1]} samples")
    print(f"  test:  {results['test']['velocity'].shape[-1]} samples")


def save_mmap(results: dict, stats: dict, output_dir: Path):
    """Save as memory-mapped .npy files in a directory.

    Creates:
        output_dir/
            train_velocity.npy
            train_bathymetry.npy
            train_ocean_mask.npy
            val_velocity.npy
            ...
            meta.json   (stats + shapes)
    """
    import json
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        for key in ("velocity", "bathymetry", "ocean_mask"):
            arr = results[split][key]
            fpath = output_dir / f"{split}_{key}.npy"
            np.save(fpath, arr)
            print(f"  Saved {fpath} ({arr.shape}, {arr.nbytes / 1e6:.1f} MB)")

    meta = {"stats": stats}
    for split in ("train", "val", "test"):
        meta[f"{split}_samples"] = int(results[split]["velocity"].shape[-1])
    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    total_mb = sum(
        (output_dir / f"{s}_{k}.npy").stat().st_size
        for s in ("train", "val", "test")
        for k in ("velocity", "bathymetry", "ocean_mask")
    ) / (1024 * 1024)
    print(f"\nSaved to {output_dir}/ ({total_mb:.1f} MB total)")
    print(f"  train: {results['train']['velocity'].shape[-1]} samples")
    print(f"  val:   {results['val']['velocity'].shape[-1]} samples")
    print(f"  test:  {results['test']['velocity'].shape[-1]} samples")


def main():
    parser = argparse.ArgumentParser(description="Create subframe dataset")
    parser.add_argument("--crops-train", type=int, default=25,
                        help="Random crops per training timestep (default: 25)")
    parser.add_argument("--crops-eval", type=int, default=5,
                        help="Random crops per val/test timestep (default: 5)")
    parser.add_argument("--output", type=str, default="data_subframes",
                        help="Output path (dir for mmap, file for pickle)")
    parser.add_argument("--format", choices=["mmap", "pickle"], default="mmap",
                        help="Storage format (default: mmap)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--mat", type=str, default=None,
                        help="Override path to .mat file")
    args = parser.parse_args()

    mat_path = Path(args.mat) if args.mat else resolve_path(MAT_FILE)
    if not mat_path.exists():
        print(f"ERROR: .mat file not found at: {mat_path}")
        sys.exit(1)

    print(f"Source: {mat_path}")
    print(f"Crops per timestep — train: {args.crops_train}, eval: {args.crops_eval}")

    results = extract_subframes(mat_path, args.crops_train, args.crops_eval,
                                args.seed)
    stats = compute_statistics(results)

    if args.format == "mmap":
        save_mmap(results, stats, Path(args.output))
    else:
        save_pickle(results, stats, Path(args.output))

    print("\nDone!")


if __name__ == "__main__":
    main()
