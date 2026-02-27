#!/usr/bin/env python3
"""
Scan the entire validation set (1965 samples) for eddies using
the Gamma1 vortex-identification method.

Outputs:
    results/val_eddy_catalogue.pt   — dict with:
        eddy_indices  : list of val-set indices that contain >= 1 eddy
        eddy_details  : dict mapping index → list of eddy info dicts
        no_eddy_count : number of eddy-free samples
        params        : detection parameters used

Usage:
    PYTHONPATH=. python3 scripts/scan_val_eddies.py
"""
import sys, os, pickle, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np

from ddpm.utils.eddy_detection import detect_eddies_gamma

# ── Gamma1 tuned parameters (same as calibration session) ────────
RADIUS = 8
GAMMA_THRESH = 0.65
MIN_AREA = 25
SHORE_BUFFER = 2
SMOOTH_SIGMA = 2.0
MIN_SPEED_RATIO = 0.3
MIN_VORTICITY = 0.03

OUT_PATH = "results/val_eddy_catalogue.pt"
os.makedirs("results", exist_ok=True)


def main():
    t0_global = time.time()

    # Load validation data from pickle (second element)
    print("Loading data.pickle ...")
    with open("data.pickle", "rb") as f:
        _, val_data_np, _ = pickle.load(f)

    val_data = torch.from_numpy(val_data_np).float()  # (94, 44, 2, N_val)
    N_val = val_data.shape[-1]
    print(f"Validation set: {N_val} samples, shape {val_data.shape}")
    print(f"Gamma1 params: radius={RADIUS}, thresh={GAMMA_THRESH}, "
          f"min_area={MIN_AREA}, smooth={SMOOTH_SIGMA}, "
          f"min_speed_ratio={MIN_SPEED_RATIO}, min_vorticity={MIN_VORTICITY}\n")

    eddy_indices = []
    eddy_details = {}
    n_total_eddies = 0

    for idx in range(N_val):
        # Extract (44, 94) velocity field
        u = val_data[..., idx][..., 0].T  # (44, 94)
        v = val_data[..., idx][..., 1].T
        vel = torch.stack([u, v], dim=0)
        vel = torch.nan_to_num(vel, nan=0.0)

        eddies, g1, omega = detect_eddies_gamma(
            vel,
            radius=RADIUS,
            gamma_threshold=GAMMA_THRESH,
            min_area=MIN_AREA,
            shore_buffer=SHORE_BUFFER,
            smooth_sigma=SMOOTH_SIGMA,
            min_mean_speed_ratio=MIN_SPEED_RATIO,
            min_vorticity=MIN_VORTICITY,
        )

        if len(eddies) > 0:
            eddy_indices.append(idx)
            eddy_details[idx] = [
                {
                    "center_y": e.center_y,
                    "center_x": e.center_x,
                    "area_pixels": e.area_pixels,
                    "is_cyclonic": e.is_cyclonic,
                    "peak_gamma1": e.swirl_fraction,
                    "mean_vorticity": e.mean_vorticity,
                }
                for e in eddies
            ]
            n_total_eddies += len(eddies)

        # Progress
        if (idx + 1) % 100 == 0 or (idx + 1) == N_val:
            elapsed = time.time() - t0_global
            rate = (idx + 1) / elapsed
            eta = (N_val - idx - 1) / rate
            print(f"  [{idx+1:5d}/{N_val}]  "
                  f"eddy samples so far: {len(eddy_indices)}  "
                  f"({rate:.0f} samples/s, ETA {eta:.0f}s)")

    # Save catalogue
    catalogue = {
        "eddy_indices": eddy_indices,
        "eddy_details": eddy_details,
        "n_val_total": N_val,
        "n_with_eddies": len(eddy_indices),
        "n_without_eddies": N_val - len(eddy_indices),
        "n_total_eddies": n_total_eddies,
        "params": {
            "method": "gamma1",
            "radius": RADIUS,
            "gamma_threshold": GAMMA_THRESH,
            "min_area": MIN_AREA,
            "shore_buffer": SHORE_BUFFER,
            "smooth_sigma": SMOOTH_SIGMA,
            "min_mean_speed_ratio": MIN_SPEED_RATIO,
            "min_vorticity": MIN_VORTICITY,
        },
    }
    torch.save(catalogue, OUT_PATH)

    elapsed_total = time.time() - t0_global
    print(f"\n{'='*60}")
    print(f"Validation eddy scan complete in {elapsed_total:.0f}s")
    print(f"  Total samples:        {N_val}")
    print(f"  Samples with eddies:  {len(eddy_indices)} "
          f"({100*len(eddy_indices)/N_val:.1f}%)")
    print(f"  Total eddies found:   {n_total_eddies}")
    print(f"  Saved → {OUT_PATH}")
    print(f"{'='*60}")

    # Show distribution of eddy counts per sample
    from collections import Counter
    cnt = Counter(len(v) for v in eddy_details.values())
    print(f"\nEddies per sample distribution:")
    for k in sorted(cnt.keys()):
        print(f"  {k} eddy(s): {cnt[k]} samples")


if __name__ == "__main__":
    main()
