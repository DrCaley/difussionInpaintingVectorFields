#!/usr/bin/env python3
"""Regenerate quiver plots from saved tensors.pt files (no inference needed)."""

import sys, time
from pathlib import Path
import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE_DIR))

from plots.visualization_tools.plot_vector_field_tool import plot_vector_field

ARROW_LEN = 0.9
STEP = 2


def plot_quiver(vel_np, ocean_mask_np, title, out_path,
                obs_mask=None, arrow_len=ARROW_LEN, step=STEP):
    vx = torch.from_numpy(vel_np[0].astype(np.float32))
    vy = torch.from_numpy(vel_np[1].astype(np.float32))
    land = torch.from_numpy((ocean_mask_np < 0.5).astype(np.float32))
    miss = None
    if obs_mask is not None:
        miss_np = ((ocean_mask_np > 0.5) & (obs_mask < 0.5)).astype(np.float32)
        miss = torch.from_numpy(miss_np)
    plot_vector_field(
        vx, vy, step=step, scale=1.0, title=title, file=str(out_path),
        land_mask=land, land_color="forestgreen",
        crop_top_right_zero_pad=True,
        auto_rescale_for_display=True,
        target_median_arrow_len=arrow_len,
        missing_mask=miss, missing_color="red", missing_alpha=0.25,
    )


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Replot from saved tensors.pt")
    parser.add_argument("results_dir", type=str,
                        help="Path to results folder containing sample_* dirs")
    parser.add_argument("--arrow-len", type=float, default=ARROW_LEN)
    parser.add_argument("--step", type=int, default=STEP)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    sample_dirs = sorted(results_dir.glob("sample_*"))
    print(f"Replotting {len(sample_dirs)} samples with arrow_len={args.arrow_len}")
    t0 = time.time()

    for i, sd in enumerate(sample_dirs):
        d = torch.load(sd / "tensors.pt", map_location="cpu", weights_only=False)
        vi = d["val_idx"]
        mse = d["mse"]
        gt = d["gt"]
        om = d["ocean_mask"]
        obs = d["obs_mask"]

        plot_quiver(gt, om, f"GT (vi={vi})",
                    sd / "gt.png", arrow_len=args.arrow_len, step=args.step)
        plot_quiver(d["gp_mean"], om, f"GP (MSE={mse['gp_mse']:.6f})",
                    sd / "gp.png", obs_mask=obs,
                    arrow_len=args.arrow_len, step=args.step)
        plot_quiver(d["vcnn"], om, f"V-CNN (MSE={mse['vcnn_mse']:.6f})",
                    sd / "vcnn.png",
                    arrow_len=args.arrow_len, step=args.step)
        plot_quiver(d["composite"], om, f"Composite (MSE={mse['composite_mse']:.6f})",
                    sd / "composite.png",
                    arrow_len=args.arrow_len, step=args.step)
        # Support both old "multistep" and new "gpdiff" keys
        if "gpdiff" in d:
            gpdiff_key, mse_key, label = "gpdiff", "gpdiff_mse", "GP-Diff"
        else:
            gpdiff_key, mse_key, label = "multistep", "multistep_mse", "Multistep"
        plot_quiver(d[gpdiff_key], om, f"{label} (MSE={mse[mse_key]:.6f})",
                    sd / f"{gpdiff_key}.png",
                    arrow_len=args.arrow_len, step=args.step)

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(sample_dirs)} done ({time.time()-t0:.0f}s)")

    print(f"Done! {len(sample_dirs)} samples replotted in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
