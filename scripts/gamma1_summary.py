#!/usr/bin/env python3
"""Quick summary of Gamma1 detections on 10 samples (no plots)."""
import sys, pickle, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
from ddpm.utils.eddy_detection import detect_eddies_gamma

with open("data.pickle", "rb") as f:
    d, _, _ = pickle.load(f)
data = torch.from_numpy(d).float()
N = data.shape[-1]
indices = np.linspace(0, N - 1, 10, dtype=int)
expert = {1: "none", 2: "EDDY", 3: "none", 4: "none", 5: "none",
          6: "none", 7: "maybe", 8: "EDDY", 9: "EDDY", 10: "EDDY"}

print("Gamma1 detection: radius=8, thresh=0.6, min_area=30, smooth=2.0, shore=3")
print()
for i, tidx in enumerate(indices, 1):
    t0 = time.time()
    u = data[..., tidx][..., 0].T
    v = data[..., tidx][..., 1].T
    vel = torch.nan_to_num(torch.stack([u, v], 0), nan=0.0)
    eddies, g1, _ = detect_eddies_gamma(
        vel, radius=8, gamma_threshold=0.6, min_area=30,
        shore_buffer=3, smooth_sigma=2.0, min_mean_speed_ratio=0.3,
    )
    dt = time.time() - t0
    sizes = [e.area_pixels for e in eddies]
    peaks = [f"{e.swirl_fraction:.2f}" for e in eddies]
    centers = [f"({e.center_y:.0f},{e.center_x:.0f})" for e in eddies]
    tag = expert[i]
    match = "ok" if (len(eddies) > 0) == (tag == "EDDY") else ("?" if tag == "maybe" else "MISS")
    print(f"  S{i:2d} expert={tag:5s} det={len(eddies)} sizes={sizes} "
          f"peak_g1={peaks} at={centers}  [{match}] ({dt:.1f}s)")
