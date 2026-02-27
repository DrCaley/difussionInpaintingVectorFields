#!/usr/bin/env python3
"""Test large-eddy-only detection on the 10 expert-labeled samples."""
import sys, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
from ddpm.utils.eddy_detection import detect_eddies

with open("data.pickle", "rb") as f:
    training_data_np, _, _ = pickle.load(f)
data = torch.from_numpy(training_data_np).float()
N = data.shape[-1]
indices = np.linspace(0, N - 1, 10, dtype=int)

has_eddy = {1: False, 2: True, 3: False, 4: False, 5: False,
            6: False, 7: True, 8: True, 9: True, 10: True}

def load_sample(tidx):
    u = data[..., tidx][..., 0].T
    v = data[..., tidx][..., 1].T
    vel = torch.stack([u, v], dim=0)
    return torch.nan_to_num(vel, nan=0.0)

# Ocean is 44x94 = 4136 px.  A "big" eddy = 5-15% of domain = 200-600 px
print(f"Ocean domain: 44x94 = {44*94} pixels")
print()

for min_area in [100, 150, 200, 300]:
    for sigma in [0.2, 0.5]:
        for min_swirl in [0.6, 0.7, 0.8]:
            tp = fp = tn = fn = 0
            detail_lines = []
            for i, tidx in enumerate(indices, 1):
                vel = load_sample(tidx)
                eddies, _, _ = detect_eddies(vel, threshold_sigma=sigma,
                                              min_area=min_area, shore_buffer=3,
                                              min_swirl_fraction=min_swirl)
                pred = len(eddies) > 0
                true = has_eddy[i]
                if pred and true: tp += 1
                elif pred and not true: fp += 1
                elif not pred and true:
                    if i == 7: tn += 1  # lenient
                    else: fn += 1
                else: tn += 1
                sizes = [e.area_pixels for e in eddies]
                swirls = [f"{e.swirl_fraction:.2f}" for e in eddies]
                mark = "✓" if pred == true else ("?" if i==7 and not pred else "✗")
                label = "EDDY" if true else "none"
                detail_lines.append(f"    S{i:2d} expert={label:4s} det={len(eddies):1d} sizes={sizes} swirl={swirls} {mark}")
            
            recall = tp/(tp+fn) if (tp+fn)>0 else 0
            prec = tp/(tp+fp) if (tp+fp)>0 else 1.0
            tag = "★" if fp == 0 and tp >= 3 else ""
            print(f"sigma={sigma:.1f}  min_area={min_area:3d}  min_swirl={min_swirl:.1f}  "
                  f"TP={tp} FP={fp} FN={fn}  prec={prec:.2f} recall={recall:.2f}  {tag}")
            if fp == 0 or (fp <= 1 and min_area >= 200):
                for line in detail_lines:
                    print(line)
                print()
