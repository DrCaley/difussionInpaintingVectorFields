#!/usr/bin/env python3
"""Calibrate eddy detection against expert labels on 10 GT samples."""
import sys, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
from ddpm.utils.eddy_detection import detect_eddies, okubo_weiss

# Load data
with open("data.pickle", "rb") as f:
    training_data_np, _, _ = pickle.load(f)
data = torch.from_numpy(training_data_np).float()
N = data.shape[-1]
indices = np.linspace(0, N - 1, 10, dtype=int)

# Expert labels: which samples have eddies (1-indexed)
# User said: 10, 9, 8, 2, and maybe 7 have 1 or maybe 2 eddies each
has_eddy = {1: False, 2: True, 3: False, 4: False, 5: False,
            6: False, 7: True, 8: True, 9: True, 10: True}
# 7 is "maybe" so we'll be lenient on it

def load_sample(tidx):
    u = data[..., tidx][..., 0].T
    v = data[..., tidx][..., 1].T
    vel = torch.stack([u, v], dim=0)
    return torch.nan_to_num(vel, nan=0.0)

# First: show what the current defaults detect
print("=" * 70)
print("CURRENT DEFAULTS (sigma=0.2, min_area=16, shore_buffer=2, min_swirl=0.6)")
print("=" * 70)
for i, tidx in enumerate(indices, 1):
    vel = load_sample(tidx)
    eddies, W, omega = detect_eddies(vel)
    label = "EDDY" if has_eddy[i] else "none"
    detected = len(eddies)
    match = "✓" if (detected > 0) == has_eddy[i] else "✗"
    sizes = [e.area_pixels for e in eddies]
    swirls = [f"{e.swirl_fraction:.2f}" for e in eddies]
    print(f"  Sample {i:2d} (t={tidx:5d}): expert={label:4s}  detected={detected}  sizes={sizes}  swirl={swirls}  {match}")

# Sweep parameters including swirl fraction
print("\n" + "=" * 70)
print("PARAMETER SWEEP (with swirl fraction)")
print("=" * 70)

best_score = -1
best_params = None

for sigma in [0.2, 0.5, 0.8, 1.0, 1.5, 2.0]:
    for min_area in [16, 25, 50, 80, 100]:
        for shore_buf in [2, 3, 4]:
            for min_swirl in [0.4, 0.5, 0.6, 0.7, 0.8]:
                tp = fp = tn = fn = 0
                for i, tidx in enumerate(indices, 1):
                    vel = load_sample(tidx)
                    eddies, _, _ = detect_eddies(vel, threshold_sigma=sigma,
                                                  min_area=min_area,
                                                  shore_buffer=shore_buf,
                                                  min_swirl_fraction=min_swirl)
                    pred_has = len(eddies) > 0
                    true_has = has_eddy[i]
                    if pred_has and true_has:
                        tp += 1
                    elif pred_has and not true_has:
                        fp += 1
                    elif not pred_has and true_has:
                        if i == 7:
                            tn += 1  # lenient on "maybe"
                        else:
                            fn += 1
                    else:
                        tn += 1

                precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                score = f1 - 0.5 * fp

                if score > best_score:
                    best_score = score
                    best_params = (sigma, min_area, shore_buf, min_swirl, tp, fp, tn, fn, f1)

sigma, min_area, shore_buf, min_swirl, tp, fp, tn, fn, f1 = best_params
print(f"\nBEST: sigma={sigma}, min_area={min_area}, shore_buffer={shore_buf}, min_swirl={min_swirl}")
print(f"  TP={tp} FP={fp} TN={tn} FN={fn}  F1={f1:.3f}")

# Show details for best params
print(f"\n{'=' * 70}")
print(f"BEST PARAMS DETAIL")
print(f"{'=' * 70}")
for i, tidx in enumerate(indices, 1):
    vel = load_sample(tidx)
    eddies, W, omega = detect_eddies(vel, threshold_sigma=sigma,
                                      min_area=min_area,
                                      shore_buffer=shore_buf,
                                      min_swirl_fraction=min_swirl)
    label = "EDDY" if has_eddy[i] else "none"
    detected = len(eddies)
    match = "✓" if (detected > 0) == has_eddy[i] else ("?" if i == 7 else "✗")
    sizes = [e.area_pixels for e in eddies]
    locs = [(f"({e.center_y:.0f},{e.center_x:.0f})") for e in eddies]
    swirls = [f"{e.swirl_fraction:.2f}" for e in eddies]
    print(f"  Sample {i:2d}: expert={label:4s}  detected={detected}  sizes={sizes}  centers={locs}  swirl={swirls}  {match}")

# All combos with 0 FP
print(f"\n{'=' * 70}")
print("ALL COMBOS WITH 0 FALSE POSITIVES (sorted by recall):")
print(f"{'=' * 70}")
results = []
for sigma in [0.2, 0.5, 0.8, 1.0, 1.5, 2.0]:
    for min_area in [16, 25, 50, 80, 100]:
        for shore_buf in [2, 3, 4]:
            for min_swirl in [0.4, 0.5, 0.6, 0.7, 0.8]:
                tp = fp = tn = fn = 0
                for i, tidx in enumerate(indices, 1):
                    vel = load_sample(tidx)
                    eddies, _, _ = detect_eddies(vel, threshold_sigma=sigma,
                                                  min_area=min_area,
                                                  shore_buffer=shore_buf,
                                                  min_swirl_fraction=min_swirl)
                    pred_has = len(eddies) > 0
                    true_has = has_eddy[i]
                    if pred_has and true_has: tp += 1
                    elif pred_has and not true_has: fp += 1
                    elif not pred_has and true_has:
                        if i == 7: tn += 1
                        else: fn += 1
                    else: tn += 1
                if fp == 0 and tp > 0:
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    results.append((recall, tp, fn, sigma, min_area, shore_buf, min_swirl))

results.sort(reverse=True)
for r in results[:20]:
    recall, tp, fn, s, ma, sb, ms = r
    print(f"  sigma={s:3.1f}  min_area={ma:3d}  shore_buf={sb}  min_swirl={ms:.1f}  "
          f"TP={tp}  FN={fn}  recall={recall:.2f}")
