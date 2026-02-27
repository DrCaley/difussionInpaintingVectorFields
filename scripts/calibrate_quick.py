#!/usr/bin/env python3
"""Quick targeted calibration with swirl fraction."""
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

# Targeted sweep: fewer combos
results = []
for sigma in [0.2, 0.5, 0.8, 1.0, 1.5]:
    for min_area in [25, 50, 100]:
        for min_swirl in [0.6, 0.7, 0.8, 0.85, 0.9]:
            shore_buf = 3
            tp = fp = tn = fn = 0
            details = []
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
                details.append((i, len(eddies), [e.area_pixels for e in eddies],
                                [f"{e.swirl_fraction:.2f}" for e in eddies]))

            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            results.append((fp, -recall, sigma, min_area, min_swirl, tp, fn, details))

# Sort: 0 FP first, then highest recall
results.sort()
print(f"{'=' * 80}")
print(f"TOP 20 COMBOS (sorted by FP asc, then recall desc):")
print(f"{'=' * 80}")
for idx, r in enumerate(results[:20]):
    fp_val, neg_recall, s, ma, ms, tp, fn, details = r
    print(f"\n  [{idx+1}] sigma={s:.1f}  min_area={ma}  min_swirl={ms:.2f}  shore_buf=3")
    print(f"       TP={tp}  FP={fp_val}  FN={fn}  recall={-neg_recall:.2f}")
    for i, n, sizes, swirls in details:
        label = "EDDY" if has_eddy[i] else "none"
        match = "✓" if (n > 0) == has_eddy[i] else ("?" if i == 7 and n == 0 else "✗")
        print(f"       S{i:2d} expert={label:4s} det={n} sizes={sizes} swirl={swirls} {match}")
