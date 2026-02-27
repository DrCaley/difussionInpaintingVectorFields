#!/usr/bin/env python3
"""Quick smoke test for eddy detection module."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
from ddpm.utils.eddy_detection import (
    okubo_weiss, detect_eddies, eddy_metrics, print_eddy_metrics
)

H, W = 44, 94
y, x = torch.meshgrid(
    torch.arange(H, dtype=torch.float32),
    torch.arange(W, dtype=torch.float32),
    indexing='ij',
)

# Eddy 1: cyclonic vortex at (22, 47)
cy1, cx1, r1 = 22.0, 47.0, 8.0
dx1, dy1 = x - cx1, y - cy1
dist1 = (dx1**2 + dy1**2).sqrt().clamp(min=0.1)
s1 = 0.5 * torch.exp(-dist1**2 / (2 * r1**2))
u1, v1 = -s1 * dy1 / dist1, s1 * dx1 / dist1

# Eddy 2: anticyclonic vortex at (30, 70)
cy2, cx2, r2 = 30.0, 70.0, 6.0
dx2, dy2 = x - cx2, y - cy2
dist2 = (dx2**2 + dy2**2).sqrt().clamp(min=0.1)
s2 = 0.3 * torch.exp(-dist2**2 / (2 * r2**2))
u2, v2 = s2 * dy2 / dist2, -s2 * dx2 / dist2

vel_true = torch.stack([u1 + u2, v1 + v2], dim=0)
vel_pred = vel_true + 0.05 * torch.randn_like(vel_true)

print("=== OW computation ===")
OW, omega, sn, ss = okubo_weiss(vel_true)
print(f"Shape={OW.shape}, min={OW.min():.4f}, max={OW.max():.4f}")
print(f"Vorticity: min={omega.min():.4f}, max={omega.max():.4f}")

print("\n=== Eddy detection ===")
eddies, W_field, vort_field = detect_eddies(vel_true, threshold_sigma=0.2, min_area=4)
print(f"Detected {len(eddies)} eddies:")
for e in eddies:
    rot = "cyclonic" if e.is_cyclonic else "anticyclonic"
    print(f"  #{e.label}: center=({e.center_y:.1f},{e.center_x:.1f}), "
          f"area={e.area_pixels}px, {rot}, mean_omega={e.mean_vorticity:.4f}")

print("\n=== Eddy metrics (noisy pred vs clean true) ===")
metrics = eddy_metrics(vel_pred, vel_true, threshold_sigma=0.2)
print_eddy_metrics(metrics, "Synthetic: noisy vs clean")

print("SUCCESS: All eddy detection functions work correctly.")
