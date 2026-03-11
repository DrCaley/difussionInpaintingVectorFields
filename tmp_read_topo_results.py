#!/usr/bin/env python3
"""Quick script to read and display topology inference results."""
import torch
from pathlib import Path

BASE = Path("experiments/10_topology_metrics/topo_aware_training/results")

def show_pt(path):
    if not path.exists():
        print(f"{path.name} NOT FOUND\n")
        return
    print(f"=== {path.name} ===")
    pt = torch.load(path, map_location="cpu", weights_only=False)
    for k, v in pt.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        elif isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                if isinstance(vv, (int, float)):
                    fmt = f"{vv:.6f}" if isinstance(vv, float) else str(vv)
                    print(f"    {kk}: {fmt}")
                elif isinstance(vv, torch.Tensor):
                    print(f"    {kk}: shape={vv.shape}")
                elif isinstance(vv, dict):
                    print(f"    {kk}:")
                    for kkk, vvv in vv.items():
                        if isinstance(vvv, (int, float)):
                            fmt = f"{vvv:.6f}" if isinstance(vvv, float) else str(vvv)
                            print(f"      {kkk}: {fmt}")
                        elif isinstance(vvv, torch.Tensor):
                            print(f"      {kkk}: shape={vvv.shape}")
                        else:
                            print(f"      {kkk}: {type(vvv).__name__}")
                elif isinstance(vv, list):
                    print(f"    {kk}: list len={len(vv)}")
                else:
                    print(f"    {kk}: {vv}")
        elif isinstance(v, (int, float, str, bool)):
            print(f"  {k}: {v}")
        elif isinstance(v, list):
            print(f"  {k}: list len={len(v)}")
        else:
            print(f"  {k}: {type(v).__name__}")
    print()

for f in ["topo_inference_results.pt", "topo_vs_baseline_inference.pt", "topo_gpdiff_eval_5.0pct.pt"]:
    show_pt(BASE / f)

# Now print the actual metrics with means
import numpy as np

print("=" * 70)
print("DETAILED METRICS")
print("=" * 70)

# topo_vs_baseline
p1 = BASE / "topo_vs_baseline_inference.pt"
if p1.exists():
    d = torch.load(p1, map_location="cpu", weights_only=False)
    print("\n--- Topo-Aware vs Baseline DDPM (10 samples) ---")
    for label, prefix in [("Topo-Aware", "topo"), ("Baseline", "base")]:
        mse = np.array(d[f"{prefix}_mse"])
        mae = np.array(d[f"{prefix}_mae"])
        div = np.array(d[f"{prefix}_div"])
        vort = np.array(d[f"{prefix}_vort_mse"])
        print(f"\n  {label}:")
        print(f"    MSE:       {mse.mean():.6f} +/- {mse.std():.6f}   (per-sample: {mse.tolist()})")
        print(f"    MAE:       {mae.mean():.6f} +/- {mae.std():.6f}")
        print(f"    Div(RMS):  {div.mean():.6f} +/- {div.std():.6f}")
        print(f"    Vort MSE:  {vort.mean():.6f} +/- {vort.std():.6f}")
    gt_div = np.array(d["gt_div"])
    print(f"\n  GT Divergence (RMS): {gt_div.mean():.6f} +/- {gt_div.std():.6f}")

# gpdiff eval
p2 = BASE / "topo_gpdiff_eval_5.0pct.pt"
if p2.exists():
    d2 = torch.load(p2, map_location="cpu", weights_only=False)
    print(f"\n--- GP-Diffusion Eval @ {d2['coverage_pct']}% coverage ({d2['n_samples']} samples) ---")
    res = d2["results"]
    for method in ["gp", "vcnn", "gpdiff", "topo_gpdiff"]:
        k = f"{method}_mse"
        if k in res:
            vals = np.array(res[k])
            print(f"  {method:15s} MSE: {vals.mean():.6f} +/- {vals.std():.6f}  (per-sample: {[f'{v:.6f}' for v in vals]})")
