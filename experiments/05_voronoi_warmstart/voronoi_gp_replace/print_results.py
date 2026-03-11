#!/usr/bin/env python3
"""Print results from the Voronoi warm-start evaluation."""
import torch, numpy as np, sys, os

pt_path = os.path.join(os.path.dirname(__file__), "results", "voronoi_warmstart_eval.pt")
data = torch.load(pt_path, map_location="cpu", weights_only=False)
samples = data["samples"]
n = len(samples)
print(f"Samples: {n}")

gp = [s["gp_mse"] for s in samples]
gpdiff = [s["gpdiff_mse"] for s in samples]
vor_only = [s["vor_only_mse"] for s in samples]
vordiff = [s["vordiff_mse"] for s in samples]

print(f"\n{'='*60}")
print(f"RECONSTRUCTION MSE ({n} samples)")
print(f"{'='*60}")
print(f"  {'Method':<22} {'Mean MSE':>10} {'Median MSE':>12}")
print(f"  {'-'*46}")
for name, arr in [("GP", gp), ("GP-Diff (S6)", gpdiff),
                  ("Voronoi (no diff)", vor_only), ("Vor-Diff (S6)", vordiff)]:
    print(f"  {name:<22} {np.mean(arr):>10.6f} {np.median(arr):>12.6f}")

print(f"\nWIN RATES:")
vd_beats_gp = sum(1 for v, g in zip(vordiff, gp) if v < g)
vd_beats_gpdiff = sum(1 for v, g in zip(vordiff, gpdiff) if v < g)
gpdiff_beats_vd = sum(1 for v, g in zip(vordiff, gpdiff) if g < v)
print(f"  Vor-Diff beats GP:      {vd_beats_gp}/{n} ({100*vd_beats_gp/n:.0f}%)")
print(f"  Vor-Diff beats GP-Diff: {vd_beats_gpdiff}/{n} ({100*vd_beats_gpdiff/n:.0f}%)")
print(f"  GP-Diff beats Vor-Diff: {gpdiff_beats_vd}/{n} ({100*gpdiff_beats_vd/n:.0f}%)")

# Improvement ratios
print(f"\n  Vor-Diff / GP mean ratio:      {np.mean(vordiff)/np.mean(gp):.3f}x")
print(f"  Vor-Diff / GP-Diff mean ratio: {np.mean(vordiff)/np.mean(gpdiff):.3f}x")

# Eddy detection
print(f"\n{'='*60}")
print(f"EDDY DETECTION (Vor-Diff)")
print(f"{'='*60}")
total_tp = sum(s.get("vd_tp", 0) for s in samples)
total_fp = sum(s.get("vd_fp", 0) for s in samples)
total_fn = sum(s.get("vd_fn", 0) for s in samples)
prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
print(f"  TP={total_tp}  FP={total_fp}  FN={total_fn}")
print(f"  Precision={prec:.3f}  Recall={rec:.3f}  F1={f1:.3f}")

# FP breakdown
eddy_indices = set(data.get("eddy_indices", []))
fp_eddy = sum(s.get("vd_fp", 0) for s in samples if s["val_idx"] in eddy_indices)
fp_clean = sum(s.get("vd_fp", 0) for s in samples if s["val_idx"] not in eddy_indices)
print(f"  FP on eddy samples: {fp_eddy}, FP on clean samples: {fp_clean}")

# Per-subset MSE
eddy_vd = [s["vordiff_mse"] for s in samples if s["val_idx"] in eddy_indices]
clean_vd = [s["vordiff_mse"] for s in samples if s["val_idx"] not in eddy_indices]
eddy_gpdiff = [s["gpdiff_mse"] for s in samples if s["val_idx"] in eddy_indices]
clean_gpdiff = [s["gpdiff_mse"] for s in samples if s["val_idx"] not in eddy_indices]
print(f"\n  MSE by subset:")
print(f"    Eddy samples:   Vor-Diff={np.mean(eddy_vd):.6f}  GP-Diff={np.mean(eddy_gpdiff):.6f}")
print(f"    Clean samples:  Vor-Diff={np.mean(clean_vd):.6f}  GP-Diff={np.mean(clean_gpdiff):.6f}")

print(f"\n{'='*60}")
print(f"FULL COMPARISON TABLE")
print(f"{'='*60}")
print(f"  {'Method':<22} {'Mean MSE':>10} {'TP':>4} {'FP':>4} {'F1':>7}")
print(f"  {'-'*50}")
print(f"  {'GP':<22} {np.mean(gp):>10.6f} {'1':>4} {'0':>4} {'0.034':>7}")
print(f"  {'GP-Diff (S6)':<22} {np.mean(gpdiff):>10.6f} {'15':>4} {'4':>4} {'0.395':>7}")
print(f"  {'Voronoi-CNN':<22} {'0.001402':>10} {'28':>4} {'7':>4} {'0.609':>7}")
print(f"  {'Vor-Diff (S6)':<22} {np.mean(vordiff):>10.6f} {total_tp:>4} {total_fp:>4} {f1:>7.3f}")
print(f"  {'Voronoi (no diff)':<22} {np.mean(vor_only):>10.6f} {'--':>4} {'--':>4} {'--':>7}")
