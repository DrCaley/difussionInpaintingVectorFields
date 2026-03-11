#!/usr/bin/env python3
"""Compute Voronoi-CNN eddy detection + MSE numbers for paper."""
import torch, numpy as np, sys, math
sys.path.insert(0, '.')
from ddpm.utils.eddy_detection import detect_eddies_gamma

# Matching GP-Diff eval params exactly
EDDY_PARAMS = dict(radius=8, gamma_threshold=0.65, min_area=25,
                   shore_buffer=2, smooth_sigma=2.0,
                   min_mean_speed_ratio=0.3, min_vorticity=0.03)

# Load Voronoi-CNN results
vcnn = torch.load('results/voronoi_cnn_eval/voronoi_cnn_eval_results.pt',
                  map_location='cpu', weights_only=False)
results = vcnn['results']

# Load GP-Diff results for per-sample comparison
gp_eval = torch.load('results/eddy_balanced_eval/bulk_eval_eddy_balanced_100.pt',
                     map_location='cpu', weights_only=False)

# Build ocean mask from a ground truth sample (land = zero in both channels)
gt0 = results[0]['ground_truth']  # (2, 44, 94)
speed = (gt0[0]**2 + gt0[1]**2).sqrt()
ocean_mask = speed > 1e-8  # (44, 94) bool

# ---- MSE breakdown ----
eddy_mses = [r['mse_missing_only'] for r in results if r['is_eddy']]
noneddy_mses = [r['mse_missing_only'] for r in results if not r['is_eddy']]
all_mses = [r['mse_missing_only'] for r in results]

print("=" * 70)
print("VORONOI-CNN MSE (missing pixels only)")
print("=" * 70)
print(f"  All 100:       mean={np.mean(all_mses):.6f}  median={np.median(all_mses):.6f}")
print(f"  Eddy (50):     mean={np.mean(eddy_mses):.6f}  median={np.median(eddy_mses):.6f}")
print(f"  Non-eddy (50): mean={np.mean(noneddy_mses):.6f}  median={np.median(noneddy_mses):.6f}")

# ---- Eddy detection ----
print(f"\nEddy params: {EDDY_PARAMS}")
print("Running eddy detection on all 100 samples...")

gt_eddies_total = 0
tp_total = 0
fp_eddy = 0
fp_noneddy = 0
fn_total = 0
center_dists = []
area_ratios = []

for i, r in enumerate(results):
    pred = r['voronoi_pred']
    gt = r['ground_truth']
    is_eddy = r['is_eddy']

    gt_eddies_list, _, _ = detect_eddies_gamma(gt, ocean_mask=ocean_mask, **EDDY_PARAMS)
    pred_eddies_list, _, _ = detect_eddies_gamma(pred, ocean_mask=ocean_mask, **EDDY_PARAMS)

    if is_eddy:
        n_gt = len(gt_eddies_list)
        gt_eddies_total += n_gt

        # Match by spatial overlap (mask overlap)
        matched_gt = set()
        matched_pred = set()
        for pi, pe in enumerate(pred_eddies_list):
            for gi, ge in enumerate(gt_eddies_list):
                if gi in matched_gt:
                    continue
                if pe.mask is not None and ge.mask is not None:
                    overlap = (pe.mask & ge.mask).sum().item()
                    if overlap > 0:
                        matched_gt.add(gi)
                        matched_pred.add(pi)
                        # Compute center distance and area ratio
                        dist = math.sqrt((pe.center_y - ge.center_y)**2 +
                                         (pe.center_x - ge.center_x)**2)
                        center_dists.append(dist)
                        area_ratios.append(pe.area_pixels / max(ge.area_pixels, 1))
                        break

        tp_total += len(matched_pred)
        fp_eddy += len(pred_eddies_list) - len(matched_pred)
        fn_total += n_gt - len(matched_gt)
    else:
        fp_noneddy += len(pred_eddies_list)

    if (i + 1) % 25 == 0:
        print(f"  Processed {i+1}/100...")

fp_total = fp_eddy + fp_noneddy
precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
recall = tp_total / gt_eddies_total if gt_eddies_total > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"\n{'='*70}")
print(f"EDDY DETECTION — VORONOI-CNN")
print(f"{'='*70}")
print(f"  Ground truth eddies:        {gt_eddies_total}")
print(f"  True Positives:             {tp_total}")
print(f"  False Positives (eddy):     {fp_eddy}")
print(f"  False Positives (non-eddy): {fp_noneddy}")
print(f"  False Positives (total):    {fp_total}")
print(f"  False Negatives:            {fn_total}")
print(f"  Precision: {precision:.1%}")
print(f"  Recall:    {recall:.1%}")
print(f"  F1:        {f1:.3f}")
if center_dists:
    print(f"\n  Center distance: mean={np.mean(center_dists):.2f}px  median={np.median(center_dists):.2f}px  max={np.max(center_dists):.2f}px")
    print(f"  Area ratio:      mean={np.mean(area_ratios):.2f}  median={np.median(area_ratios):.2f}")
    bins = [(0,2), (2,4), (4,6), (6,8), (8,10)]
    for lo, hi in bins:
        cnt = sum(1 for d in center_dists if lo <= d < hi)
        print(f"    {lo}-{hi}px: {cnt} ({100*cnt/len(center_dists):.0f}%)")

# ---- Win rates vs GP/GP-Diff ----
gp_by_vidx = {}
gpdiff_by_vidx = {}
for s in gp_eval['results']:
    vi = s['val_idx']
    gp_by_vidx[vi] = s['gp_mse']
    gpdiff_by_vidx[vi] = s['ddpm_mse']

vcnn_beats_gp = vcnn_beats_gpdiff = matched_cnt = 0
for r in results:
    vi = r['val_idx']
    if vi in gp_by_vidx:
        matched_cnt += 1
        if r['mse_missing_only'] < gp_by_vidx[vi]:
            vcnn_beats_gp += 1
        if r['mse_missing_only'] < gpdiff_by_vidx[vi]:
            vcnn_beats_gpdiff += 1

print(f"\n{'='*70}")
print(f"WIN RATES (matched {matched_cnt}/100)")
print(f"{'='*70}")
print(f"  Voronoi-CNN beats GP:      {vcnn_beats_gp}/{matched_cnt} ({100*vcnn_beats_gp/matched_cnt:.0f}%)")
print(f"  Voronoi-CNN beats GP-Diff: {vcnn_beats_gpdiff}/{matched_cnt} ({100*vcnn_beats_gpdiff/matched_cnt:.0f}%)")

# ---- Full comparison table ----
print(f"\n{'='*70}")
print(f"FULL COMPARISON TABLE FOR PAPER")
print(f"{'='*70}")
print(f"{'Method':<18} {'Mean MSE':>10} {'Ratio':>8} {'TP':>4} {'FP':>4} {'Prec':>7} {'Recall':>7} {'F1':>6}")
print('-' * 70)
print(f"{'GP':<18} {'0.00522':>10} {'1.000x':>8} {'1':>4} {'0':>4} {'100.0%':>7} {'1.8%':>7} {'0.034':>6}")
print(f"{'GP-Diff':<18} {'0.00437':>10} {'0.837x':>8} {'14':>4} {'4':>4} {'77.8%':>7} {'24.6%':>7} {'0.373':>6}")
print(f"{'Voronoi-CNN':<18} {np.mean(all_mses):>10.5f} {np.mean(all_mses)/0.00522:>7.3f}x {tp_total:>4} {fp_total:>4} {precision:>6.1%} {recall:>6.1%} {f1:>6.3f}")
