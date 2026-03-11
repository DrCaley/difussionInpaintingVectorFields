"""
Full 4-method comparison using 50 eddy samples.
Vor-Diff only has 50 (all eddy), so we compare on that common subset.
Also computes eddy detection for Vor-Diff.
"""
import torch
import numpy as np
import sys
sys.path.insert(0, '.')
from ddpm.utils.eddy_detection import detect_eddies_gamma, gamma1_field

# Load all three eval files
print("Loading eval files...")
vd = torch.load('experiments/05_voronoi_warmstart/voronoi_gp_replace/results/voronoi_warmstart_eval.pt',
                 map_location='cpu', weights_only=False)
gp_eval = torch.load('results/eddy_balanced_eval/bulk_eval_eddy_balanced_100.pt',
                      map_location='cpu', weights_only=False)
vcnn_eval = torch.load('results/voronoi_cnn_eval/voronoi_cnn_eval_results.pt',
                        map_location='cpu', weights_only=False)

print(f"Vor-Diff samples: {len(vd['samples'])}")
print(f"GP/GP-Diff samples: {len(gp_eval['samples'])}")
print(f"Voronoi-CNN samples: {len(vcnn_eval['results'])}")

# --- Build lookup tables by val_idx ---
# GP/GP-Diff
gp_by_idx = {}
for s in gp_eval['samples']:
    gp_by_idx[s['val_idx']] = s

# Voronoi-CNN
vcnn_by_idx = {}
for r in vcnn_eval['results']:
    vcnn_by_idx[r['val_idx']] = r

# --- Collect MSE for the 50 Vor-Diff samples (all eddy) ---
gp_mses, gpdiff_mses, vcnn_mses, vordiff_mses, vor_raw_mses = [], [], [], [], []
matched = 0
unmatched_gp = 0
unmatched_vcnn = 0

# Also collect Vor-Diff outputs for eddy detection
vordiff_eddy_detection_inputs = []

for r in vd['samples']:
    vidx = r['val_idx']
    
    # Vor-Diff MSE
    vordiff_mses.append(r['vordiff_mse'])
    vor_raw_mses.append(r['vor_only_mse'])
    
    # GP and GP-Diff MSE from original eval
    if vidx in gp_by_idx:
        gp_mses.append(gp_by_idx[vidx]['gp_mse'])
        gpdiff_mses.append(gp_by_idx[vidx]['ddpm_mse'])
    else:
        gp_mses.append(r['gp_mse'])  # Vor-Diff eval also records these
        gpdiff_mses.append(r['gpdiff_mse'])
        unmatched_gp += 1
    
    # Voronoi-CNN MSE
    if vidx in vcnn_by_idx:
        vcnn_mses.append(vcnn_by_idx[vidx]['mse_all_ocean'])
        matched += 1
    else:
        unmatched_vcnn += 1
        vcnn_mses.append(float('nan'))
    
    # Save for eddy detection
    if 'vordiff_output' in r:
        vordiff_eddy_detection_inputs.append({
            'val_idx': vidx,
            'vordiff_output': r['vordiff_output'],
            'ground_truth': r['ground_truth'],
        })

print(f"\nMatched {matched} Voronoi-CNN samples, {unmatched_vcnn} unmatched")
print(f"Unmatched GP: {unmatched_gp}")

# Filter out NaN vcnn entries for clean stats
valid_vcnn = [x for x in vcnn_mses if not np.isnan(x)]

print(f"\n{'='*70}")
print(f"MSE COMPARISON — 50 EDDY SAMPLES")
print(f"{'='*70}")
print(f"{'Method':<20} {'Mean MSE':>12} {'Median MSE':>12} {'vs GP':>10}")
print(f"{'-'*60}")

gp_mean = np.mean(gp_mses)
gpdiff_mean = np.mean(gpdiff_mses)
vcnn_mean = np.mean(valid_vcnn)
vordiff_mean = np.mean(vordiff_mses)
vor_raw_mean = np.mean(vor_raw_mses)

gp_med = np.median(gp_mses)
gpdiff_med = np.median(gpdiff_mses)
vcnn_med = np.median(valid_vcnn)
vordiff_med = np.median(vordiff_mses)
vor_raw_med = np.median(vor_raw_mses)

print(f"{'GP':<20} {gp_mean:>12.6f} {gp_med:>12.6f} {'1.000x':>10}")
print(f"{'GP-Diff':<20} {gpdiff_mean:>12.6f} {gpdiff_med:>12.6f} {gpdiff_mean/gp_mean:>9.3f}x")
print(f"{'Voronoi (raw)':<20} {vor_raw_mean:>12.6f} {vor_raw_med:>12.6f} {vor_raw_mean/gp_mean:>9.3f}x")
print(f"{'Vor-Diff':<20} {vordiff_mean:>12.6f} {vordiff_med:>12.6f} {vordiff_mean/gp_mean:>9.3f}x")
if valid_vcnn:
    print(f"{'Voronoi-CNN':<20} {vcnn_mean:>12.6f} {vcnn_med:>12.6f} {vcnn_mean/gp_mean:>9.3f}x")

# --- Win rates (pairwise) on the 50 eddy samples ---
print(f"\n{'='*70}")
print(f"WIN RATES — 50 EDDY SAMPLES (pairwise)")
print(f"{'='*70}")

n = len(vordiff_mses)
gpdiff_beats_gp = sum(1 for i in range(n) if gpdiff_mses[i] < gp_mses[i])
vordiff_beats_gp = sum(1 for i in range(n) if vordiff_mses[i] < gp_mses[i])
vordiff_beats_gpdiff = sum(1 for i in range(n) if vordiff_mses[i] < gpdiff_mses[i])
gpdiff_beats_vordiff = sum(1 for i in range(n) if gpdiff_mses[i] < vordiff_mses[i])

vcnn_beats_gp = sum(1 for i in range(n) if not np.isnan(vcnn_mses[i]) and vcnn_mses[i] < gp_mses[i])
vcnn_beats_gpdiff = sum(1 for i in range(n) if not np.isnan(vcnn_mses[i]) and vcnn_mses[i] < gpdiff_mses[i])
vcnn_beats_vordiff = sum(1 for i in range(n) if not np.isnan(vcnn_mses[i]) and vcnn_mses[i] < vordiff_mses[i])
n_vcnn = len(valid_vcnn)

print(f"GP-Diff beats GP:       {gpdiff_beats_gp}/{n} ({100*gpdiff_beats_gp/n:.0f}%)")
print(f"Vor-Diff beats GP:      {vordiff_beats_gp}/{n} ({100*vordiff_beats_gp/n:.0f}%)")
print(f"Vor-Diff beats GP-Diff: {vordiff_beats_gpdiff}/{n} ({100*vordiff_beats_gpdiff/n:.0f}%)")
print(f"GP-Diff beats Vor-Diff: {gpdiff_beats_vordiff}/{n} ({100*gpdiff_beats_vordiff/n:.0f}%)")
if n_vcnn > 0:
    print(f"Voronoi-CNN beats GP:       {vcnn_beats_gp}/{n_vcnn} ({100*vcnn_beats_gp/n_vcnn:.0f}%)")
    print(f"Voronoi-CNN beats GP-Diff:  {vcnn_beats_gpdiff}/{n_vcnn} ({100*vcnn_beats_gpdiff/n_vcnn:.0f}%)")
    print(f"Voronoi-CNN beats Vor-Diff: {vcnn_beats_vordiff}/{n_vcnn} ({100*vcnn_beats_vordiff/n_vcnn:.0f}%)")

# --- Eddy detection for Vor-Diff ---
print(f"\n{'='*70}")
print(f"EDDY DETECTION — Vor-Diff (50 eddy samples)")
print(f"{'='*70}")

# Also recompute GP-Diff detection on the same 50 for fair comparison
# We need the GP-Diff outputs from the GP eval file
eddy_params = dict(radius=8, gamma_threshold=0.65, min_area=25,
                   shore_buffer=2, smooth_sigma=2.0,
                   min_mean_speed_ratio=0.3, min_vorticity=0.03)

def run_eddy_detection(velocity_tensor, ocean_mask):
    """velocity_tensor: (2, H, W) numpy array"""
    vel_t = torch.from_numpy(velocity_tensor).float()
    mask_t = torch.from_numpy(ocean_mask) if not torch.is_tensor(ocean_mask) else ocean_mask
    eddies, g1, vort = detect_eddies_gamma(vel_t, ocean_mask=mask_t, **eddy_params)
    return eddies, g1

def match_eddies(pred_eddies, gt_eddies):
    """Match predicted eddies to ground truth by overlap. Returns TP, FP, matched pairs."""
    matched_gt = set()
    tp_pairs = []
    fp_list = []
    
    for pe in pred_eddies:
        best_iou = 0
        best_gt_idx = -1
        for gi, ge in enumerate(gt_eddies):
            if gi in matched_gt:
                continue
            # Compute overlap — masks may be torch tensors
            pm = pe.mask.numpy() if torch.is_tensor(pe.mask) else np.asarray(pe.mask)
            gm = ge.mask.numpy() if torch.is_tensor(ge.mask) else np.asarray(ge.mask)
            overlap = np.sum(pm & gm)
            if overlap > 0:
                union = np.sum(pm | gm)
                iou = overlap / union
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gi
        if best_gt_idx >= 0:
            matched_gt.add(best_gt_idx)
            tp_pairs.append((pe, gt_eddies[best_gt_idx]))
        else:
            fp_list.append(pe)
    
    return tp_pairs, fp_list

# Run detection
vordiff_tp, vordiff_fp_total = 0, 0
gpdiff_tp_50, gpdiff_fp_50 = 0, 0
gt_total_eddies = 0
vordiff_center_dists = []
vordiff_area_ratios = []

for r in vd['samples']:
    vidx = r['val_idx']
    gt_raw = r['ground_truth']
    if torch.is_tensor(gt_raw):
        gt_raw = gt_raw.squeeze()  # remove batch dim if present
        gt_tensor = gt_raw.numpy()
    else:
        gt_tensor = np.squeeze(gt_raw)
    
    # Ocean mask from ground truth — shape should be (2, H, W)
    speed = np.sqrt(gt_tensor[0]**2 + gt_tensor[1]**2)
    ocean_mask = speed > 1e-8
    
    # Ground truth eddies
    gt_eddies, _ = run_eddy_detection(gt_tensor, ocean_mask)
    gt_total_eddies += len(gt_eddies)
    
    # Vor-Diff eddies
    if 'vordiff_output' in r:
        vd_raw = r['vordiff_output']
        if torch.is_tensor(vd_raw):
            vd_raw = vd_raw.squeeze()
            vd_tensor = vd_raw.numpy()
        else:
            vd_tensor = np.squeeze(vd_raw)
        vd_eddies, _ = run_eddy_detection(vd_tensor, ocean_mask)
        tp_pairs, fp_list = match_eddies(vd_eddies, gt_eddies)
        vordiff_tp += len(tp_pairs)
        vordiff_fp_total += len(fp_list)
        
        for pe, ge in tp_pairs:
            dist = np.sqrt((pe.center_y - ge.center_y)**2 + (pe.center_x - ge.center_x)**2)
            vordiff_center_dists.append(dist)
            if ge.area_pixels > 0:
                vordiff_area_ratios.append(pe.area_pixels / ge.area_pixels)
    
    # GP-Diff eddies (from same sample)
    if vidx in gp_by_idx and 'ddpm_output' in gp_by_idx[vidx]:
        dd_raw = gp_by_idx[vidx]['ddpm_output']
        if torch.is_tensor(dd_raw):
            dd_raw = dd_raw.squeeze()
            dd_tensor = dd_raw.numpy()
        else:
            dd_tensor = np.squeeze(dd_raw)
        dd_eddies, _ = run_eddy_detection(dd_tensor, ocean_mask)
        dd_tp_pairs, dd_fp_list = match_eddies(dd_eddies, gt_eddies)
        gpdiff_tp_50 += len(dd_tp_pairs)
        gpdiff_fp_50 += len(dd_fp_list)

print(f"\nGround-truth eddies in 50 eddy samples: {gt_total_eddies}")

# Vor-Diff eddy detection
vd_prec = vordiff_tp / (vordiff_tp + vordiff_fp_total) * 100 if (vordiff_tp + vordiff_fp_total) > 0 else 0
vd_rec = vordiff_tp / gt_total_eddies * 100 if gt_total_eddies > 0 else 0
vd_f1 = 2 * (vd_prec/100 * vd_rec/100) / (vd_prec/100 + vd_rec/100) if (vd_prec + vd_rec) > 0 else 0

print(f"\nVor-Diff: TP={vordiff_tp}, FP={vordiff_fp_total}, "
      f"Precision={vd_prec:.1f}%, Recall={vd_rec:.1f}%, F1={vd_f1:.3f}")
if vordiff_center_dists:
    print(f"  Center dist: mean={np.mean(vordiff_center_dists):.2f}px, "
          f"median={np.median(vordiff_center_dists):.2f}px")
if vordiff_area_ratios:
    print(f"  Area ratio: mean={np.mean(vordiff_area_ratios):.2f}, "
          f"median={np.median(vordiff_area_ratios):.2f}")

# GP-Diff on same 50 for comparison
gd_prec = gpdiff_tp_50 / (gpdiff_tp_50 + gpdiff_fp_50) * 100 if (gpdiff_tp_50 + gpdiff_fp_50) > 0 else 0
gd_rec = gpdiff_tp_50 / gt_total_eddies * 100 if gt_total_eddies > 0 else 0
gd_f1 = 2 * (gd_prec/100 * gd_rec/100) / (gd_prec/100 + gd_rec/100) if (gd_prec + gd_rec) > 0 else 0

print(f"\nGP-Diff (same 50): TP={gpdiff_tp_50}, FP={gpdiff_fp_50}, "
      f"Precision={gd_prec:.1f}%, Recall={gd_rec:.1f}%, F1={gd_f1:.3f}")

# --- Final summary table ---
print(f"\n{'='*70}")
print(f"FULL COMPARISON TABLE — 50 EDDY SAMPLES")
print(f"{'='*70}")
print(f"{'Method':<16} {'MSE':>8} {'vs GP':>8} {'TP':>4} {'FP':>4} {'Prec':>7} {'Rec':>7} {'F1':>6}")
print(f"{'-'*65}")
print(f"{'GP':<16} {gp_mean:>.5f} {'1.00x':>8} {'1':>4} {'0':>4} {'100%':>7} {'1.8%':>7} {'0.034':>6}")
print(f"{'GP-Diff':<16} {gpdiff_mean:>.5f} {gpdiff_mean/gp_mean:>.3f}x {gpdiff_tp_50:>4} {gpdiff_fp_50:>4} {gd_prec:>6.1f}% {gd_rec:>6.1f}% {gd_f1:>6.3f}")
print(f"{'Voronoi (raw)':<16} {vor_raw_mean:>.5f} {vor_raw_mean/gp_mean:>.3f}x {'—':>4} {'—':>4} {'—':>7} {'—':>7} {'—':>6}")
print(f"{'Vor-Diff':<16} {vordiff_mean:>.5f} {vordiff_mean/gp_mean:>.3f}x {vordiff_tp:>4} {vordiff_fp_total:>4} {vd_prec:>6.1f}% {vd_rec:>6.1f}% {vd_f1:>6.3f}")
if valid_vcnn:
    print(f"{'Voronoi-CNN':<16} {vcnn_mean:>.5f} {vcnn_mean/gp_mean:>.3f}x {'28':>4} {'1':>4} {'96.6%':>7} {'49.1%':>7} {'0.651':>6}")

print(f"\nNote: GP and Voronoi-CNN eddy numbers are from full 100-sample eval.")
print(f"GP-Diff eddy numbers above are recomputed on just these 50 eddy samples.")
print(f"Voronoi-CNN eddy detection on eddy samples: TP=28, FP=1, Prec=96.6%, Rec=49.1%, F1=0.651")
