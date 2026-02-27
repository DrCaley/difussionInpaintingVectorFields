#!/usr/bin/env python3
"""Analyze eddy detection vs distance from observation track.

The mask is a single horizontal line at row 22 spanning 94px — that IS the
OBSERVED track (~2.4% of the ocean domain is known).  Everything else is
missing (97.6%).  This script asks: do eddies closer to the observation line
get detected more often than those farther away?
"""

import torch
import numpy as np
import pandas as pd
from ddpm.utils.eddy_detection import detect_eddies_gamma

# Load results
d = torch.load('results/eddy_balanced_eval/bulk_eval_eddy_balanced_100.pt',
               map_location='cpu', weights_only=False)

OBS_ROW = 22          # the ONE row we actually observe (known data)
H, W = 44, 94        # ocean region

DETECT_PARAMS = dict(
    radius=8, gamma_threshold=0.65, min_area=25, shore_buffer=2,
    smooth_sigma=2.0, min_mean_speed_ratio=0.3, min_vorticity=0.03,
)

print("=" * 80)
print("EDDY DISTANCE FROM OBSERVATION TRACK ANALYSIS")
print("=" * 80)
print(f"Observations: single row at y={OBS_ROW}  (~2.4% of ocean domain known)")
print(f"Everything else is missing (97.6%)")
print(f"Distance = |eddy_center_y − {OBS_ROW}| (distance to observation track)")
print()

# For each eddy sample, detect GT eddies, measure distance to obs row
records = []
for si, s in enumerate(d['samples'][:50]):  # first 50 are eddy samples
    val_idx = s['val_idx']
    gt = s['ground_truth'][0, :, :H, :W]              # [2, 44, 94] tensor
    ddpm_out = s['ddpm_output'][0, :, :H, :W]
    gp_out = s['gp_output'][0, :, :H, :W]
    land = s['land_mask'][0, 0, :H, :W]
    ocean = land > 0  # torch bool tensor

    # detect_eddies_gamma takes a single (2,H,W) tensor
    gt_eddies, _, _ = detect_eddies_gamma(gt, ocean_mask=ocean, **DETECT_PARAMS)
    ddpm_eddies, _, _ = detect_eddies_gamma(ddpm_out, ocean_mask=ocean, **DETECT_PARAMS)
    gp_eddies, _, _ = detect_eddies_gamma(gp_out, ocean_mask=ocean, **DETECT_PARAMS)

    for gt_e in gt_eddies:
        cy, cx = gt_e.center_y, gt_e.center_x
        dist_to_obs = abs(cy - OBS_ROW)

        # Fraction of eddy pixels overlapping with observation row
        eddy_ys, eddy_xs = np.where(gt_e.mask)
        n_on_obs = np.sum(eddy_ys == OBS_ROW)
        frac_on_obs = n_on_obs / len(eddy_ys) if len(eddy_ys) > 0 else 0

        # Min distance from ANY eddy pixel to obs row
        if len(eddy_ys) > 0:
            min_pixel_dist = int(np.min(np.abs(eddy_ys - OBS_ROW)))
        else:
            min_pixel_dist = int(dist_to_obs)
        overlaps_obs = n_on_obs > 0

        # Was this eddy detected in DDPM?
        ddpm_detected = False
        ddpm_match_dist = None
        for de in ddpm_eddies:
            d_center = np.sqrt((cy - de.center_y)**2 + (cx - de.center_x)**2)
            if d_center < 10:
                ddpm_detected = True
                ddpm_match_dist = d_center
                break

        # Was this eddy detected in GP?
        gp_detected = False
        for ge in gp_eddies:
            d_center = np.sqrt((cy - ge.center_y)**2 + (cx - ge.center_x)**2)
            if d_center < 10:
                gp_detected = True
                break

        records.append({
            'val_idx': val_idx,
            'center_y': cy, 'center_x': cx,
            'area': gt_e.area_pixels,
            'dist_to_obs': dist_to_obs,
            'min_pixel_dist_to_obs': min_pixel_dist,
            'overlaps_obs': overlaps_obs,
            'frac_on_obs': frac_on_obs,
            'ddpm_detected': ddpm_detected,
            'ddpm_match_dist': ddpm_match_dist,
            'gp_detected': gp_detected,
            'vorticity': gt_e.mean_vorticity,
        })

df = pd.DataFrame(records)
print(f"Total GT eddies in 50 eddy samples: {len(df)}")
print(f"Detected by DDPM: {df['ddpm_detected'].sum()}")
print(f"Detected by GP:   {df['gp_detected'].sum()}")
print()

# ── Distance bins (center distance to obs row) ──
print("─" * 80)
print("DETECTION RATE BY EDDY CENTER DISTANCE TO OBSERVATION ROW")
print("─" * 80)
print(f"{'Dist (px)':>10s} {'Count':>6s} {'DDPM':>5s} {'Recall':>8s} {'Avg Area':>9s} {'Avg|vort|':>10s} {'Overlap':>8s}")
print("─" * 80)
bins = [(0, 4), (4, 8), (8, 12), (12, 16), (16, 22)]
for lo, hi in bins:
    sub = df[(df['dist_to_obs'] >= lo) & (df['dist_to_obs'] < hi)]
    if len(sub) == 0:
        continue
    n_det = sub['ddpm_detected'].sum()
    recall = n_det / len(sub) * 100
    avg_area = sub['area'].mean()
    avg_vort = sub['vorticity'].mean()
    n_overlap = sub['overlaps_obs'].sum()
    print(f"  {lo:>2d}–{hi:<2d}     {len(sub):>4d}    {n_det:>3d}   {recall:>5.1f}%    {avg_area:>6.1f}    {avg_vort:>8.4f}    {n_overlap:>3d}/{len(sub)}")

# ── Split: overlapping vs non-overlapping ──
print()
print("─" * 80)
print("DETECTION RATE: EDDIES THAT OVERLAP OBS ROW vs DON'T")
print("─" * 80)
over = df[df['overlaps_obs']]
noover = df[~df['overlaps_obs']]
if len(over) > 0:
    print(f"  Overlaps obs row ({len(over):>2d} eddies): "
          f"DDPM recall = {over['ddpm_detected'].sum()}/{len(over)} "
          f"({over['ddpm_detected'].mean()*100:.1f}%)")
if len(noover) > 0:
    print(f"  No overlap       ({len(noover):>2d} eddies): "
          f"DDPM recall = {noover['ddpm_detected'].sum()}/{len(noover)} "
          f"({noover['ddpm_detected'].mean()*100:.1f}%)")

# ── Individual eddies table ──
print()
print("─" * 95)
print("ALL GT EDDIES (sorted by distance to observation row)")
print(f"{'ValIdx':>7s} {'CtrY':>5s} {'CtrX':>5s} {'Dist':>5s} {'MinPx':>6s} {'Area':>5s} "
      f"{'Overlap':>7s} {'|Vort|':>8s} {'DDPM':>5s} {'GP':>4s}")
print("─" * 95)
for _, r in df.sort_values('dist_to_obs').iterrows():
    det_str = " YES" if r['ddpm_detected'] else "  no"
    gp_str = "YES" if r['gp_detected'] else " no"
    ovr_str = "yes" if r['overlaps_obs'] else " no"
    print(f"  {r['val_idx']:>5.0f} {r['center_y']:>5.1f} {r['center_x']:>5.1f} "
          f"{r['dist_to_obs']:>5.1f} {r['min_pixel_dist_to_obs']:>5d}  "
          f"{r['area']:>5.0f}   {ovr_str:>4s}   "
          f"{r['vorticity']:>7.4f}  {det_str}  {gp_str}")

# ── Detected vs missed summary ──
print()
det = df[df['ddpm_detected']]
mis = df[~df['ddpm_detected']]
print("=" * 80)
print("DETECTED vs MISSED eddies (DDPM S6):")
if len(det) > 0:
    print(f"  Detected ({len(det):>2d}): avg dist to obs = {det['dist_to_obs'].mean():.1f}px, "
          f"avg area = {det['area'].mean():.0f}, "
          f"avg |vort| = {det['vorticity'].mean():.4f}, "
          f"overlaps obs = {det['overlaps_obs'].sum()}/{len(det)}")
if len(mis) > 0:
    print(f"  Missed   ({len(mis):>2d}): avg dist to obs = {mis['dist_to_obs'].mean():.1f}px, "
          f"avg area = {mis['area'].mean():.0f}, "
          f"avg |vort| = {mis['vorticity'].mean():.4f}, "
          f"overlaps obs = {mis['overlaps_obs'].sum()}/{len(mis)}")

# ── Statistical tests (numpy-only Mann-Whitney U) ──
def mann_whitney_u(x, y):
    """Simple two-sided Mann-Whitney U using ranks."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    n1, n2 = len(x), len(y)
    combined = np.concatenate([x, y])
    ranks = np.empty_like(combined)
    order = combined.argsort()
    ranks[order] = np.arange(1, len(combined) + 1)
    # handle ties: average ranks
    for val in np.unique(combined):
        idx = combined == val
        ranks[idx] = ranks[idx].mean()
    R1 = ranks[:n1].sum()
    U1 = R1 - n1 * (n1 + 1) / 2
    # normal approximation for p-value
    mu = n1 * n2 / 2
    sigma = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    z = (U1 - mu) / sigma if sigma > 0 else 0
    # two-sided p via normal CDF approximation (Abramowitz & Stegun)
    from math import erfc, sqrt
    p = erfc(abs(z) / sqrt(2))
    return U1, p, z

print()
if len(det) > 1 and len(mis) > 1:
    u1, p1, z1 = mann_whitney_u(
        det['dist_to_obs'].values,
        mis['dist_to_obs'].values)
    print(f"Mann-Whitney U (dist to obs): U={u1:.1f}, z={z1:.2f}, p={p1:.4f}")

    u2, p2, z2 = mann_whitney_u(
        det['area'].values, mis['area'].values)
    print(f"Mann-Whitney U (area):        U={u2:.1f}, z={z2:.2f}, p={p2:.4f}")

    u3, p3, z3 = mann_whitney_u(
        det['vorticity'].values, mis['vorticity'].values)
    print(f"Mann-Whitney U (vorticity):   U={u3:.1f}, z={z3:.2f}, p={p3:.4f}")
print()
