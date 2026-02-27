#!/usr/bin/env python3
"""
Evaluate eddy detection across ALL 100 bulk-eval samples.

Splits samples into:
  - Eddy samples (GT has eddies): measures TP, FP, FN, distances, area ratios
  - Non-eddy samples (GT has no eddies): measures hallucinated FPs by GP and DDPM

Outputs:
    results/eddy_eval_all100_summary.txt
    results/eddy_eval_all100_results.pt
"""
import sys, os, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
from collections import defaultdict

from ddpm.utils.eddy_detection import detect_eddies_gamma

# ── Paths ────────────────────────────────────────────────────────
BULK_PT = ("experiments/02_inpaint_algorithm/"
           "repaint_gaussian_attn/results/"
           "bulk_eval_best_100samples.pt")
DIST_THRESH = 10.0

# ── Gamma1 params ────────────────────────────────────────────────
RADIUS = 8
GAMMA_THRESH = 0.65
MIN_AREA = 25
SHORE_BUFFER = 2
SMOOTH_SIGMA = 2.0
MIN_SPEED_RATIO = 0.3
MIN_VORTICITY = 0.03

OCEAN_H, OCEAN_W = 44, 94
OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)


def crop_to_ocean(tensor_4d):
    return tensor_4d.squeeze(0)[:, :OCEAN_H, :OCEAN_W]


def run_gamma1(vel):
    vel = torch.nan_to_num(vel, nan=0.0)
    eddies, _, _ = detect_eddies_gamma(
        vel, radius=RADIUS, gamma_threshold=GAMMA_THRESH,
        min_area=MIN_AREA, shore_buffer=SHORE_BUFFER,
        smooth_sigma=SMOOTH_SIGMA, min_mean_speed_ratio=MIN_SPEED_RATIO,
        min_vorticity=MIN_VORTICITY,
    )
    return eddies


def match_eddies(gt_eddies, pred_eddies, dist_thresh):
    if not gt_eddies or not pred_eddies:
        return [], list(range(len(gt_eddies))), list(range(len(pred_eddies)))

    n_gt, n_pred = len(gt_eddies), len(pred_eddies)
    dist = np.zeros((n_gt, n_pred))
    for i, ge in enumerate(gt_eddies):
        for j, pe in enumerate(pred_eddies):
            dy = ge.center_y - pe.center_y
            dx = ge.center_x - pe.center_x
            dist[i, j] = np.sqrt(dy**2 + dx**2)

    matches, used_gt, used_pred = [], set(), set()
    for _ in range(min(n_gt, n_pred)):
        best_d, best_i, best_j = float("inf"), -1, -1
        for i in range(n_gt):
            if i in used_gt:
                continue
            for j in range(n_pred):
                if j in used_pred:
                    continue
                if dist[i, j] < best_d:
                    best_d, best_i, best_j = dist[i, j], i, j
        if best_d <= dist_thresh:
            matches.append((best_i, best_j, best_d))
            used_gt.add(best_i)
            used_pred.add(best_j)
        else:
            break

    return (matches,
            [i for i in range(n_gt) if i not in used_gt],
            [j for j in range(n_pred) if j not in used_pred])


def main():
    t0 = time.time()
    print(f"Loading {BULK_PT}")
    bulk = torch.load(BULK_PT, map_location="cpu", weights_only=False)
    samples = bulk["samples"]
    print(f"  {len(samples)} samples\n")

    # ── Process all 100 ──────────────────────────────────────────
    eddy_results = []       # samples where GT has eddies
    noeddy_results = []     # samples where GT has NO eddies

    gp_stats = defaultdict(list)
    ddpm_stats = defaultdict(list)

    # Track non-eddy FPs separately
    gp_noeddy_fp = 0
    ddpm_noeddy_fp = 0
    gp_noeddy_fp_details = []
    ddpm_noeddy_fp_details = []

    print(f"{'#':>3} {'ValIdx':>6}  {'GT':>4}  {'GP':>10}  {'DDPM':>10}  Notes")
    print("-" * 70)

    for i, s in enumerate(samples):
        val_idx = s["val_idx"]
        gt_vel = crop_to_ocean(s["ground_truth"])
        gp_vel = crop_to_ocean(s["gp_output"])
        ddpm_vel = crop_to_ocean(s["ddpm_output"])

        gt_eddies = run_gamma1(gt_vel)
        gp_eddies = run_gamma1(gp_vel)
        ddpm_eddies = run_gamma1(ddpm_vel)

        n_gt = len(gt_eddies)
        n_gp = len(gp_eddies)
        n_ddpm = len(ddpm_eddies)

        if n_gt > 0:
            # ── Eddy sample ──
            gp_m, gp_fn, gp_fp = match_eddies(gt_eddies, gp_eddies, DIST_THRESH)
            dd_m, dd_fn, dd_fp = match_eddies(gt_eddies, ddpm_eddies, DIST_THRESH)

            gp_dists = [m[2] for m in gp_m]
            dd_dists = [m[2] for m in dd_m]
            gp_areas = [gp_eddies[m[1]].area_pixels / gt_eddies[m[0]].area_pixels for m in gp_m]
            dd_areas = [ddpm_eddies[m[1]].area_pixels / gt_eddies[m[0]].area_pixels for m in dd_m]

            gp_stats["detected"].append(len(gp_m))
            gp_stats["total_gt"].append(n_gt)
            gp_stats["false_pos"].append(len(gp_fp))
            gp_stats["distances"].extend(gp_dists)
            gp_stats["area_ratios"].extend(gp_areas)

            ddpm_stats["detected"].append(len(dd_m))
            ddpm_stats["total_gt"].append(n_gt)
            ddpm_stats["false_pos"].append(len(dd_fp))
            ddpm_stats["distances"].extend(dd_dists)
            ddpm_stats["area_ratios"].extend(dd_areas)

            gp_str = f"{len(gp_m)}/{n_gt}+{len(gp_fp)}fp"
            dd_str = f"{len(dd_m)}/{n_gt}+{len(dd_fp)}fp"
            dd_d = f"d={np.mean(dd_dists):.1f}" if dd_dists else "MISS"
            notes = f"EDDY  {dd_d}"

            eddy_results.append({
                "val_idx": val_idx, "n_gt": n_gt,
                "gp_det": len(gp_m), "gp_fp": len(gp_fp), "gp_dists": gp_dists,
                "ddpm_det": len(dd_m), "ddpm_fp": len(dd_fp), "ddpm_dists": dd_dists,
            })

        else:
            # ── Non-eddy sample ──
            gp_fp_count = n_gp
            ddpm_fp_count = n_ddpm
            gp_noeddy_fp += gp_fp_count
            ddpm_noeddy_fp += ddpm_fp_count

            if gp_fp_count > 0:
                gp_noeddy_fp_details.append((val_idx, gp_fp_count))
            if ddpm_fp_count > 0:
                ddpm_noeddy_fp_details.append((val_idx, ddpm_fp_count))

            gp_str = f"0fp" if gp_fp_count == 0 else f"{gp_fp_count}fp!"
            dd_str = f"0fp" if ddpm_fp_count == 0 else f"{ddpm_fp_count}fp!"
            notes = "no-eddy"
            if gp_fp_count or ddpm_fp_count:
                notes += " HALLUC"

            noeddy_results.append({
                "val_idx": val_idx,
                "gp_fp": gp_fp_count,
                "ddpm_fp": ddpm_fp_count,
            })

        print(f"{i+1:>3} {val_idx:>6}  {n_gt:>4}  {gp_str:>10}  {dd_str:>10}  {notes}")

    # ── Aggregate stats ──────────────────────────────────────────
    n_eddy_samples = len(eddy_results)
    n_noeddy_samples = len(noeddy_results)
    total_gt = sum(gp_stats["total_gt"])
    
    lines = []
    lines.append("=" * 70)
    lines.append("EDDY EVALUATION — ALL 100 SAMPLES")
    lines.append("=" * 70)
    lines.append(f"  Total samples:           {len(samples)}")
    lines.append(f"  Eddy samples (GT>0):     {n_eddy_samples}")
    lines.append(f"  Non-eddy samples (GT=0): {n_noeddy_samples}")
    lines.append(f"  Total GT eddies:         {total_gt}")
    lines.append(f"  Gamma1 params:           radius={RADIUS}, thresh={GAMMA_THRESH}, "
                 f"min_area={MIN_AREA}, shore_buffer={SHORE_BUFFER}")
    
    lines.append("\n" + "─" * 70)
    lines.append("  EDDY SAMPLES (detection quality)")
    lines.append("─" * 70)

    for method, label, ms in [("gp", "GP", gp_stats), ("ddpm", "DDPM S6", ddpm_stats)]:
        total_det = sum(ms["detected"])
        total_fp = sum(ms["false_pos"])
        det_rate = total_det / total_gt if total_gt else 0
        dists = ms["distances"]
        areas = ms["area_ratios"]

        lines.append(f"\n  {label}:")
        lines.append(f"    True positives:    {total_det}/{total_gt} = {det_rate:.1%}")
        lines.append(f"    False positives:   {total_fp} (in {n_eddy_samples} eddy samples)")
        lines.append(f"    False negatives:   {total_gt - total_det}")
        if dists:
            lines.append(f"    Center distance:   mean={np.mean(dists):.2f}px, "
                         f"median={np.median(dists):.2f}px, max={np.max(dists):.2f}px")
        if areas:
            lines.append(f"    Area ratio (P/GT): mean={np.mean(areas):.2f}, "
                         f"median={np.median(areas):.2f}")

        # Per-sample breakdown
        det_rates = [d / t if t > 0 else 0
                     for d, t in zip(ms["detected"], ms["total_gt"])]
        n_perfect = sum(1 for r in det_rates if r >= 1.0)
        n_partial = sum(1 for r in det_rates if 0 < r < 1.0)
        n_miss = sum(1 for r in det_rates if r == 0)
        lines.append(f"    Per-sample:        {n_perfect} perfect, "
                     f"{n_partial} partial, {n_miss} total miss")

    lines.append("\n" + "─" * 70)
    lines.append("  NON-EDDY SAMPLES (hallucinated / false positives)")
    lines.append("─" * 70)
    
    lines.append(f"\n  GP:")
    lines.append(f"    Hallucinated eddies: {gp_noeddy_fp} across "
                 f"{len(gp_noeddy_fp_details)} samples (of {n_noeddy_samples})")
    if gp_noeddy_fp_details:
        for vi, cnt in sorted(gp_noeddy_fp_details):
            lines.append(f"      val{vi}: {cnt} fake eddy(s)")

    lines.append(f"\n  DDPM S6:")
    lines.append(f"    Hallucinated eddies: {ddpm_noeddy_fp} across "
                 f"{len(ddpm_noeddy_fp_details)} samples (of {n_noeddy_samples})")
    if ddpm_noeddy_fp_details:
        for vi, cnt in sorted(ddpm_noeddy_fp_details):
            lines.append(f"      val{vi}: {cnt} fake eddy(s)")

    # ── Combined FP stats ────────────────────────────────────────
    lines.append("\n" + "─" * 70)
    lines.append("  COMBINED FALSE POSITIVE SUMMARY (eddy + non-eddy)")
    lines.append("─" * 70)
    
    gp_total_fp = sum(gp_stats["false_pos"]) + gp_noeddy_fp
    ddpm_total_fp = sum(ddpm_stats["false_pos"]) + ddpm_noeddy_fp
    lines.append(f"  GP:       {gp_total_fp} total FP across all 100 samples "
                 f"({gp_total_fp/100:.2f}/sample)")
    lines.append(f"  DDPM S6:  {ddpm_total_fp} total FP across all 100 samples "
                 f"({ddpm_total_fp/100:.2f}/sample)")

    # ── Precision / Recall ───────────────────────────────────────
    lines.append("\n" + "─" * 70)
    lines.append("  PRECISION & RECALL (all 100 samples)")
    lines.append("─" * 70)

    for label, det, fp_eddy, fp_noeddy in [
        ("GP", sum(gp_stats["detected"]), sum(gp_stats["false_pos"]), gp_noeddy_fp),
        ("DDPM S6", sum(ddpm_stats["detected"]), sum(ddpm_stats["false_pos"]), ddpm_noeddy_fp),
    ]:
        tp = det
        fp = fp_eddy + fp_noeddy
        fn = total_gt - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        lines.append(f"\n  {label}:")
        lines.append(f"    TP={tp}  FP={fp}  FN={fn}")
        lines.append(f"    Precision: {precision:.1%}")
        lines.append(f"    Recall:    {recall:.1%}")
        lines.append(f"    F1 score:  {f1:.3f}")

    # ── Distance distribution ────────────────────────────────────
    lines.append("\n" + "─" * 70)
    lines.append("  CENTER DISTANCE DISTRIBUTION (DDPM matched eddies)")
    lines.append("─" * 70)
    dists = ddpm_stats["distances"]
    if dists:
        buckets = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10)]
        for lo, hi in buckets:
            n = sum(1 for d in dists if lo <= d < hi)
            lines.append(f"    {lo}-{hi}px: {n} ({100*n/len(dists):.0f}%)")

    lines.append("\n" + "=" * 70)

    summary = "\n".join(lines)
    print(f"\n{summary}")

    # Save
    out_pt = os.path.join(OUT_DIR, "eddy_eval_all100_results.pt")
    torch.save({
        "eddy_results": eddy_results,
        "noeddy_results": noeddy_results,
        "gp_stats": dict(gp_stats),
        "ddpm_stats": dict(ddpm_stats),
        "gp_noeddy_fp": gp_noeddy_fp,
        "ddpm_noeddy_fp": ddpm_noeddy_fp,
        "gp_noeddy_fp_details": gp_noeddy_fp_details,
        "ddpm_noeddy_fp_details": ddpm_noeddy_fp_details,
    }, out_pt)

    out_txt = os.path.join(OUT_DIR, "eddy_eval_all100_summary.txt")
    with open(out_txt, "w") as f:
        f.write(summary + "\n")

    print(f"\nSaved → {out_pt}")
    print(f"Saved → {out_txt}")
    print(f"Elapsed: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
