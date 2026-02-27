#!/usr/bin/env python3
"""
Evaluate eddy detection quality: compare GT eddies with GP and DDPM predictions.

Uses the existing bulk_eval_best .pt file (100 samples with GT, GP, DDPM outputs).
For each sample that has eddies in the ground truth, runs Gamma1 detection on all
three fields and computes:
  - Detection rate (did GP/DDPM find the eddy?)
  - Center distance (Euclidean pixels between GT and nearest predicted eddy center)
  - Area ratio (predicted area / GT area)
  - False positive count

Visualisations use the standard plotting infrastructure from
``plots/visualization_tools/standard_plots.py``.

Outputs:
    results/eddy_eval_results.pt         — full results dict
    results/eddy_eval_summary.txt        — human-readable summary
    results/eddy_comparison_plots/       — per-sample quiver panels (--plot)
    results/eddy_eval_bar_detection.png  — detection-rate bar chart
    results/eddy_eval_bar_distance.png   — center-distance bar chart

Usage:
    PYTHONPATH=. python3 scripts/eval_eddy_comparison.py
    PYTHONPATH=. python3 scripts/eval_eddy_comparison.py --plot
"""
import sys, os, argparse, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")

import torch
import numpy as np
from collections import defaultdict

from ddpm.utils.eddy_detection import detect_eddies_gamma
from plots.visualization_tools.standard_plots import (
    plot_inpaint_panels,
)

# ── Args ─────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--bulk-pt", type=str,
                    default="experiments/02_inpaint_algorithm/"
                            "repaint_gaussian_attn/results/"
                            "bulk_eval_best_100samples.pt",
                    help="Path to bulk eval .pt file")
parser.add_argument("--catalogue", type=str,
                    default="results/val_eddy_catalogue.pt",
                    help="Path to validation eddy catalogue")
parser.add_argument("--distance-threshold", type=float, default=10.0,
                    help="Max center distance (px) to count as a match")
args = parser.parse_args()

# ── Gamma1 params (same as calibration) ──────────────────────────
RADIUS = 8
GAMMA_THRESH = 0.65
MIN_AREA = 25
SHORE_BUFFER = 2
SMOOTH_SIGMA = 2.0
MIN_SPEED_RATIO = 0.3
MIN_VORTICITY = 0.03

# Ocean region in the padded 64×128 grid
OCEAN_H, OCEAN_W = 44, 94

OUT_DIR = "results"
PLOT_DIR = os.path.join(OUT_DIR, "eddy_comparison_plots")
os.makedirs(OUT_DIR, exist_ok=True)


def crop_to_ocean(tensor_4d):
    """Crop (1, 2, 64, 128) → (2, 44, 94) velocity field."""
    return tensor_4d.squeeze(0)[:, :OCEAN_H, :OCEAN_W]


def run_gamma1(vel):
    """Run Gamma1 detection, return list of eddies."""
    vel = torch.nan_to_num(vel, nan=0.0)
    eddies, g1, omega = detect_eddies_gamma(
        vel,
        radius=RADIUS,
        gamma_threshold=GAMMA_THRESH,
        min_area=MIN_AREA,
        shore_buffer=SHORE_BUFFER,
        smooth_sigma=SMOOTH_SIGMA,
        min_mean_speed_ratio=MIN_SPEED_RATIO,
        min_vorticity=MIN_VORTICITY,
    )
    return eddies


def match_eddies(gt_eddies, pred_eddies, dist_thresh):
    """
    Greedy matching of GT eddies to predicted eddies.

    Returns:
        matches: list of (gt_idx, pred_idx, distance)
        unmatched_gt: list of gt_idx (missed eddies / false negatives)
        unmatched_pred: list of pred_idx (false positives)
    """
    if not gt_eddies or not pred_eddies:
        return (
            [],
            list(range(len(gt_eddies))),
            list(range(len(pred_eddies))),
        )

    n_gt, n_pred = len(gt_eddies), len(pred_eddies)
    dist = np.zeros((n_gt, n_pred))
    for i, ge in enumerate(gt_eddies):
        for j, pe in enumerate(pred_eddies):
            dy = ge.center_y - pe.center_y
            dx = ge.center_x - pe.center_x
            dist[i, j] = np.sqrt(dy**2 + dx**2)

    matches = []
    used_gt, used_pred = set(), set()

    for _ in range(min(n_gt, n_pred)):
        best_d = float("inf")
        best_i, best_j = -1, -1
        for i in range(n_gt):
            if i in used_gt:
                continue
            for j in range(n_pred):
                if j in used_pred:
                    continue
                if dist[i, j] < best_d:
                    best_d = dist[i, j]
                    best_i, best_j = i, j

        if best_d <= dist_thresh:
            matches.append((best_i, best_j, best_d))
            used_gt.add(best_i)
            used_pred.add(best_j)
        else:
            break

    unmatched_gt = [i for i in range(n_gt) if i not in used_gt]
    unmatched_pred = [j for j in range(n_pred) if j not in used_pred]
    return matches, unmatched_gt, unmatched_pred


def eddy_info_str(eddies):
    """Short string describing eddies."""
    if not eddies:
        return "none"
    parts = []
    for e in eddies:
        cyc = "C" if e.is_cyclonic else "A"
        parts.append(f"{cyc}@({e.center_y:.0f},{e.center_x:.0f}) a={e.area_pixels}")
    return "; ".join(parts)


def main():
    t0 = time.time()

    # Load data
    print(f"Loading bulk eval: {args.bulk_pt}")
    bulk = torch.load(args.bulk_pt, map_location="cpu", weights_only=False)
    samples = bulk["samples"]
    print(f"  {len(samples)} samples loaded")

    print(f"Loading eddy catalogue: {args.catalogue}")
    catalogue = torch.load(args.catalogue, map_location="cpu", weights_only=False)
    eddy_set = set(catalogue["eddy_indices"])
    print(f"  {len(eddy_set)} val indices with eddies")

    # Filter to eddy-containing samples
    eddy_samples = [s for s in samples if s["val_idx"] in eddy_set]
    print(f"  {len(eddy_samples)} bulk eval samples have GT eddies\n")

    # ── Run detection and matching ────────────────────────────────
    results = []
    stats = {"gp": defaultdict(list), "ddpm": defaultdict(list)}

    print(f"{'#':>3} {'ValIdx':>6}  {'GT':>12}  {'GP':>12}  {'DDPM':>12}  "
          f"{'GP_dist':>8}  {'DDPM_dist':>8}  {'GP_det':>6}  {'DDPM_det':>7}")
    print("-" * 100)

    for i, s in enumerate(eddy_samples):
        val_idx = s["val_idx"]

        gt_vel = crop_to_ocean(s["ground_truth"])
        gp_vel = crop_to_ocean(s["gp_output"])
        ddpm_vel = crop_to_ocean(s["ddpm_output"])

        gt_eddies = run_gamma1(gt_vel)
        gp_eddies = run_gamma1(gp_vel)
        ddpm_eddies = run_gamma1(ddpm_vel)

        gp_matches, gp_fn, gp_fp = match_eddies(
            gt_eddies, gp_eddies, args.distance_threshold)
        ddpm_matches, ddpm_fn, ddpm_fp = match_eddies(
            gt_eddies, ddpm_eddies, args.distance_threshold)

        n_gt = len(gt_eddies)
        gp_detected = len(gp_matches)
        ddpm_detected = len(ddpm_matches)

        gp_dists = [m[2] for m in gp_matches]
        ddpm_dists = [m[2] for m in ddpm_matches]

        gp_area_ratios = [
            gp_eddies[m[1]].area_pixels / gt_eddies[m[0]].area_pixels
            for m in gp_matches
        ]
        ddpm_area_ratios = [
            ddpm_eddies[m[1]].area_pixels / gt_eddies[m[0]].area_pixels
            for m in ddpm_matches
        ]

        gp_mean_dist = np.mean(gp_dists) if gp_dists else float("nan")
        ddpm_mean_dist = np.mean(ddpm_dists) if ddpm_dists else float("nan")
        gp_det_rate = gp_detected / n_gt if n_gt > 0 else 0
        ddpm_det_rate = ddpm_detected / n_gt if n_gt > 0 else 0

        result = {
            "val_idx": val_idx,
            "n_gt_eddies": n_gt,
            "gt_eddies": [(e.center_y, e.center_x, e.area_pixels, e.is_cyclonic)
                          for e in gt_eddies],
            "gp_n_detected": gp_detected,
            "gp_n_false_pos": len(gp_fp),
            "gp_distances": gp_dists,
            "gp_area_ratios": gp_area_ratios,
            "gp_mean_distance": gp_mean_dist,
            "gp_detection_rate": gp_det_rate,
            "ddpm_n_detected": ddpm_detected,
            "ddpm_n_false_pos": len(ddpm_fp),
            "ddpm_distances": ddpm_dists,
            "ddpm_area_ratios": ddpm_area_ratios,
            "ddpm_mean_distance": ddpm_mean_dist,
            "ddpm_detection_rate": ddpm_det_rate,
        }
        results.append(result)

        # Accumulate
        for key, method_stats, matches, fp, dists, area_ratios in [
            ("gp", stats["gp"], gp_matches, gp_fp, gp_dists, gp_area_ratios),
            ("ddpm", stats["ddpm"], ddpm_matches, ddpm_fp, ddpm_dists, ddpm_area_ratios),
        ]:
            method_stats["detected"].append(len(matches))
            method_stats["total_gt"].append(n_gt)
            method_stats["false_pos"].append(len(fp))
            method_stats["distances"].extend(dists)
            method_stats["area_ratios"].extend(area_ratios)

        # Print row
        gt_str = eddy_info_str(gt_eddies)[:12]
        gp_str = f"{gp_detected}/{n_gt}+{len(gp_fp)}fp"
        ddpm_str = f"{ddpm_detected}/{n_gt}+{len(ddpm_fp)}fp"
        gp_d = f"{gp_mean_dist:.1f}" if not np.isnan(gp_mean_dist) else "MISS"
        ddpm_d = f"{ddpm_mean_dist:.1f}" if not np.isnan(ddpm_mean_dist) else "MISS"

        print(f"{i+1:>3} {val_idx:>6}  {gt_str:>12}  {gp_str:>12}  {ddpm_str:>12}  "
              f"{gp_d:>8}  {ddpm_d:>8}  {gp_det_rate:>5.0%}  {ddpm_det_rate:>6.0%}")

    # ── Summary statistics ────────────────────────────────────────
    total_gt_eddies = sum(stats["gp"]["total_gt"])

    lines = []
    lines.append("=" * 70)
    lines.append("EDDY DETECTION EVALUATION SUMMARY")
    lines.append(f"  Samples evaluated: {len(eddy_samples)}")
    lines.append(f"  Total GT eddies:   {total_gt_eddies}")
    lines.append(f"  Match threshold:   {args.distance_threshold} px")
    lines.append(f"  Gamma1 params:     radius={RADIUS}, thresh={GAMMA_THRESH}, "
                 f"min_area={MIN_AREA}, min_speed_ratio={MIN_SPEED_RATIO}")
    lines.append("=" * 70)

    for method, label in [("gp", "GP"), ("ddpm", "DDPM S6 (GP-init)")]:
        s = stats[method]
        total_det = sum(s["detected"])
        total_fp = sum(s["false_pos"])
        det_rate = total_det / total_gt_eddies if total_gt_eddies > 0 else 0
        dists = s["distances"]
        area_rs = s["area_ratios"]

        lines.append(f"\n  {label}:")
        lines.append(f"    Detection rate:     {total_det}/{total_gt_eddies} = {det_rate:.1%}")
        lines.append(f"    False positives:    {total_fp} total "
                     f"({total_fp/len(eddy_samples):.1f}/sample)")
        if dists:
            lines.append(f"    Center distance:    mean={np.mean(dists):.2f} px, "
                         f"median={np.median(dists):.2f} px, "
                         f"max={np.max(dists):.2f} px")
        else:
            lines.append(f"    Center distance:    N/A (no matches)")
        if area_rs:
            lines.append(f"    Area ratio (P/GT):  mean={np.mean(area_rs):.2f}, "
                         f"median={np.median(area_rs):.2f}, "
                         f"range=[{np.min(area_rs):.2f}, {np.max(area_rs):.2f}]")

        det_rates = [d / t if t > 0 else 0
                     for d, t in zip(s["detected"], s["total_gt"])]
        n_perfect = sum(1 for r in det_rates if r >= 1.0)
        n_partial = sum(1 for r in det_rates if 0 < r < 1.0)
        n_miss = sum(1 for r in det_rates if r == 0)
        lines.append(f"    Per-sample:         {n_perfect} perfect, "
                     f"{n_partial} partial, {n_miss} total miss "
                     f"({100*n_miss/len(eddy_samples):.0f}% miss rate)")

    lines.append("\n" + "=" * 70)

    # Distance distribution
    lines.append("\n  CENTER DISTANCE DISTRIBUTION (matched eddies):")
    for method, label in [("gp", "GP"), ("ddpm", "DDPM S6")]:
        dists = stats[method]["distances"]
        if not dists:
            continue
        buckets = [0, 2, 4, 6, 8, 10]
        lines.append(f"    {label}:")
        for lo, hi in zip(buckets, buckets[1:]):
            n = sum(1 for d in dists if lo <= d < hi)
            lines.append(f"      {lo}-{hi} px: {n} ({100*n/len(dists):.0f}%)")
        n_over = sum(1 for d in dists if d >= buckets[-1])
        lines.append(f"      ≥{buckets[-1]} px: {n_over} ({100*n_over/len(dists):.0f}%)")

    summary_text = "\n".join(lines)
    print(f"\n{summary_text}")

    # Save results
    out_pt = os.path.join(OUT_DIR, "eddy_eval_results.pt")
    torch.save({
        "results": results,
        "stats": dict(stats),
        "params": {
            "radius": RADIUS,
            "gamma_threshold": GAMMA_THRESH,
            "min_area": MIN_AREA,
            "shore_buffer": SHORE_BUFFER,
            "smooth_sigma": SMOOTH_SIGMA,
            "min_mean_speed_ratio": MIN_SPEED_RATIO,
            "distance_threshold": args.distance_threshold,
        },
        "n_eddy_samples": len(eddy_samples),
        "n_total_gt_eddies": total_gt_eddies,
    }, out_pt)

    out_txt = os.path.join(OUT_DIR, "eddy_eval_summary.txt")
    with open(out_txt, "w") as f:
        f.write(summary_text + "\n")

    print(f"\nSaved → {out_pt}")
    print(f"Saved → {out_txt}")

    # ── Per-sample quiver panels ──────────────────────────────────
    generate_sample_plots(eddy_samples, results)

    print(f"\nElapsed: {time.time()-t0:.0f}s")


# ═══════════════════════════════════════════════════════════════════
#  PER-SAMPLE PLOTS (using standard_plots.plot_inpaint_panels)
# ═══════════════════════════════════════════════════════════════════

def generate_sample_plots(eddy_samples, results):
    """Generate per-sample quiver panels using the standard plotting infra."""
    os.makedirs(PLOT_DIR, exist_ok=True)
    print(f"\nGenerating standard comparison plots → {PLOT_DIR}/")

    for i, (s, r) in enumerate(zip(eddy_samples, results)):
        val_idx = s["val_idx"]

        # Build extra titles with eddy match info
        extra_titles = {}
        n_gt = r["n_gt_eddies"]

        gp_det = r["gp_n_detected"]
        if not np.isnan(r["gp_mean_distance"]):
            extra_titles["GP"] = (
                f"({gp_det}/{n_gt} eddies, d={r['gp_mean_distance']:.1f}px)"
            )
        else:
            extra_titles["GP"] = f"({gp_det}/{n_gt} eddies)"

        ddpm_det = r["ddpm_n_detected"]
        if not np.isnan(r["ddpm_mean_distance"]):
            extra_titles["DDPM S6"] = (
                f"({ddpm_det}/{n_gt} eddies, d={r['ddpm_mean_distance']:.1f}px)"
            )
        else:
            extra_titles["DDPM S6"] = f"({ddpm_det}/{n_gt} eddies)"

        # missing_mask convention: saved as 0=missing, 1=known
        # plot_inpaint_panels expects 1=missing, so invert
        mm_inv = 1.0 - s["missing_mask"]

        plot_inpaint_panels(
            gt=s["ground_truth"],
            missing_mask=mm_inv,
            methods={
                "GP": s["gp_output"],
                "DDPM S6": s["ddpm_output"],
            },
            mse={
                "GP": s["gp_mse"],
                "DDPM S6": s["ddpm_mse"],
            },
            out_dir=PLOT_DIR,
            prefix=f"eddy_{i+1:03d}_val{val_idx}",
            extra_titles=extra_titles,
            mark_eddies=True,
            mark_eddies_on_methods=True,
            eddy_method="gamma1",
            eddy_radius=RADIUS,
            eddy_gamma_threshold=GAMMA_THRESH,
            eddy_min_area=MIN_AREA,
            eddy_shore_buffer=SHORE_BUFFER,
            eddy_smooth_sigma=SMOOTH_SIGMA,
            eddy_min_speed_ratio=MIN_SPEED_RATIO,
            eddy_min_vorticity=MIN_VORTICITY,
        )

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(eddy_samples)} samples done")

    print(f"  All {len(eddy_samples)} sample plots saved → {PLOT_DIR}/")


if __name__ == "__main__":
    main()
