#!/usr/bin/env python3
"""
Profile eddy characteristics: compare size & strength of GT eddies,
DDPM true positives, DDPM false positives (in eddy samples),
and DDPM hallucinated eddies (in non-eddy samples).
"""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
from ddpm.utils.eddy_detection import detect_eddies_gamma

BULK_PT = ("experiments/02_inpaint_algorithm/"
           "repaint_gaussian_attn/results/"
           "bulk_eval_best_100samples.pt")
DIST_THRESH = 10.0

RADIUS = 8
GAMMA_THRESH = 0.65
MIN_AREA = 25
SHORE_BUFFER = 2
SMOOTH_SIGMA = 2.0
MIN_SPEED_RATIO = 0.3
MIN_VORTICITY = 0.03
OCEAN_H, OCEAN_W = 44, 94


def crop(t):
    return t.squeeze(0)[:, :OCEAN_H, :OCEAN_W]


def detect(vel):
    vel = torch.nan_to_num(vel, nan=0.0)
    eddies, _, _ = detect_eddies_gamma(
        vel, radius=RADIUS, gamma_threshold=GAMMA_THRESH,
        min_area=MIN_AREA, shore_buffer=SHORE_BUFFER,
        smooth_sigma=SMOOTH_SIGMA, min_mean_speed_ratio=MIN_SPEED_RATIO,
        min_vorticity=MIN_VORTICITY)
    return eddies


def match(gt_eddies, pred_eddies, dist_thresh):
    if not gt_eddies or not pred_eddies:
        return [], list(range(len(gt_eddies))), list(range(len(pred_eddies)))
    n_gt, n_pred = len(gt_eddies), len(pred_eddies)
    dist = np.zeros((n_gt, n_pred))
    for i, ge in enumerate(gt_eddies):
        for j, pe in enumerate(pred_eddies):
            dist[i, j] = np.sqrt((ge.center_y - pe.center_y)**2 +
                                  (ge.center_x - pe.center_x)**2)
    matches, used_gt, used_pred = [], set(), set()
    for _ in range(min(n_gt, n_pred)):
        best_d, bi, bj = float("inf"), -1, -1
        for i in range(n_gt):
            if i in used_gt: continue
            for j in range(n_pred):
                if j in used_pred: continue
                if dist[i, j] < best_d:
                    best_d, bi, bj = dist[i, j], i, j
        if best_d <= dist_thresh:
            matches.append((bi, bj, best_d))
            used_gt.add(bi); used_pred.add(bj)
        else:
            break
    return (matches,
            [i for i in range(n_gt) if i not in used_gt],
            [j for j in range(n_pred) if j not in used_pred])


def eddy_props(e):
    """Extract a dict of properties from an Eddy."""
    # Compute speed from velocity field if available — use area and vorticity
    return {
        "area": e.area_pixels,
        "vorticity": abs(e.mean_vorticity),
        "mean_ow": e.mean_ow,
        "min_ow": e.min_ow,
        "swirl_frac": e.swirl_fraction,
        "cyclonic": e.is_cyclonic,
    }


def fmt_stats(props_list, key):
    vals = [p[key] for p in props_list]
    if not vals:
        return "N/A"
    return (f"n={len(vals):>3}  mean={np.mean(vals):>7.2f}  "
            f"med={np.median(vals):>7.2f}  "
            f"min={np.min(vals):>7.2f}  max={np.max(vals):>7.2f}")


def main():
    bulk = torch.load(BULK_PT, map_location="cpu", weights_only=False)
    samples = bulk["samples"]

    # Collect eddy properties by category
    gt_all = []          # all GT eddies
    gt_detected = []     # GT eddies that DDPM found (matched GT side)
    gt_missed = []       # GT eddies that DDPM missed
    ddpm_tp = []         # DDPM eddies that matched a GT eddy
    ddpm_fp_eddy = []    # DDPM FP in eddy-containing samples
    ddpm_fp_noeddy = []  # DDPM hallucinations in non-eddy samples
    gp_all = []          # all GP eddies (for reference)

    # Also track val_idx for FP details
    fp_details = []

    for s in samples:
        val_idx = s["val_idx"]
        gt_vel = crop(s["ground_truth"])
        ddpm_vel = crop(s["ddpm_output"])

        gt_e = detect(gt_vel)
        dd_e = detect(ddpm_vel)

        gt_props = [eddy_props(e) for e in gt_e]
        dd_props = [eddy_props(e) for e in dd_e]

        gt_all.extend(gt_props)

        if gt_e:
            ms, fn_idx, fp_idx = match(gt_e, dd_e, DIST_THRESH)
            for gi, di, dist in ms:
                gt_detected.append(gt_props[gi])
                ddpm_tp.append(dd_props[di])
            for gi in fn_idx:
                gt_missed.append(gt_props[gi])
            for di in fp_idx:
                ddpm_fp_eddy.append(dd_props[di])
                fp_details.append(("eddy", val_idx, dd_e[di]))
        else:
            for di, dp in enumerate(dd_props):
                ddpm_fp_noeddy.append(dp)
                fp_details.append(("noeddy", val_idx, dd_e[di]))

    # ── Report ───────────────────────────────────────────────────
    lines = []
    lines.append("=" * 80)
    lines.append("EDDY SIZE & STRENGTH PROFILE")
    lines.append("=" * 80)

    categories = [
        ("All GT eddies", gt_all),
        ("GT eddies DDPM detected (TP-GT side)", gt_detected),
        ("GT eddies DDPM missed (FN)", gt_missed),
        ("DDPM true positives (TP-pred side)", ddpm_tp),
        ("DDPM false pos in eddy samples", ddpm_fp_eddy),
        ("DDPM hallucinations in non-eddy samples", ddpm_fp_noeddy),
        ("ALL DDPM false positives combined", ddpm_fp_eddy + ddpm_fp_noeddy),
    ]

    metrics = [
        ("area", "Area (pixels)"),
        ("vorticity", "|Vorticity| (abs mean)"),
        ("mean_ow", "Mean Okubo-Weiss"),
        ("min_ow", "Min Okubo-Weiss (peak rotation)"),
        ("swirl_frac", "Swirl fraction"),
    ]

    for cat_name, props in categories:
        lines.append(f"\n  {cat_name}  (n={len(props)})")
        lines.append("  " + "─" * 70)
        for key, label in metrics:
            lines.append(f"    {label:40s} {fmt_stats(props, key)}")

    # ── Individual FP detail ─────────────────────────────────────
    lines.append("\n" + "=" * 80)
    lines.append("INDIVIDUAL FALSE POSITIVE DETAILS")
    lines.append("=" * 80)
    lines.append(f"  {'Type':>8} {'ValIdx':>7} {'Area':>5} {'|Vort|':>8} "
                 f"{'MeanOW':>9} {'MinOW':>9} {'Swirl':>6} {'Cyc':>4} "
                 f"{'Center':>12}")

    for typ, vi, e in fp_details:
        cyc = "C" if e.is_cyclonic else "A"
        lines.append(f"  {typ:>8} {vi:>7} {e.area_pixels:>5} "
                     f"{abs(e.mean_vorticity):>8.4f} "
                     f"{e.mean_ow:>9.5f} {e.min_ow:>9.5f} "
                     f"{e.swirl_fraction:>6.2f} {cyc:>4} "
                     f"({e.center_y:.0f},{e.center_x:.0f})")

    # ── Comparison table ─────────────────────────────────────────
    lines.append("\n" + "=" * 80)
    lines.append("QUICK COMPARISON: FP vs TP vs GT")
    lines.append("=" * 80)

    all_fp = ddpm_fp_eddy + ddpm_fp_noeddy
    for key, label in metrics:
        gt_vals = [p[key] for p in gt_all]
        tp_vals = [p[key] for p in ddpm_tp]
        fp_vals = [p[key] for p in all_fp]

        gt_m = f"{np.mean(gt_vals):.3f}" if gt_vals else "N/A"
        tp_m = f"{np.mean(tp_vals):.3f}" if tp_vals else "N/A"
        fp_m = f"{np.mean(fp_vals):.3f}" if fp_vals else "N/A"

        gt_md = f"{np.median(gt_vals):.3f}" if gt_vals else "N/A"
        tp_md = f"{np.median(tp_vals):.3f}" if tp_vals else "N/A"
        fp_md = f"{np.median(fp_vals):.3f}" if fp_vals else "N/A"

        lines.append(f"\n  {label}:")
        lines.append(f"    GT:       mean={gt_m:>8s}  median={gt_md:>8s}")
        lines.append(f"    DDPM TP:  mean={tp_m:>8s}  median={tp_md:>8s}")
        lines.append(f"    DDPM FP:  mean={fp_m:>8s}  median={fp_md:>8s}")

    summary = "\n".join(lines)
    print(summary)

    out = os.path.join("results", "eddy_fp_profile.txt")
    with open(out, "w") as f:
        f.write(summary + "\n")
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
