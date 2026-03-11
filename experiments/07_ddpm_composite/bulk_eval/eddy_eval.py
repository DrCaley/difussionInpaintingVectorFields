#!/usr/bin/env python3
"""
Eddy detection evaluation on saved bulk-eval tensors.

For each sample, runs Gamma-1 eddy detection on ground truth and each
reconstruction method, then matches predicted eddies to GT eddies and
computes precision / recall / F1 plus eddy-region MSE metrics.

Usage:
    PYTHONPATH=. python experiments/07_ddpm_composite/bulk_eval/eddy_eval.py \
        experiments/07_ddpm_composite/bulk_eval/results/0.1pct_n100_tc200_s6
"""

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE_DIR))

from ddpm.utils.eddy_detection import detect_eddies_gamma, Eddy

# ---------------------------------------------------------------------------
# Eddy detection parameters (same as eddy_compare_composite_vcnn.py)
# ---------------------------------------------------------------------------
EDDY_PARAMS = dict(
    radius=8,
    gamma_threshold=0.65,
    min_area=25,
    shore_buffer=2,
    smooth_sigma=2.0,
    min_mean_speed_ratio=0.3,
    min_vorticity=0.03,
)

METHODS = ["gp", "vcnn", "composite", "gpdiff"]
METHOD_LABELS = {"gp": "GP", "vcnn": "V-CNN", "composite": "Composite", "gpdiff": "GP-Diff(S6)"}


# ---------------------------------------------------------------------------
# Eddy matching (greedy nearest-center)
# ---------------------------------------------------------------------------

def match_eddies(gt_eddies, pred_eddies, dist_thresh=8.0):
    """Greedy nearest-center matching.  Returns (matches, fn_indices, fp_indices)."""
    if not gt_eddies or not pred_eddies:
        return [], list(range(len(gt_eddies))), list(range(len(pred_eddies)))

    n_gt, n_pred = len(gt_eddies), len(pred_eddies)
    dist = np.zeros((n_gt, n_pred))
    for i, ge in enumerate(gt_eddies):
        for j, pe in enumerate(pred_eddies):
            dist[i, j] = np.sqrt((ge.center_y - pe.center_y) ** 2 +
                                  (ge.center_x - pe.center_x) ** 2)

    matches, used_gt, used_pred = [], set(), set()
    for _ in range(min(n_gt, n_pred)):
        best_d, bi, bj = float("inf"), -1, -1
        for i in range(n_gt):
            if i in used_gt:
                continue
            for j in range(n_pred):
                if j in used_pred:
                    continue
                if dist[i, j] < best_d:
                    best_d, bi, bj = dist[i, j], i, j
        if best_d <= dist_thresh:
            matches.append((bi, bj, best_d))
            used_gt.add(bi)
            used_pred.add(bj)
        else:
            break

    fn_list = [i for i in range(n_gt) if i not in used_gt]
    fp_list = [j for j in range(n_pred) if j not in used_pred]
    return matches, fn_list, fp_list


# ---------------------------------------------------------------------------
# Per-sample evaluation
# ---------------------------------------------------------------------------

def detect(vel, ocean_mask):
    """Run Gamma1 eddy detection, returning list of Eddy."""
    vel = torch.nan_to_num(vel.float(), nan=0.0)
    om = ocean_mask.bool() if ocean_mask is not None else None
    eddies, _, _ = detect_eddies_gamma(vel, ocean_mask=om, **EDDY_PARAMS)
    return eddies


def eddy_region_mse(gt_vel, pred_vel, gt_eddies, ocean_mask):
    """MSE of pred vs GT inside GT-eddy regions and outside them."""
    H, W = gt_vel.shape[1], gt_vel.shape[2]
    eddy_mask = torch.zeros(H, W, dtype=torch.bool)
    for e in gt_eddies:
        if e.mask is not None:
            eddy_mask |= e.mask.cpu().bool()

    ocean = ocean_mask.bool().cpu()
    in_eddy = eddy_mask & ocean
    out_eddy = (~eddy_mask) & ocean

    diff = (gt_vel.cpu() - pred_vel.cpu()) ** 2
    mse_in = diff[:, in_eddy].mean().item() if in_eddy.sum() > 0 else float("nan")
    mse_out = diff[:, out_eddy].mean().item() if out_eddy.sum() > 0 else float("nan")
    return mse_in, mse_out


def eval_sample(d):
    """Evaluate one sample dict (from tensors.pt).  Returns dict of metrics per method."""
    gt = torch.from_numpy(d["gt"]) if isinstance(d["gt"], np.ndarray) else d["gt"]
    om = torch.from_numpy(d["ocean_mask"]) if isinstance(d["ocean_mask"], np.ndarray) else d["ocean_mask"]

    gt_eddies = detect(gt, om)
    n_gt = len(gt_eddies)

    results = {"n_gt_eddies": n_gt, "val_idx": d["val_idx"]}

    for method in METHODS:
        key = "gp_mean" if method == "gp" else method
        pred = d[key]
        pred = torch.from_numpy(pred) if isinstance(pred, np.ndarray) else pred

        pred_eddies = detect(pred, om)
        matches, fn_list, fp_list = match_eddies(gt_eddies, pred_eddies, dist_thresh=8.0)

        tp = len(matches)
        fp = len(fp_list)
        fn = len(fn_list)
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else float("nan")

        mean_dist = np.mean([m[2] for m in matches]) if matches else float("nan")

        mse_in, mse_out = eddy_region_mse(gt, pred, gt_eddies, om)

        results[f"{method}_n_pred"] = len(pred_eddies)
        results[f"{method}_tp"] = tp
        results[f"{method}_fp"] = fp
        results[f"{method}_fn"] = fn
        results[f"{method}_precision"] = precision
        results[f"{method}_recall"] = recall
        results[f"{method}_f1"] = f1
        results[f"{method}_center_dist"] = mean_dist
        results[f"{method}_eddy_mse"] = mse_in
        results[f"{method}_noneddy_mse"] = mse_out

    return results


# ---------------------------------------------------------------------------
# Aggregate and print
# ---------------------------------------------------------------------------

def aggregate(records):
    """Aggregate per-sample records into summary stats."""
    # Filter to samples that have at least 1 GT eddy for detection metrics
    has_eddy = [r for r in records if r["n_gt_eddies"] > 0]
    n_total = len(records)
    n_eddy = len(has_eddy)
    n_no_eddy = n_total - n_eddy

    print(f"\n{'='*72}")
    print(f"  Eddy Detection Evaluation  ({n_total} samples, {n_eddy} with GT eddies)")
    print(f"{'='*72}")

    # GT eddy stats
    gt_counts = [r["n_gt_eddies"] for r in records]
    print(f"\n  GT eddies: total={sum(gt_counts)}, "
          f"samples_with_eddies={n_eddy}/{n_total}, "
          f"avg/sample={np.mean(gt_counts):.2f}")

    # Per-method summary
    print(f"\n  {'Method':>14}  {'Prec':>6}  {'Recall':>6}  {'F1':>6}  "
          f"{'TP':>4}  {'FP':>4}  {'FN':>4}  {'CtrDist':>7}  "
          f"{'EddyMSE':>9}  {'NonEddyMSE':>10}")
    print(f"  {'-'*14}  {'-'*6}  {'-'*6}  {'-'*6}  "
          f"{'-'*4}  {'-'*4}  {'-'*4}  {'-'*7}  "
          f"{'-'*9}  {'-'*10}")

    summary = {}
    for method in METHODS:
        label = METHOD_LABELS[method]

        # TP/FP/FN totalled across all samples (micro-average)
        total_tp = sum(r[f"{method}_tp"] for r in records)
        total_fp = sum(r[f"{method}_fp"] for r in records)
        total_fn = sum(r[f"{method}_fn"] for r in records)
        micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        micro_f1 = (2 * micro_prec * micro_recall / (micro_prec + micro_recall)
                     if (micro_prec + micro_recall) > 0 else 0)

        # Mean centre distance (over matched eddies)
        dists = [r[f"{method}_center_dist"] for r in records
                 if not np.isnan(r[f"{method}_center_dist"])]
        mean_dist = np.mean(dists) if dists else float("nan")

        # Eddy-region MSE (average over samples that have GT eddies)
        eddy_mses = [r[f"{method}_eddy_mse"] for r in has_eddy
                     if not np.isnan(r[f"{method}_eddy_mse"])]
        noneddy_mses = [r[f"{method}_noneddy_mse"] for r in has_eddy
                        if not np.isnan(r[f"{method}_noneddy_mse"])]
        avg_eddy_mse = np.mean(eddy_mses) if eddy_mses else float("nan")
        avg_noneddy_mse = np.mean(noneddy_mses) if noneddy_mses else float("nan")

        print(f"  {label:>14}  {micro_prec:6.3f}  {micro_recall:6.3f}  {micro_f1:6.3f}  "
              f"{total_tp:4d}  {total_fp:4d}  {total_fn:4d}  {mean_dist:7.2f}  "
              f"{avg_eddy_mse:9.6f}  {avg_noneddy_mse:10.6f}")

        summary[method] = {
            "precision": micro_prec, "recall": micro_recall, "f1": micro_f1,
            "tp": total_tp, "fp": total_fp, "fn": total_fn,
            "center_dist": mean_dist,
            "eddy_mse": avg_eddy_mse, "noneddy_mse": avg_noneddy_mse,
        }

    # False positive rate for no-eddy samples
    print(f"\n  False positives on {n_no_eddy} samples with no GT eddies:")
    for method in METHODS:
        label = METHOD_LABELS[method]
        fps_no_eddy = [r[f"{method}_n_pred"] for r in records if r["n_gt_eddies"] == 0]
        total_fp_no = sum(fps_no_eddy)
        avg_fp_no = np.mean(fps_no_eddy) if fps_no_eddy else 0
        print(f"    {label:>14}: total={total_fp_no}, avg/sample={avg_fp_no:.2f}")

    print(f"{'='*72}\n")
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Eddy detection evaluation on bulk eval results")
    parser.add_argument("results_dir", type=str, help="Path to results dir with sample_* folders")
    parser.add_argument("--dist-thresh", type=float, default=8.0,
                        help="Max center distance for eddy matching (default: 8 pixels)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    sample_dirs = sorted(results_dir.glob("sample_*"))
    if not sample_dirs:
        print(f"No sample_* dirs found in {results_dir}")
        sys.exit(1)

    print(f"Running eddy detection on {len(sample_dirs)} samples from {results_dir.name}")
    print(f"Eddy params: {EDDY_PARAMS}")
    print(f"Match distance threshold: {args.dist_thresh} pixels")

    t0 = time.time()
    records = []
    for i, sd in enumerate(sample_dirs):
        d = torch.load(sd / "tensors.pt", map_location="cpu", weights_only=False)
        rec = eval_sample(d)
        records.append(rec)
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(sample_dirs)} done ({elapsed:.0f}s)")

    elapsed = time.time() - t0
    print(f"\nAll {len(sample_dirs)} samples processed in {elapsed:.0f}s")

    summary = aggregate(records)

    # Save results
    out_path = results_dir / "eddy_eval.pt"
    torch.save({
        "records": records,
        "summary": summary,
        "eddy_params": EDDY_PARAMS,
        "dist_thresh": args.dist_thresh,
        "elapsed_s": elapsed,
    }, out_path)
    print(f"Saved eddy evaluation to {out_path}")


if __name__ == "__main__":
    main()
