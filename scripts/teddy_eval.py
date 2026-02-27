#!/usr/bin/env python3
"""
Evaluate the trained Teddy baseline on the same 100 balanced eddy-eval samples
used for DDPM/GP comparison.

Usage:
    PYTHONPATH=. python scripts/teddy_eval.py [--checkpoint results/teddy_baseline/teddy_best.pt]

Loads the bulk_eval .pt, extracts observations from ground truth, runs Teddy,
extracts predicted eddy centres, and matches against GT.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

from ddpm.utils.eddy_detection import detect_eddies_gamma, Eddy
from scripts.teddy_model import TeddyNet

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OCEAN_H, OCEAN_W = 44, 94
OBS_ROW = 22
MATCH_DIST = 5.0  # pixels — same threshold as bulk eval

EDDY_PARAMS = dict(
    radius=8, gamma_threshold=0.65, min_area=25, shore_buffer=2,
    smooth_sigma=2.0, min_mean_speed_ratio=0.3, min_vorticity=0.03,
)

BULK_EVAL_PATH = "results/eddy_balanced_eval/bulk_eval_eddy_balanced_100.pt"


# ---------------------------------------------------------------------------
# Connected-component extraction (numpy-only, no scipy needed)
# ---------------------------------------------------------------------------

def connected_components(binary: np.ndarray):
    """4-connected components via BFS. Returns (labels, n_components)."""
    H, W = binary.shape
    labels = np.zeros((H, W), dtype=np.int32)
    current = 0
    for i in range(H):
        for j in range(W):
            if binary[i, j] and labels[i, j] == 0:
                current += 1
                queue = [(i, j)]
                labels[i, j] = current
                head = 0
                while head < len(queue):
                    y, x = queue[head]; head += 1
                    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < H and 0 <= nx < W and binary[ny, nx] and labels[ny, nx] == 0:
                            labels[ny, nx] = current
                            queue.append((ny, nx))
    return labels, current


def extract_eddy_centres(prob_map: np.ndarray, threshold: float = 0.5,
                         min_area: int = 25):
    """Threshold probability map and extract eddy centres.

    Returns list of (cy, cx, area) tuples.
    """
    binary = prob_map > threshold
    labels, n = connected_components(binary)
    centres = []
    for k in range(1, n + 1):
        ys, xs = np.where(labels == k)
        area = len(ys)
        if area >= min_area:
            centres.append((ys.mean(), xs.mean(), area))
    return centres


# ---------------------------------------------------------------------------
# Eddy matching (same logic as bulk eval)
# ---------------------------------------------------------------------------

def match_eddies(gt_centres, pred_centres, max_dist=MATCH_DIST):
    """Match predicted to GT centres. Returns (tp, fp, fn, match_dists)."""
    gt_matched = [False] * len(gt_centres)
    match_dists = []
    tp = 0
    for pcy, pcx, _ in pred_centres:
        best_d, best_j = float("inf"), -1
        for j, (gcy, gcx, _) in enumerate(gt_centres):
            d = ((pcy - gcy) ** 2 + (pcx - gcx) ** 2) ** 0.5
            if d < best_d:
                best_d, best_j = d, j
        if best_d <= max_dist and not gt_matched[best_j]:
            gt_matched[best_j] = True
            tp += 1
            match_dists.append(best_d)
    fp = len(pred_centres) - tp
    fn = len(gt_centres) - tp
    return tp, fp, fn, match_dists


def gt_centres_from_eddies(eddies):
    """Convert list[Eddy] → list[(cy, cx, area)]."""
    return [(e.center_y, e.center_x, e.area_pixels) for e in eddies]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(args):
    # ----- load model -----
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    obs_mean = ckpt["obs_mean"]
    obs_std = ckpt["obs_std"]

    model = TeddyNet(
        obs_dim=2, d_model=64, n_heads=4, n_enc_layers=3,
        n_cnn_layers=8, ocean_h=OCEAN_H, ocean_w=OCEAN_W, obs_row=OBS_ROW,
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded model from {args.checkpoint} (epoch {ckpt['epoch']}, "
          f"val_loss={ckpt['val_loss']:.4f})")

    # ----- load bulk eval data -----
    data = torch.load(BULK_EVAL_PATH, map_location="cpu", weights_only=False)
    samples = data["samples"]
    print(f"Loaded {len(samples)} evaluation samples from {BULK_EVAL_PATH}")

    # ----- run evaluation -----
    total_tp = total_fp = total_fn = 0
    total_gt_eddies = 0
    n_eddy_samples = 0
    all_match_dists = []
    hallucinations = 0  # FP on non-eddy samples

    results = []

    for i, s in enumerate(samples):
        gt_4d = s["ground_truth"]               # (1, 2, 64, 128)
        gt_vel = gt_4d.squeeze(0)[:, :OCEAN_H, :OCEAN_W]  # (2, 44, 94)
        gt_vel = torch.nan_to_num(gt_vel, nan=0.0)

        # Ground-truth eddies
        ocean = (gt_vel[0] ** 2 + gt_vel[1] ** 2 > 1e-10)
        gt_eddies, _, _ = detect_eddies_gamma(gt_vel, ocean_mask=ocean, **EDDY_PARAMS)
        gt_cents = gt_centres_from_eddies(gt_eddies)

        # Observation input for Teddy
        obs = gt_vel[:, OBS_ROW, :].T.unsqueeze(0)  # (1, W, 2)
        obs = (obs - obs_mean) / obs_std

        with torch.no_grad():
            logits = model(obs)                # (1, 1, H, W)
            prob = torch.sigmoid(logits).squeeze().cpu().numpy()  # (H, W)

        # Extract predicted eddy centres
        pred_cents = extract_eddy_centres(prob, threshold=0.5, min_area=EDDY_PARAMS["min_area"])

        # Match
        is_eddy = s["is_eddy_sample"]
        if gt_cents:
            tp, fp, fn, dists = match_eddies(gt_cents, pred_cents)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_gt_eddies += len(gt_cents)
            all_match_dists.extend(dists)
            n_eddy_samples += 1
        else:
            # Non-eddy sample: any prediction is a false positive / hallucination
            fp = len(pred_cents)
            total_fp += fp
            if fp > 0:
                hallucinations += 1

        results.append({
            "idx": s["idx"],
            "val_idx": s["val_idx"],
            "is_eddy": is_eddy,
            "n_gt": len(gt_cents),
            "n_pred": len(pred_cents),
            "tp": tp if gt_cents else 0,
            "fp": fp,
        })

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(samples)}] TP={total_tp} FP={total_fp} FN={total_fn}")

    # ----- print summary -----
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / total_gt_eddies if total_gt_eddies > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print("\n" + "=" * 60)
    print("TEDDY BASELINE — Eddy Detection Results")
    print("=" * 60)
    print(f"Total GT eddies:        {total_gt_eddies}")
    print(f"True Positives:         {total_tp}")
    print(f"False Positives:        {total_fp}")
    print(f"False Negatives:        {total_fn}")
    print(f"Precision:              {precision:.3f}  ({total_tp}/{total_tp+total_fp})")
    print(f"Recall:                 {recall:.3f}  ({total_tp}/{total_gt_eddies})")
    print(f"F1 Score:               {f1:.3f}")
    print(f"Hallucinations (FP on non-eddy samples): {hallucinations}")
    if all_match_dists:
        print(f"Mean match distance:    {np.mean(all_match_dists):.1f} px")
        print(f"Matches within 4 px:    {sum(d <= 4 for d in all_match_dists)}/{len(all_match_dists)}")

    print("\n--- Comparison ---")
    print(f"{'Method':<12} {'TP':>4} {'FP':>4} {'FN':>4} {'Prec':>6} {'Recall':>6} {'F1':>6}")
    print(f"{'Teddy':<12} {total_tp:>4} {total_fp:>4} {total_fn:>4} "
          f"{precision:>6.3f} {recall:>6.3f} {f1:>6.3f}")
    print(f"{'DDPM':<12} {'15':>4} {'4':>4} {'42':>4} "
          f"{'0.789':>6} {'0.263':>6} {'0.395':>6}")
    print(f"{'GP':<12} {'1':>4} {'0':>4} {'56':>4} "
          f"{'1.000':>6} {'0.018':>6} {'0.034':>6}")

    # Save results
    out_path = Path("results/teddy_baseline/teddy_eval_100.pt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "results": results,
        "metrics": {"tp": total_tp, "fp": total_fp, "fn": total_fn,
                     "precision": precision, "recall": recall, "f1": f1,
                     "hallucinations": hallucinations},
        "match_dists": all_match_dists,
        "checkpoint": args.checkpoint,
    }, out_path)
    print(f"\nResults saved to {out_path}")


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate Teddy eddy baseline")
    parser.add_argument("--checkpoint", type=str,
                        default="results/teddy_baseline/teddy_best.pt")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
