#!/usr/bin/env python3
"""
Evaluate Voronoi-CNN baseline on the same 100 balanced samples used for
the DDPM/GP evaluation, enabling direct comparison.

Loads the saved DDPM/GP eval .pt file to get the same sample indices,
ground truth, and masks, then runs Voronoi-CNN inference and compares.

Usage:
    PYTHONPATH=. python scripts/voronoi_cnn_eval.py [--n-samples 100]
"""

import argparse
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from scripts.voronoi_cnn_model import VoronoiCNN, build_voronoi_input
from ddpm.utils.eddy_detection import detect_eddies_gamma

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OCEAN_H, OCEAN_W = 44, 94
OBS_ROW = 22

EDDY_PARAMS = dict(
    radius=8,
    gamma_threshold=0.65,
    min_area=25,
    shore_buffer=2,
    smooth_sigma=2.0,
    min_mean_speed_ratio=0.3,
    min_vorticity=0.03,
)

DDPM_EVAL_PT = Path("results/eddy_balanced_eval/bulk_eval_eddy_balanced_100.pt")
VORONOI_CKPT = Path("results/voronoi_cnn/voronoi_cnn_best.pt")
OUT_DIR = Path("results/voronoi_cnn_eval")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_pickle_data(path: str = "data.pickle"):
    """Load raw velocity data → (train, val) as (N, 2, H, W) tensors."""
    with open(path, "rb") as f:
        train_np, val_np, _test_np = pickle.load(f)

    def to_tensor(arr):
        t = torch.from_numpy(np.ascontiguousarray(arr)).float()
        t = t.permute(3, 2, 1, 0)
        t = torch.nan_to_num(t, nan=0.0)
        return t

    return to_tensor(train_np), to_tensor(val_np)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def voronoi_infer_single(model, vel_phys: np.ndarray, mask: np.ndarray,
                         ocean_mask: np.ndarray, norm_mean: np.ndarray,
                         norm_std: np.ndarray, device: torch.device
                         ) -> torch.Tensor:
    """
    Run Voronoi-CNN inference on a single sample.

    vel_phys:  (2, H, W) physical-units velocity
    mask:      (H, W) 1=known, 0=missing
    Returns:   (2, H, W) reconstructed velocity in physical units
    """
    # Normalise velocity
    vel_n = (vel_phys - norm_mean[:, None, None]) / norm_std[:, None, None]
    vel_n *= ocean_mask[None, :, :]

    # Build Voronoi input from normalised velocity
    voronoi_in = build_voronoi_input(vel_n, mask, ocean_mask)
    voronoi_t = torch.from_numpy(voronoi_in).unsqueeze(0).to(device)

    # Forward pass
    with torch.no_grad():
        pred_n = model(voronoi_t)  # (1, 2, H, W)

    # Unnormalise
    mean_t = torch.tensor(norm_mean, dtype=torch.float32).view(1, 2, 1, 1).to(device)
    std_t = torch.tensor(norm_std, dtype=torch.float32).view(1, 2, 1, 1).to(device)
    pred_phys = pred_n * std_t + mean_t

    # Zero out land
    ocean_t = torch.from_numpy(ocean_mask).to(device).unsqueeze(0).unsqueeze(0)
    pred_phys = pred_phys * ocean_t

    return pred_phys.squeeze(0).cpu()  # (2, H, W)


# ---------------------------------------------------------------------------
# Eddy matching (same as bulk_eval)
# ---------------------------------------------------------------------------

def run_gamma1(vel):
    vel = torch.nan_to_num(vel, nan=0.0)
    eddies, _, _ = detect_eddies_gamma(vel, **EDDY_PARAMS)
    return eddies


def match_eddies(gt_eddies, pred_eddies, dist_thresh=8.0):
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
    return (matches,
            [i for i in range(n_gt) if i not in used_gt],
            [j for j in range(n_pred) if j not in used_pred])


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Load Voronoi-CNN checkpoint
    print(f"Loading model from {VORONOI_CKPT}...")
    ckpt = torch.load(VORONOI_CKPT, map_location="cpu", weights_only=False)
    cfg = ckpt["model_config"]
    model = VoronoiCNN(**cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  Loaded epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.6f}")
    print(f"  Parameters: {model.count_parameters():,}")

    norm_mean = ckpt["norm_mean"].numpy()
    norm_std = ckpt["norm_std"].numpy()
    ocean_mask = ckpt["ocean_mask"]  # (H, W) numpy

    # Load raw validation data
    _, val_vel = load_pickle_data()

    # Build row-22 observation mask (1=known, 0=missing)
    row22_mask = np.zeros((OCEAN_H, OCEAN_W), dtype=np.float32)
    row22_mask[OBS_ROW, :] = 1.0
    row22_mask *= ocean_mask

    # Load DDPM eval results to get the same sample indices
    use_ddpm_indices = DDPM_EVAL_PT.exists()
    if use_ddpm_indices:
        print(f"Loading DDPM eval indices from {DDPM_EVAL_PT}...")
        ddpm_data = torch.load(DDPM_EVAL_PT, map_location="cpu", weights_only=False)
        ddpm_samples = ddpm_data["samples"]
        val_indices = [s["val_idx"] for s in ddpm_samples]
        eddy_set = set(ddpm_data.get("eddy_indices", []))
        n_total = min(args.n_samples, len(val_indices))
        val_indices = val_indices[:n_total]
        print(f"  Using {n_total} samples from DDPM eval")
    else:
        print("No DDPM eval file found — using first N validation samples")
        n_total = min(args.n_samples, len(val_vel))
        val_indices = list(range(n_total))
        eddy_set = set()

    # Run inference
    print(f"\n{'=' * 70}")
    print(f"Running Voronoi-CNN inference on {n_total} samples")
    print(f"{'=' * 70}")

    results = []
    t0_global = time.time()

    for run_i, vi in enumerate(val_indices):
        t0 = time.time()
        gt_vel = val_vel[vi].numpy()  # (2, H, W) physical

        # Run Voronoi-CNN
        pred_vel = voronoi_infer_single(
            model, gt_vel, row22_mask, ocean_mask,
            norm_mean, norm_std, device
        )  # (2, H, W) tensor

        # MSE over ocean pixels
        gt_t = torch.from_numpy(gt_vel)
        ocean_t = torch.from_numpy(ocean_mask).bool()
        diff = (pred_vel - gt_t)[:, ocean_t]
        mse = (diff ** 2).mean().item()

        # Also compute MSE on missing-only ocean pixels
        known_t = torch.from_numpy(row22_mask).bool()
        missing_ocean = ocean_t & ~known_t  # ocean but not observed
        diff_miss = (pred_vel - gt_t)[:, missing_ocean]
        mse_missing = (diff_miss ** 2).mean().item()

        elapsed = time.time() - t0
        is_eddy = vi in eddy_set
        tag = "EDDY" if is_eddy else "clean"
        print(f"  [{run_i+1:3d}/{n_total}] val_idx={vi:5d} {tag:>5}  "
              f"MSE_all={mse:.6f}  MSE_miss={mse_missing:.6f}  ({elapsed:.2f}s)")

        results.append({
            "run_idx": run_i,
            "val_idx": vi,
            "is_eddy": is_eddy,
            "voronoi_pred": pred_vel,     # (2, H, W)
            "ground_truth": gt_t,          # (2, H, W)
            "mse_all_ocean": mse,
            "mse_missing_only": mse_missing,
        })

    elapsed_total = time.time() - t0_global
    print(f"\nInference done: {n_total} samples in {elapsed_total:.1f}s "
          f"({elapsed_total / n_total:.2f}s/sample)")

    # Save results
    save_path = OUT_DIR / "voronoi_cnn_eval_results.pt"
    torch.save({
        "results": results,
        "n_samples": len(results),
        "val_indices": val_indices,
        "eddy_set": sorted(eddy_set),
        "mask_type": "row22",
        "eddy_params": EDDY_PARAMS,
    }, save_path)
    print(f"Saved to {save_path}")

    # ------------------------------------------------------------------
    # Eddy detection evaluation
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("EDDY DETECTION EVALUATION (Gamma1)")
    print(f"{'=' * 70}\n")

    voronoi_stats = defaultdict(list)
    total_gt_eddies = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    noeddy_fp = 0

    for r in results:
        gt_vel = r["ground_truth"]
        pred_vel = r["voronoi_pred"]

        gt_eddies = run_gamma1(gt_vel)
        pred_eddies = run_gamma1(pred_vel)

        is_eddy_sample = r["is_eddy"]
        n_gt = len(gt_eddies)
        n_pred = len(pred_eddies)

        if n_gt > 0:
            matches, fn_list, fp_list = match_eddies(gt_eddies, pred_eddies)
            tp = len(matches)
            fn = len(fn_list)
            fp = len(fp_list)
        else:
            tp, fn = 0, 0
            fp = n_pred

        total_gt_eddies += n_gt
        total_tp += tp
        total_fp += fp
        total_fn += fn

        if not is_eddy_sample and n_pred > 0:
            noeddy_fp += 1

        voronoi_stats["tp"].append(tp)
        voronoi_stats["fp"].append(fp)
        voronoi_stats["fn"].append(fn)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # MSE summaries
    mse_all = [r["mse_all_ocean"] for r in results]
    mse_miss = [r["mse_missing_only"] for r in results]
    mse_eddy = [r["mse_missing_only"] for r in results if r["is_eddy"]]
    mse_clean = [r["mse_missing_only"] for r in results if not r["is_eddy"]]

    print(f"  Samples:  {len(results)} total")
    print(f"  GT eddies found:  {total_gt_eddies}")
    print(f"  TP={total_tp}  FP={total_fp}  FN={total_fn}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1:        {f1:.3f}")
    print(f"  FP on clean samples: {noeddy_fp}")
    print()
    print(f"  MSE (all ocean):  mean={np.mean(mse_all):.6f}  median={np.median(mse_all):.6f}")
    print(f"  MSE (missing):    mean={np.mean(mse_miss):.6f}  median={np.median(mse_miss):.6f}")
    if mse_eddy:
        print(f"  MSE (eddy samps): mean={np.mean(mse_eddy):.6f}")
    if mse_clean:
        print(f"  MSE (clean samps): mean={np.mean(mse_clean):.6f}")

    # Compare with GP/DDPM if available
    if use_ddpm_indices and DDPM_EVAL_PT.exists():
        print(f"\n{'=' * 70}")
        print("COMPARISON WITH GP / DDPM")
        print(f"{'=' * 70}\n")

        gp_mses = [s["gp_mse"] for s in ddpm_samples[:n_total]]
        ddpm_mses = [s["ddpm_mse"] for s in ddpm_samples[:n_total]]
        vor_mses = [r["mse_missing_only"] for r in results]

        vor_beats_gp = sum(1 for v, g in zip(vor_mses, gp_mses) if v < g)
        vor_beats_ddpm = sum(1 for v, d in zip(vor_mses, ddpm_mses) if v < d)

        print(f"  {'Method':<15} {'Mean MSE':>12} {'Median MSE':>12}")
        print(f"  {'-' * 40}")
        print(f"  {'GP':<15} {np.mean(gp_mses):>12.6f} {np.median(gp_mses):>12.6f}")
        print(f"  {'DDPM':<15} {np.mean(ddpm_mses):>12.6f} {np.median(ddpm_mses):>12.6f}")
        print(f"  {'Voronoi-CNN':<15} {np.mean(vor_mses):>12.6f} {np.median(vor_mses):>12.6f}")
        print()
        print(f"  Voronoi-CNN beats GP:   {vor_beats_gp}/{n_total} "
              f"({100 * vor_beats_gp / n_total:.1f}%)")
        print(f"  Voronoi-CNN beats DDPM: {vor_beats_ddpm}/{n_total} "
              f"({100 * vor_beats_ddpm / n_total:.1f}%)")

    print(f"\nDone. Results saved to {OUT_DIR}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Voronoi-CNN on balanced eddy/non-eddy samples"
    )
    parser.add_argument("--n-samples", type=int, default=100)
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
