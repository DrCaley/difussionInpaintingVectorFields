#!/usr/bin/env python3
"""
Evaluate GP-CNN baseline on the same 100 balanced samples used for
the DDPM/GP and Voronoi-CNN evaluations, enabling direct comparison.

Computes GP on-the-fly for each validation sample (only 100 samples,
so the ~0.3s/sample GP cost is acceptable).

Usage:
    PYTHONPATH=. python scripts/gp_cnn_eval.py [--n-samples 100]
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

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from scripts.voronoi_cnn_model import VoronoiCNN
from ddpm.helper_functions.interpolation_tool import gp_fill
from ddpm.utils.eddy_detection import detect_eddies_gamma

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OCEAN_H, OCEAN_W = 44, 94
FULL_H, FULL_W = 64, 128
OBS_ROW = 22

GP_PARAMS = {
    "lengthscale": 14.1,
    "variance": 0.0103420345,
    "noise": 1e-8,
    "kernel_type": "rbf_legacy",
    "coord_system": "pixels",
}

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
GP_CNN_CKPT = Path("results/gp_cnn/gp_cnn_best.pt")
OUT_DIR = Path("results/gp_cnn_eval")


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
# GP computation (on-the-fly for eval)
# ---------------------------------------------------------------------------

def compute_gp_mean(vel_phys: np.ndarray, obs_mask: np.ndarray,
                    ocean_mask: np.ndarray) -> np.ndarray:
    """
    Compute GP posterior mean for a single sample.

    vel_phys:   (2, H_ocean, W_ocean) physical-units velocity
    obs_mask:   (H_ocean, W_ocean) 1=known, 0=missing
    ocean_mask: (H_ocean, W_ocean) 1=ocean

    Returns: (2, H_ocean, W_ocean) GP posterior mean (raw space)
    """
    # Embed into 64×128 grid
    vel_full = np.zeros((1, 2, FULL_H, FULL_W), dtype=np.float32)
    vel_full[0, :, :OCEAN_H, :OCEAN_W] = vel_phys * ocean_mask[None]

    # GP mask: 0=known, 1=unknown (gp_fill convention)
    gp_mask = np.ones((FULL_H, FULL_W), dtype=np.float32)
    gp_mask[:OCEAN_H, :OCEAN_W] = 1.0 - obs_mask
    gp_mask[:OCEAN_H, :OCEAN_W] *= ocean_mask
    gp_mask[OCEAN_H:, :] = 0.0
    gp_mask[:, OCEAN_W:] = 0.0

    gp_mask_t = torch.from_numpy(gp_mask).unsqueeze(0).unsqueeze(0).expand(1, 2, -1, -1).clone()
    vel_t = torch.from_numpy(vel_full)

    gp_result = gp_fill(
        vel_t, gp_mask_t,
        lengthscale=GP_PARAMS["lengthscale"],
        variance=GP_PARAMS["variance"],
        noise=GP_PARAMS["noise"],
        use_double=True,
        kernel_type=GP_PARAMS["kernel_type"],
        coord_system=GP_PARAMS["coord_system"],
        return_variance=False,
    )
    # gp_fill returns (mean, var) tuple when return_variance=True, else just mean
    gp_mean = gp_result if not isinstance(gp_result, tuple) else gp_result[0]

    # Extract ocean sub-domain
    return gp_mean[0, :, :OCEAN_H, :OCEAN_W].numpy()


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def gp_cnn_infer_single(model, vel_phys: np.ndarray, obs_mask: np.ndarray,
                        ocean_mask: np.ndarray, norm_mean: np.ndarray,
                        norm_std: np.ndarray, gp_std_map: np.ndarray,
                        device: torch.device) -> torch.Tensor:
    """
    Run GP-CNN inference on a single sample.

    vel_phys:  (2, H, W) physical-units velocity
    obs_mask:  (H, W) 1=known, 0=missing
    Returns:   (2, H, W) reconstructed velocity in physical units
    """
    # Compute GP posterior mean
    gp_mean_raw = compute_gp_mean(vel_phys, obs_mask, ocean_mask)  # (2, H, W)

    # Normalize GP mean same way as training
    gp_mean_norm = (gp_mean_raw - norm_mean[:, None, None]) / norm_std[:, None, None]
    gp_mean_norm *= ocean_mask[None, :, :]

    # Build 6-channel input: [gp_u, gp_v, std_u, std_v, sensor_mask, ocean_mask]
    gp_mean_t = torch.from_numpy(gp_mean_norm.astype(np.float32))
    gp_std_t = torch.from_numpy(gp_std_map.astype(np.float32))           # (2, H, W)
    sensor_t = torch.from_numpy(obs_mask.astype(np.float32)).unsqueeze(0)  # (1, H, W)
    ocean_ch = torch.from_numpy(ocean_mask.astype(np.float32)).unsqueeze(0)  # (1, H, W)

    gp_input = torch.cat([gp_mean_t, gp_std_t, sensor_t, ocean_ch], dim=0)  # (6, H, W)
    gp_input = gp_input.unsqueeze(0).to(device)

    # Forward pass
    with torch.no_grad():
        pred_n = model(gp_input)  # (1, 2, H, W)

    # Unnormalize
    mean_t = torch.tensor(norm_mean, dtype=torch.float32).view(1, 2, 1, 1).to(device)
    std_t = torch.tensor(norm_std, dtype=torch.float32).view(1, 2, 1, 1).to(device)
    pred_phys = pred_n * std_t + mean_t

    # Zero out land
    ocean_t = torch.from_numpy(ocean_mask).to(device).unsqueeze(0).unsqueeze(0)
    pred_phys = pred_phys * ocean_t

    return pred_phys.squeeze(0).cpu()  # (2, H, W)


# ---------------------------------------------------------------------------
# Eddy helpers (same as voronoi_cnn_eval)
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

    # Load GP-CNN checkpoint
    print(f"Loading model from {GP_CNN_CKPT}...")
    ckpt = torch.load(GP_CNN_CKPT, map_location="cpu", weights_only=False)
    cfg = ckpt["model_config"]
    model = VoronoiCNN(
        in_channels=cfg["in_channels"],
        out_channels=cfg["out_channels"],
        base_ch=cfg.get("base_ch", 32),
        depth=cfg.get("depth", 3),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  Loaded epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.6f}")
    print(f"  Parameters: {model.count_parameters():,}")

    norm_mean = ckpt["norm_mean"].numpy()
    norm_std = ckpt["norm_std"].numpy()
    ocean_mask = ckpt["ocean_mask"]     # (H, W) numpy
    gp_std_map = ckpt["gp_std_map"]    # (2, H, W) numpy

    # Load raw validation data (2nd pickle element — unseen by GP-CNN)
    _, val_vel = load_pickle_data()

    # Build row-22 observation mask
    row22_mask = np.zeros((OCEAN_H, OCEAN_W), dtype=np.float32)
    row22_mask[OBS_ROW, :] = 1.0
    row22_mask *= ocean_mask

    # Load DDPM eval results for same sample indices
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
    print(f"Running GP-CNN inference on {n_total} samples")
    print(f"{'=' * 70}")

    results = []
    gp_mse_list = []  # GP-only baseline MSE for comparison
    t0_global = time.time()

    for run_i, vi in enumerate(val_indices):
        t0 = time.time()
        gt_vel = val_vel[vi].numpy()  # (2, H, W) physical

        # Run GP-CNN
        pred_vel = gp_cnn_infer_single(
            model, gt_vel, row22_mask, ocean_mask,
            norm_mean, norm_std, gp_std_map, device
        )  # (2, H, W) tensor

        # Also get GP-only baseline (recompute for consistency)
        gp_mean_raw = compute_gp_mean(gt_vel, row22_mask, ocean_mask)
        gp_pred = torch.from_numpy(gp_mean_raw)

        # MSE over ocean pixels
        gt_t = torch.from_numpy(gt_vel)
        ocean_t = torch.from_numpy(ocean_mask).bool()

        diff = (pred_vel - gt_t)[:, ocean_t]
        mse = (diff ** 2).mean().item()

        # MSE on missing-only ocean pixels
        known_t = torch.from_numpy(row22_mask).bool()
        missing_ocean = ocean_t & ~known_t
        diff_miss = (pred_vel - gt_t)[:, missing_ocean]
        mse_missing = (diff_miss ** 2).mean().item()

        # GP baseline MSE (missing only)
        gp_diff_miss = (gp_pred - gt_t)[:, missing_ocean]
        gp_mse_miss = (gp_diff_miss ** 2).mean().item()

        elapsed = time.time() - t0
        is_eddy = vi in eddy_set
        tag = "EDDY" if is_eddy else "clean"
        print(f"  [{run_i+1:3d}/{n_total}] val_idx={vi:5d} {tag:>5}  "
              f"MSE_miss={mse_missing:.6f}  GP_MSE={gp_mse_miss:.6f}  ({elapsed:.2f}s)")

        results.append({
            "run_idx": run_i,
            "val_idx": vi,
            "is_eddy": is_eddy,
            "gp_cnn_pred": pred_vel,       # (2, H, W)
            "gp_only_pred": gp_pred,       # (2, H, W)
            "ground_truth": gt_t,           # (2, H, W)
            "mse_all_ocean": mse,
            "mse_missing_only": mse_missing,
            "gp_mse_missing": gp_mse_miss,
        })
        gp_mse_list.append(gp_mse_miss)

    elapsed_total = time.time() - t0_global
    print(f"\nInference done: {n_total} samples in {elapsed_total:.1f}s "
          f"({elapsed_total / n_total:.2f}s/sample)")

    # Save results
    save_path = OUT_DIR / "gp_cnn_eval_results.pt"
    torch.save({
        "results": results,
        "n_samples": len(results),
        "val_indices": val_indices,
        "eddy_set": sorted(eddy_set),
        "mask_type": "row22",
        "eddy_params": EDDY_PARAMS,
        "gp_params": GP_PARAMS,
    }, save_path)
    print(f"Saved to {save_path}")

    # ------------------------------------------------------------------
    # Eddy detection evaluation
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("EDDY DETECTION EVALUATION (Gamma1)")
    print(f"{'=' * 70}\n")

    total_gt_eddies = 0
    total_tp, total_fp, total_fn = 0, 0, 0
    noeddy_fp = 0

    for r in results:
        gt_vel = r["ground_truth"]
        pred_vel = r["gp_cnn_pred"]

        gt_eddies = run_gamma1(gt_vel)
        pred_eddies = run_gamma1(pred_vel)

        n_gt = len(gt_eddies)
        n_pred = len(pred_eddies)

        if n_gt > 0:
            matches, fn_list, fp_list = match_eddies(gt_eddies, pred_eddies)
            tp, fn, fp = len(matches), len(fn_list), len(fp_list)
        else:
            tp, fn = 0, 0
            fp = n_pred

        total_gt_eddies += n_gt
        total_tp += tp
        total_fp += fp
        total_fn += fn

        if not r["is_eddy"] and n_pred > 0:
            noeddy_fp += 1

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
    print(f"  GP-only baseline: mean={np.mean(gp_mse_list):.6f}")

    # Compare with GP/DDPM/VCNN if available
    print(f"\n{'=' * 70}")
    print("COMPARISON WITH OTHER METHODS")
    print(f"{'=' * 70}\n")

    gpcnn_mses = [r["mse_missing_only"] for r in results]

    rows = [("GP-CNN", gpcnn_mses)]
    rows.append(("GP-only", gp_mse_list))

    # Load VCNN results if available
    vcnn_path = Path("results/voronoi_cnn_eval/voronoi_cnn_eval_results.pt")
    vcnn_mses = None
    if vcnn_path.exists():
        vcnn_data = torch.load(vcnn_path, map_location="cpu", weights_only=False)
        vcnn_mses = [float(r["mse_missing_only"]) for r in vcnn_data["results"][:n_total]]
        rows.append(("Voronoi-CNN", vcnn_mses))

    # Load DDPM results
    if use_ddpm_indices:
        gp_eval_mses = [float(s["gp_mse"]) for s in ddpm_samples[:n_total]]
        ddpm_eval_mses = [float(s["ddpm_mse"]) for s in ddpm_samples[:n_total]]
        rows.append(("GP (DDPM eval)", gp_eval_mses))
        rows.append(("DDPM (FiLM)", ddpm_eval_mses))

    print(f"  {'Method':<20} {'Mean MSE':>12} {'Median MSE':>12} {'Ratio vs GP':>12}")
    print(f"  {'-' * 58}")
    gp_baseline = np.mean(gp_mse_list)
    for name, mses in rows:
        m, md = np.mean(mses), np.median(mses)
        ratio = m / gp_baseline if gp_baseline > 0 else float('inf')
        print(f"  {name:<20} {m:>12.6f} {md:>12.6f} {ratio:>11.3f}x")

    # Win rates
    print(f"\n  Win rates (GP-CNN vs others):")
    wins_gp = sum(1 for g, gp in zip(gpcnn_mses, gp_mse_list) if g < gp)
    print(f"    GP-CNN beats GP-only:     {wins_gp}/{n_total} ({100*wins_gp/n_total:.1f}%)")

    if vcnn_mses:
        wins_vcnn = sum(1 for g, v in zip(gpcnn_mses, vcnn_mses) if g < v)
        print(f"    GP-CNN beats Voronoi-CNN: {wins_vcnn}/{n_total} ({100*wins_vcnn/n_total:.1f}%)")

    print(f"\nDone. Results saved to {OUT_DIR}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate GP-CNN on balanced eddy/non-eddy samples"
    )
    parser.add_argument("--n-samples", type=int, default=100)
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
