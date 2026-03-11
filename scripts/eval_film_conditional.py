#!/usr/bin/env python
"""Evaluate the converged FiLM conditional diffusion model on the same
100-sample eddy-balanced evaluation set used for GP-Diff.

This uses mask_aware_inpaint() — pure Palette-style conditional reverse
diffusion. No GP, no RePaint, no multi-stage refinement. The model alone
produces the reconstruction.

Usage:
    PYTHONPATH=. python scripts/eval_film_conditional.py
    PYTHONPATH=. python scripts/eval_film_conditional.py --eval-only
"""
import argparse, os, sys, time
from pathlib import Path

import torch
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import matplotlib
matplotlib.use("Agg")

from data_prep.data_initializer import DDInitializer
from ddpm.neural_networks.ddpm import GaussianDDPM
from ddpm.neural_networks.unets.unet_film import MyUNet_FiLM
from ddpm.helper_functions.masks.border_mask import BorderMaskGenerator
from ddpm.utils.inpainting_utils import mask_aware_inpaint
from ddpm.utils.noise_utils import get_noise_strategy
from ddpm.utils.eddy_detection import detect_eddies_gamma

# ── args ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--eval-only", action="store_true",
                    help="Skip inference, just run eddy eval from saved .pt")
parser.add_argument("--fresh", action="store_true",
                    help="Ignore existing checkpoint and start fresh")
parser.add_argument("--n-samples", type=int, default=1,
                    help="Number of diffusion samples to average (default: 1)")
args = parser.parse_args()

# ── paths ─────────────────────────────────────────────────────────────
MODEL_CKPT = (
    "ddpm/training/training_output/inpaint_film_t250/"
    "inpaint_spectral_div_free_t250_best_checkpoint.pt"
)
# Load the same evaluation indices from the GP-Diff eval
GPDIFF_PT = "results/eddy_balanced_eval/bulk_eval_eddy_balanced_100.pt"
OUT_DIR = "results/film_conditional_eval"
os.makedirs(OUT_DIR, exist_ok=True)
PT_PATH = os.path.join(OUT_DIR, "film_conditional_eval_100.pt")
SUMMARY_PATH = os.path.join(OUT_DIR, "eddy_eval_summary.txt")

# ── eddy detection params (same as GP-Diff eval) ─────────────────────
RADIUS = 8
GAMMA_THRESH = 0.65
MIN_AREA = 25
SHORE_BUFFER = 2
SMOOTH_SIGMA = 2.0
MIN_SPEED_RATIO = 0.3
MIN_VORTICITY = 0.03
DIST_THRESH = 10.0
OCEAN_H, OCEAN_W = 44, 94


# ── fixed center mask (same as GP-Diff eval) ─────────────────────────
_fixed_mask = None
def get_fixed_center_mask(image_shape):
    global _fixed_mask
    if _fixed_mask is not None:
        return _fixed_mask
    _, _, h, w = image_shape
    area_height, area_width = 44, 94
    mid_row = area_height // 2  # row 22
    mask = np.ones((h, w), dtype=np.float32)
    mask[mid_row:mid_row + 1, 0:area_width] = 0.0
    mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    border = BorderMaskGenerator().generate_mask(image_shape)
    mask = mask.to(border.device) * border
    _fixed_mask = mask
    return mask


# ═══════════════════════════════════════════════════════════════════════
#  INFERENCE
# ═══════════════════════════════════════════════════════════════════════

def run_inference():
    # Load the same indices from the GP-Diff eval
    gpdiff_data = torch.load(GPDIFF_PT, map_location="cpu", weights_only=False)
    val_indices = gpdiff_data["val_indices"]
    eddy_set = set(gpdiff_data.get("eddy_indices", []))
    noeddy_set = set(gpdiff_data.get("noeddy_indices", []))
    n_total = len(val_indices)
    print(f"Using same {len(eddy_set)} eddy + {len(noeddy_set)} non-eddy = "
          f"{n_total} samples as GP-Diff eval")

    dd = DDInitializer()
    device = dd.get_device()
    standardizer = dd.get_standardizer()

    # Load the FiLM model
    ckpt = torch.load(
        os.path.join(BASE_DIR, MODEL_CKPT), map_location=device, weights_only=False
    )
    n_steps  = ckpt.get("n_steps", 250)
    min_beta = ckpt.get("min_beta", 0.0004)
    max_beta = ckpt.get("max_beta", 0.08)
    epoch    = ckpt.get("epoch", "?")
    noise_fn = ckpt.get("noise_function", "spectral_div_free")

    # FiLM UNet: 5-channel input [x_t(2) + mask(1) + known(2)]
    network = MyUNet_FiLM(n_steps=n_steps, time_emb_dim=100, in_channels=5)
    ddpm = GaussianDDPM(
        network, n_steps=n_steps,
        min_beta=min_beta, max_beta=max_beta, device=device,
    )
    ddpm.load_state_dict(ckpt["model_state_dict"])
    ddpm = ddpm.to(device)
    ddpm.eval()

    noise_strategy = get_noise_strategy(noise_fn)
    print(f"Loaded FiLM model: epoch={epoch}, n_steps={n_steps}, "
          f"noise={noise_fn}, min_beta={min_beta}, max_beta={max_beta}")

    val_data = dd.get_validation_data()
    print(f"Validation set: {len(val_data)} samples")

    # Resume from checkpoint if available
    completed_samples = []
    start_idx = 0
    if not args.fresh and os.path.exists(PT_PATH):
        existing = torch.load(PT_PATH, map_location="cpu", weights_only=False)
        completed_samples = existing.get("samples", [])
        start_idx = len(completed_samples)
        if start_idx >= n_total:
            print(f"Already have {start_idx} samples in {PT_PATH}, nothing to do.")
            return val_indices, eddy_set, noeddy_set
        print(f"Resuming from sample {start_idx + 1} ({start_idx} already done)")

    t0_global = time.time()
    print(f"\nFiLM conditional eval: mask_aware_inpaint, n_samples={args.n_samples}")
    print(f"{'='*80}")
    print(f"{'#':>4}  {'ValIdx':>7} {'Type':>7}  {'FiLM MSE':>12}  {'Time':>7}")
    print(f"{'-'*80}")

    for run_i in range(start_idx, n_total):
        t0 = time.time()
        vi = val_indices[run_i]
        is_eddy = vi in eddy_set
        input_image = val_data[vi][0].unsqueeze(0).to(device)
        input_orig = standardizer.unstandardize(
            input_image.squeeze(0)
        ).to(device).unsqueeze(0)

        # Same mask as GP-Diff eval
        land_mask = (input_orig.abs() > 1e-5).float().to(device)
        raw_mask = get_fixed_center_mask(input_image.shape).to(device)
        missing_mask = raw_mask * land_mask

        # Run Palette-style conditional inpainting (in standardized space)
        torch.manual_seed(args.seed + run_i)
        with torch.no_grad():
            film_std = mask_aware_inpaint(
                ddpm,
                input_image,       # standardized known values
                missing_mask,      # 1=missing, 0=known
                n_samples=args.n_samples,
                device=device,
                noise_strategy=noise_strategy,
                mask_xt=False,     # match training config
            )

        # Convert to physical space
        film_phys = standardizer.unstandardize(
            film_std.squeeze(0)
        ).to(device).unsqueeze(0)

        # MSE over missing ocean pixels
        film_mse = ((film_phys - input_orig) * missing_mask).pow(2).sum() / (
            missing_mask.sum() + 1e-8
        )

        elapsed = time.time() - t0
        tag = "EDDY" if is_eddy else "clean"
        print(f"{run_i+1:>4}  {vi:>7} {tag:>7}  {film_mse.item():>12.6f}  "
              f"{elapsed:>6.1f}s")

        completed_samples.append({
            "idx": run_i,
            "val_idx": vi,
            "is_eddy_sample": is_eddy,
            "ground_truth": input_orig.cpu(),
            "film_output": film_phys.cpu(),
            "missing_mask": missing_mask.cpu(),
            "land_mask": land_mask.cpu(),
            "film_mse": film_mse.item(),
        })

        # Checkpoint every 10 samples
        if (run_i + 1) % 10 == 0 or (run_i + 1) == n_total:
            _save_pt(completed_samples, val_indices, eddy_set, noeddy_set)
            print(f"  [checkpoint saved: {len(completed_samples)} samples]")

    _save_pt(completed_samples, val_indices, eddy_set, noeddy_set)

    elapsed_total = time.time() - t0_global
    n = len(completed_samples)
    mses = [s["film_mse"] for s in completed_samples]
    print(f"\n{'='*80}")
    print(f"Inference complete: {n} samples in {elapsed_total:.0f}s "
          f"({elapsed_total/n:.1f}s/sample)")
    print(f"  Mean MSE: {np.mean(mses):.6f}")
    print(f"  Median MSE: {np.median(mses):.6f}")
    print(f"{'='*80}")

    return val_indices, eddy_set, noeddy_set


def _save_pt(samples, val_indices, eddy_set, noeddy_set):
    torch.save({
        "samples": samples,
        "n_samples": len(samples),
        "val_indices": val_indices,
        "eddy_indices": sorted(eddy_set),
        "noeddy_indices": sorted(noeddy_set),
        "model": "inpaint_film_t250",
        "inference": "mask_aware_inpaint (Palette-style)",
    }, PT_PATH)


# ═══════════════════════════════════════════════════════════════════════
#  EDDY EVALUATION
# ═══════════════════════════════════════════════════════════════════════

def crop_to_ocean(tensor_4d):
    return tensor_4d.squeeze(0)[:, :OCEAN_H, :OCEAN_W]


def run_gamma1(vel, ocean_mask=None):
    vel = torch.nan_to_num(vel, nan=0.0)
    eddies, _, _ = detect_eddies_gamma(
        vel, ocean_mask=ocean_mask,
        radius=RADIUS, gamma_threshold=GAMMA_THRESH,
        min_area=MIN_AREA, shore_buffer=SHORE_BUFFER,
        smooth_sigma=SMOOTH_SIGMA, min_mean_speed_ratio=MIN_SPEED_RATIO,
        min_vorticity=MIN_VORTICITY,
    )
    return eddies


def match_eddies(gt_eddies, pred_eddies, dist_thresh):
    matched = []
    unmatched_gt = list(range(len(gt_eddies)))
    fp = 0
    for pe in pred_eddies:
        best_d, best_gi = float("inf"), None
        for gi in unmatched_gt:
            ge = gt_eddies[gi]
            d = np.sqrt((pe["center_y"] - ge["center_y"])**2 +
                        (pe["center_x"] - ge["center_x"])**2)
            # Also check mask overlap
            overlap = (pe["mask"] & ge["mask"]).sum()
            if overlap > 0 and d < dist_thresh and d < best_d:
                best_d = d
                best_gi = gi
        if best_gi is not None:
            matched.append((best_gi, pe, best_d))
            unmatched_gt.remove(best_gi)
        else:
            fp += 1
    return matched, fp


def run_eddy_eval():
    print("\n" + "=" * 80)
    print("EDDY DETECTION EVALUATION (FiLM Conditional)")
    print("=" * 80)

    data = torch.load(PT_PATH, map_location="cpu", weights_only=False)
    samples = data["samples"]

    eddy_samples = [s for s in samples if s["is_eddy_sample"]]
    print(f"Evaluating {len(eddy_samples)} eddy-containing samples")

    gt_total = 0
    film_tp, film_fp = 0, 0
    all_dists = []

    for s in eddy_samples:
        gt_vel = crop_to_ocean(s["ground_truth"])
        film_vel = crop_to_ocean(s["film_output"])
        mask = s["missing_mask"].squeeze(0)[0, :OCEAN_H, :OCEAN_W]
        ocean_mask = (s["land_mask"].squeeze(0)[0, :OCEAN_H, :OCEAN_W] > 0.5)

        gt_eddies = run_gamma1(gt_vel, ocean_mask=ocean_mask)
        film_eddies = run_gamma1(film_vel, ocean_mask=ocean_mask)

        gt_total += len(gt_eddies)

        matched, fp = match_eddies(gt_eddies, film_eddies, DIST_THRESH)
        film_tp += len(matched)
        film_fp += fp
        all_dists.extend([d for _, _, d in matched])

    # Compute metrics
    film_prec = film_tp / (film_tp + film_fp) if (film_tp + film_fp) > 0 else 0
    film_rec = film_tp / gt_total if gt_total > 0 else 0
    film_f1 = 2 * film_prec * film_rec / (film_prec + film_rec) if (film_prec + film_rec) > 0 else 0

    # MSE summary
    eddy_mses = [s["film_mse"] for s in eddy_samples]
    noeddy_samples = [s for s in samples if not s["is_eddy_sample"]]
    noeddy_mses = [s["film_mse"] for s in noeddy_samples]
    all_mses = [s["film_mse"] for s in samples]

    # Load GP-Diff results for comparison
    gpdiff_data = torch.load(GPDIFF_PT, map_location="cpu", weights_only=False)
    gpdiff_samples = gpdiff_data["samples"]

    gp_mses_all = [s["gp_mse"] for s in gpdiff_samples]
    gpdiff_mses_all = [s["ddpm_mse"] for s in gpdiff_samples]

    lines = []
    lines.append("=" * 70)
    lines.append("FiLM Conditional Diffusion — Evaluation Summary")
    lines.append("=" * 70)
    lines.append(f"\nMSE (missing ocean pixels):")
    lines.append(f"  {'Subset':<15} {'FiLM':>10} {'GP-Diff':>10} {'GP':>10}")
    lines.append(f"  {'-'*45}")
    lines.append(f"  {'All '+str(len(samples)):<15} {np.mean(all_mses):>10.5f} "
                 f"{np.mean(gpdiff_mses_all):>10.5f} {np.mean(gp_mses_all):>10.5f}")
    if eddy_mses:
        gp_eddy = [s["gp_mse"] for s in gpdiff_samples if s["is_eddy_sample"]]
        gpdiff_eddy = [s["ddpm_mse"] for s in gpdiff_samples if s["is_eddy_sample"]]
        lines.append(f"  {'Eddy ('+str(len(eddy_mses))+')':<15} {np.mean(eddy_mses):>10.5f} "
                     f"{np.mean(gpdiff_eddy):>10.5f} {np.mean(gp_eddy):>10.5f}")
    if noeddy_mses:
        gp_noeddy = [s["gp_mse"] for s in gpdiff_samples if not s["is_eddy_sample"]]
        gpdiff_noeddy = [s["ddpm_mse"] for s in gpdiff_samples if not s["is_eddy_sample"]]
        lines.append(f"  {'Non-eddy ('+str(len(noeddy_mses))+')':<15} {np.mean(noeddy_mses):>10.5f} "
                     f"{np.mean(gpdiff_noeddy):>10.5f} {np.mean(gp_noeddy):>10.5f}")

    # Win rates vs GP and GP-Diff
    film_beats_gp = sum(1 for fm, gm in zip(all_mses, gp_mses_all) if fm < gm)
    film_beats_gpdiff = sum(1 for fm, dm in zip(all_mses, gpdiff_mses_all) if fm < dm)

    lines.append(f"\nWin rates:")
    lines.append(f"  FiLM beats GP:      {film_beats_gp}/{len(all_mses)} "
                 f"({100*film_beats_gp/len(all_mses):.1f}%)")
    lines.append(f"  FiLM beats GP-Diff: {film_beats_gpdiff}/{len(all_mses)} "
                 f"({100*film_beats_gpdiff/len(all_mses):.1f}%)")

    lines.append(f"\nEddy Detection ({len(eddy_samples)} eddy samples, "
                 f"{gt_total} ground-truth eddies):")
    lines.append(f"  TP={film_tp}, FP={film_fp}")
    lines.append(f"  Precision: {100*film_prec:.1f}%")
    lines.append(f"  Recall:    {100*film_rec:.1f}%")
    lines.append(f"  F1:        {film_f1:.3f}")
    if all_dists:
        lines.append(f"  Median center dist: {np.median(all_dists):.2f} px")
        lines.append(f"  Mean center dist:   {np.mean(all_dists):.2f} px")

    lines.append("=" * 70)

    summary = "\n".join(lines)
    print(summary)

    with open(SUMMARY_PATH, "w") as f:
        f.write(summary)
    print(f"\nSaved summary to {SUMMARY_PATH}")


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if not args.eval_only:
        run_inference()
    run_eddy_eval()
