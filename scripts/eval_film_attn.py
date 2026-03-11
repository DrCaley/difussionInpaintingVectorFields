#!/usr/bin/env python
"""Evaluate the FiLM+Attn conditional diffusion model (x0-prediction)
on the same 100-sample eddy-balanced evaluation set used for GP-Diff.

Uses x0_full_reverse_inpaint() — full 250-step reverse diffusion with
x₀-prediction posterior. No GP, no multi-stage refinement.

Usage:
    PYTHONPATH=. python scripts/eval_film_attn.py                    # full run
    PYTHONPATH=. python scripts/eval_film_attn.py --n-samples 3      # 3 samples
    PYTHONPATH=. python scripts/eval_film_attn.py --eval-only        # skip inference
    PYTHONPATH=. python scripts/eval_film_attn.py --quick 5          # just 5 samples
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
from ddpm.neural_networks.unets.unet_film_attn import MyUNet_FiLM_Attn
from ddpm.helper_functions.masks.border_mask import BorderMaskGenerator
from ddpm.utils.inpainting_utils import x0_full_reverse_inpaint
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
parser.add_argument("--quick", type=int, default=0,
                    help="Run only this many samples (0 = all)")
parser.add_argument("--repaint-steps", type=int, default=0,
                    help="RePaint resampling steps per timestep (default: 0)")
parser.add_argument("--project-steps", type=int, default=0,
                    help="Post-hoc div-free projection cycles (default: 0)")
args = parser.parse_args()

# ── paths ─────────────────────────────────────────────────────────────
EMA_WEIGHTS = (
    "experiments/03_conditioning/film_attn_divfree/results/"
    "model_weights_epoch161.pt"
)
GPDIFF_PT = "results/eddy_balanced_eval/bulk_eval_eddy_balanced_100.pt"
OUT_DIR = "results/film_attn_eval"
os.makedirs(OUT_DIR, exist_ok=True)
PT_PATH = os.path.join(OUT_DIR, "film_attn_eval_100.pt")
SUMMARY_PATH = os.path.join(OUT_DIR, "eddy_eval_summary.txt")

# ── model config (from resolved_config.yaml) ─────────────────────────
N_STEPS = 250
MIN_BETA = 0.0001
MAX_BETA = 0.02
NOISE_FN = "forward_diff_div_free"
SHARED_MEAN = -0.05084468695562498
SHARED_STD = 0.11479844598042026

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
    mask[mid_row:mid_row + 1, 0:area_width] = 0.0  # row 22 = known
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
    if args.quick > 0:
        n_total = min(args.quick, n_total)
    print(f"Using same {len(eddy_set)} eddy + {len(noeddy_set)} non-eddy = "
          f"{len(val_indices)} samples as GP-Diff eval (running {n_total})")

    dd = DDInitializer()
    device = dd.get_device()
    standardizer = dd.get_standardizer()

    # ── Build and load FiLM+Attn model ────────────────────────────────
    ema_path = os.path.join(BASE_DIR, EMA_WEIGHTS)
    print(f"Loading EMA weights from: {ema_path}")
    network = MyUNet_FiLM_Attn(n_steps=N_STEPS, time_emb_dim=256, in_channels=5)
    ddpm = GaussianDDPM(
        network, n_steps=N_STEPS,
        min_beta=MIN_BETA, max_beta=MAX_BETA, device=device,
    )

    # EMA weights are a bare state_dict (not a checkpoint dict)
    ema_state = torch.load(ema_path, map_location=device, weights_only=False)
    ddpm.load_state_dict(ema_state)
    ddpm = ddpm.to(device)
    ddpm.eval()

    noise_strategy = get_noise_strategy(NOISE_FN)
    n_params = sum(p.numel() for p in ddpm.parameters()) / 1e6
    print(f"FiLM+Attn model: {n_params:.1f}M params, n_steps={N_STEPS}, "
          f"noise={NOISE_FN}, beta=[{MIN_BETA}, {MAX_BETA}]")
    print(f"  mask_xt=True, prediction_target=x0, "
          f"repaint_steps={args.repaint_steps}, project_steps={args.project_steps}")

    val_data = dd.get_validation_data()
    print(f"Validation set: {len(val_data)} samples")

    # Resume from checkpoint if available
    completed_samples = []
    start_idx = 0
    if not args.fresh and os.path.exists(PT_PATH) and args.quick == 0:
        existing = torch.load(PT_PATH, map_location="cpu", weights_only=False)
        completed_samples = existing.get("samples", [])
        start_idx = len(completed_samples)
        if start_idx >= n_total:
            print(f"Already have {start_idx} samples in {PT_PATH}, nothing to do.")
            return val_indices, eddy_set, noeddy_set
        print(f"Resuming from sample {start_idx + 1} ({start_idx} already done)")

    t0_global = time.time()
    print(f"\nFiLM+Attn eval: x0_full_reverse_inpaint, n_samples={args.n_samples}")
    print(f"{'='*80}")
    print(f"{'#':>4}  {'ValIdx':>7} {'Type':>7}  {'FiLM-Attn MSE':>14}  {'Time':>7}")
    print(f"{'-'*80}")

    for run_i in range(start_idx, n_total):
        t0 = time.time()
        vi = val_indices[run_i]
        is_eddy = vi in eddy_set
        input_image = val_data[vi][0].unsqueeze(0).to(device)
        input_orig = standardizer.unstandardize(
            input_image.squeeze(0)
        ).to(device).unsqueeze(0)

        # Build mask: same as GP-Diff eval
        land_mask = (input_orig.abs() > 1e-5).float().to(device)
        raw_mask = get_fixed_center_mask(input_image.shape).to(device)
        # missing_mask: (1, 1, H, W) → need (1, 2, H, W) for inpainting fn
        missing_mask_1ch = raw_mask * land_mask[:, 0:1]
        missing_mask = missing_mask_1ch.expand(-1, 2, -1, -1)  # (1, 2, H, W)

        # Run x0-prediction full reverse inpainting
        torch.manual_seed(args.seed + run_i)
        with torch.no_grad():
            film_std = x0_full_reverse_inpaint(
                ddpm,
                input_image,        # (1, 2, H, W) standardised
                missing_mask,       # (1, 2, H, W) 1=missing, 0=known
                n_samples=args.n_samples,
                device=device,
                noise_strategy=noise_strategy,
                mask_xt=True,       # must match training config
                repaint_steps=args.repaint_steps,
                project_steps=args.project_steps,
            )

        # Convert to physical space
        film_phys = standardizer.unstandardize(
            film_std.squeeze(0)
        ).to(device).unsqueeze(0)

        # MSE over missing ocean pixels (use 1ch mask for counting)
        film_mse = ((film_phys - input_orig) * missing_mask).pow(2).sum() / (
            missing_mask.sum() + 1e-8
        )

        elapsed = time.time() - t0
        tag = "EDDY" if is_eddy else "clean"
        print(f"{run_i+1:>4}  {vi:>7} {tag:>7}  {film_mse.item():>14.6f}  "
              f"{elapsed:>6.1f}s")

        completed_samples.append({
            "idx": run_i,
            "val_idx": vi,
            "is_eddy_sample": is_eddy,
            "ground_truth": input_orig.cpu(),
            "film_attn_output": film_phys.cpu(),
            "missing_mask": missing_mask.cpu(),
            "land_mask": land_mask.cpu(),
            "film_attn_mse": film_mse.item(),
        })

        # Checkpoint every 10 samples
        if (run_i + 1) % 10 == 0 or (run_i + 1) == n_total:
            _save_pt(completed_samples, val_indices, eddy_set, noeddy_set)
            print(f"  [checkpoint saved: {len(completed_samples)} samples]")

    _save_pt(completed_samples, val_indices, eddy_set, noeddy_set)

    elapsed_total = time.time() - t0_global
    n = len(completed_samples)
    mses = [s["film_attn_mse"] for s in completed_samples]
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
        "model": "film_attn_divfree_x0_t250 (EMA, epoch ~96)",
        "inference": "x0_full_reverse_inpaint (mask_xt=True)",
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
    print("EDDY DETECTION EVALUATION (FiLM+Attn Conditional)")
    print("=" * 80)

    data = torch.load(PT_PATH, map_location="cpu", weights_only=False)
    samples = data["samples"]

    eddy_samples = [s for s in samples if s["is_eddy_sample"]]
    noeddy_samples = [s for s in samples if not s["is_eddy_sample"]]
    print(f"Evaluating {len(eddy_samples)} eddy + {len(noeddy_samples)} non-eddy samples")

    gt_total = 0
    film_tp, film_fp = 0, 0
    all_dists = []

    for s in eddy_samples:
        gt_vel = crop_to_ocean(s["ground_truth"])
        film_vel = crop_to_ocean(s["film_attn_output"])
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
    film_f1 = (2 * film_prec * film_rec / (film_prec + film_rec)
               if (film_prec + film_rec) > 0 else 0)

    # MSE summary
    eddy_mses = [s["film_attn_mse"] for s in eddy_samples]
    noeddy_mses = [s["film_attn_mse"] for s in noeddy_samples]
    all_mses = [s["film_attn_mse"] for s in samples]

    # Load GP-Diff results for comparison
    gpdiff_data = torch.load(GPDIFF_PT, map_location="cpu", weights_only=False)
    gpdiff_samples = gpdiff_data["samples"]

    gp_mses_all = [s["gp_mse"] for s in gpdiff_samples]
    gpdiff_mses_all = [s["ddpm_mse"] for s in gpdiff_samples]

    lines = []
    lines.append("=" * 70)
    lines.append("FiLM+Attn Conditional Diffusion — Evaluation Summary")
    lines.append("=" * 70)
    lines.append(f"\nMSE (missing ocean pixels):")
    lines.append(f"  {'Subset':<15} {'FiLM-Attn':>10} {'GP-Diff':>10} {'GP':>10}")
    lines.append(f"  {'-'*50}")

    # Only compare up to the number of samples we ran
    n_ran = len(samples)
    gp_sub = gp_mses_all[:n_ran]
    gpdiff_sub = gpdiff_mses_all[:n_ran]

    lines.append(f"  {'All '+str(n_ran):<15} {np.mean(all_mses):>10.5f} "
                 f"{np.mean(gpdiff_sub):>10.5f} {np.mean(gp_sub):>10.5f}")

    if eddy_mses:
        # Match indices for eddy/non-eddy comparison
        gp_eddy = [gp_mses_all[s["idx"]] for s in eddy_samples if s["idx"] < len(gp_mses_all)]
        gpdiff_eddy = [gpdiff_mses_all[s["idx"]] for s in eddy_samples if s["idx"] < len(gpdiff_mses_all)]
        if gp_eddy:
            lines.append(f"  {'Eddy ('+str(len(eddy_mses))+')':<15} {np.mean(eddy_mses):>10.5f} "
                         f"{np.mean(gpdiff_eddy):>10.5f} {np.mean(gp_eddy):>10.5f}")
    if noeddy_mses:
        gp_noeddy = [gp_mses_all[s["idx"]] for s in noeddy_samples if s["idx"] < len(gp_mses_all)]
        gpdiff_noeddy = [gpdiff_mses_all[s["idx"]] for s in noeddy_samples if s["idx"] < len(gpdiff_mses_all)]
        if gp_noeddy:
            lines.append(f"  {'Non-eddy ('+str(len(noeddy_mses))+')':<15} {np.mean(noeddy_mses):>10.5f} "
                         f"{np.mean(gpdiff_noeddy):>10.5f} {np.mean(gp_noeddy):>10.5f}")

    # Win rates
    film_beats_gp = sum(1 for fm, gm in zip(all_mses, gp_sub) if fm < gm)
    film_beats_gpdiff = sum(1 for fm, dm in zip(all_mses, gpdiff_sub) if fm < dm)

    lines.append(f"\nWin rates (n={n_ran}):")
    lines.append(f"  FiLM-Attn beats GP:      {film_beats_gp}/{n_ran} "
                 f"({100*film_beats_gp/n_ran:.1f}%)")
    lines.append(f"  FiLM-Attn beats GP-Diff: {film_beats_gpdiff}/{n_ran} "
                 f"({100*film_beats_gpdiff/n_ran:.1f}%)")

    if eddy_samples:
        lines.append(f"\nEddy Detection ({len(eddy_samples)} eddy samples, "
                     f"{gt_total} ground-truth eddies):")
        lines.append(f"  TP={film_tp}, FP={film_fp}")
        lines.append(f"  Precision: {100*film_prec:.1f}%")
        lines.append(f"  Recall:    {100*film_rec:.1f}%")
        lines.append(f"  F1:        {film_f1:.3f}")
        if all_dists:
            lines.append(f"  Median center dist: {np.median(all_dists):.2f} px")
            lines.append(f"  Mean center dist:   {np.mean(all_dists):.2f} px")

    lines.append("\n" + "=" * 70)

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
    if args.quick == 0 or args.eval_only:
        run_eddy_eval()
    else:
        # Quick mode: just print MSE summary
        data = torch.load(PT_PATH, map_location="cpu", weights_only=False)
        samples = data["samples"]
        mses = [s["film_attn_mse"] for s in samples]
        print(f"\nQuick summary ({len(samples)} samples):")
        print(f"  Mean MSE:   {np.mean(mses):.6f}")
        print(f"  Median MSE: {np.median(mses):.6f}")
        print(f"  Min MSE:    {np.min(mses):.6f}")
        print(f"  Max MSE:    {np.max(mses):.6f}")
