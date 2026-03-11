#!/usr/bin/env python
"""Evaluate FiLM+Attn conditional diffusion with GP-warm-start + adaptive RePaint.

Adapts the GP-Diff inference strategy for the conditional FiLM model:
  - GP warm-start: starts from GP posterior at t_start (not pure noise)
  - Variance-adaptive noise: modulates noise using GP confidence map
  - RePaint resampling: re-noises and re-denoises for boundary coherence
  - Multi-stage refinement: cascades stages with decaying variance map

Evaluates on the same 100-sample eddy-balanced set as GP-Diff.

Usage:
    PYTHONPATH=. python scripts/eval_film_attn_gp.py                     # full 100
    PYTHONPATH=. python scripts/eval_film_attn_gp.py --quick 20          # quick test
    PYTHONPATH=. python scripts/eval_film_attn_gp.py --quick 5 --fresh   # from scratch
    PYTHONPATH=. python scripts/eval_film_attn_gp.py --eval-only         # eddy eval only
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
from ddpm.utils.inpainting_utils import x0_film_gp_repaint
from ddpm.utils.noise_utils import get_noise_strategy
from ddpm.helper_functions.interpolation_tool import gp_fill
from ddpm.utils.eddy_detection import detect_eddies_gamma

# ── args ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--eval-only", action="store_true")
parser.add_argument("--fresh", action="store_true")
parser.add_argument("--quick", type=int, default=0)
# GP-Diff inference hyperparams (matching paper defaults)
parser.add_argument("--max-stages", type=int, default=6,
                    help="Number of multi-stage refinement stages (default: 6)")
parser.add_argument("--t-start", type=int, default=75,
                    help="Stage 1 starting timestep (default: 75)")
parser.add_argument("--t-refine", type=int, default=50,
                    help="Stages 2+ starting timestep (default: 50)")
parser.add_argument("--resample-steps", type=int, default=5,
                    help="RePaint resample steps per timestep (default: 5)")
parser.add_argument("--noise-floor", type=float, default=0.2,
                    help="Stage 1 noise floor (default: 0.2)")
parser.add_argument("--noise-floor-refine", type=float, default=0.3,
                    help="Stages 2+ noise floor (default: 0.3)")
parser.add_argument("--var-decay", type=float, default=0.1,
                    help="Variance decay per stage (default: 0.1)")
parser.add_argument("--gamma", type=float, default=3.0,
                    help="Variance nonlinearity exponent (default: 3.0)")
parser.add_argument("--project-final", type=int, default=0,
                    help="Final CG div-free projection iterations (default: 0)")
args = parser.parse_args()

# ── paths ─────────────────────────────────────────────────────────────
MODEL_WEIGHTS = (
    "experiments/03_conditioning/film_attn_divfree/results/"
    "model_weights_epoch161.pt"
)
GPDIFF_PT = "results/eddy_balanced_eval/bulk_eval_eddy_balanced_100.pt"
OUT_DIR = "results/film_attn_gp_eval"
os.makedirs(OUT_DIR, exist_ok=True)
PT_PATH = os.path.join(OUT_DIR, "film_attn_gp_eval.pt")
SUMMARY_PATH = os.path.join(OUT_DIR, "eddy_eval_summary.txt")

# ── model config ──────────────────────────────────────────────────────
N_STEPS = 250
MIN_BETA = 0.0001
MAX_BETA = 0.02
NOISE_FN = "forward_diff_div_free"

# ── eddy detection params ────────────────────────────────────────────
RADIUS = 8
GAMMA_THRESH = 0.65
MIN_AREA = 25
SHORE_BUFFER = 2
SMOOTH_SIGMA = 2.0
MIN_SPEED_RATIO = 0.3
MIN_VORTICITY = 0.03
DIST_THRESH = 10.0
OCEAN_H, OCEAN_W = 44, 94


# ── fixed center mask ────────────────────────────────────────────────
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


def run_stage(ddpm, input_image, missing_mask, gp_std, var_map,
              t_start, noise_floor, seed, noise_strategy, resample_steps,
              gamma, device, mask_xt=True, project_final=0):
    """Run a single stage of FiLM GP-warm-started adaptive RePaint."""
    torch.manual_seed(seed)
    with torch.no_grad():
        out = x0_film_gp_repaint(
            ddpm, input_image, missing_mask,
            gp_image=gp_std,
            gp_variance_map=var_map,
            t_start=t_start,
            noise_floor=noise_floor,
            n_samples=1, device=device,
            noise_strategy=noise_strategy,
            mask_xt=mask_xt,
            resample_steps=resample_steps,
            gamma=gamma,
            project_final_steps=project_final,
        )
    return out


# ═══════════════════════════════════════════════════════════════════════
#  INFERENCE
# ═══════════════════════════════════════════════════════════════════════

def run_inference():
    # Load same indices as GP-Diff eval
    gpdiff_data = torch.load(GPDIFF_PT, map_location="cpu", weights_only=False)
    val_indices = gpdiff_data["val_indices"]
    eddy_set = set(gpdiff_data.get("eddy_indices", []))
    noeddy_set = set(gpdiff_data.get("noeddy_indices", []))
    n_total = len(val_indices)
    if args.quick > 0:
        n_total = min(args.quick, n_total)
    print(f"Using same {len(eddy_set)} eddy + {len(noeddy_set)} non-eddy = "
          f"{len(val_indices)} samples (running {n_total})")

    dd = DDInitializer()
    device = dd.get_device()
    standardizer = dd.get_standardizer()
    noise_strategy = get_noise_strategy(NOISE_FN)

    # ── Build and load FiLM+Attn model ────────────────────────────────
    weights_path = os.path.join(BASE_DIR, MODEL_WEIGHTS)
    print(f"Loading weights from: {weights_path}")
    network = MyUNet_FiLM_Attn(n_steps=N_STEPS, time_emb_dim=256, in_channels=5)
    ddpm = GaussianDDPM(
        network, n_steps=N_STEPS,
        min_beta=MIN_BETA, max_beta=MAX_BETA, device=device,
    )
    state = torch.load(weights_path, map_location=device, weights_only=False)
    ddpm.load_state_dict(state)
    ddpm = ddpm.to(device)
    ddpm.eval()

    n_params = sum(p.numel() for p in ddpm.parameters()) / 1e6
    print(f"FiLM+Attn: {n_params:.1f}M params, T={N_STEPS}, noise={NOISE_FN}")
    print(f"Inference: GP warm-start + adaptive RePaint (FiLM conditioning)")
    print(f"  S1: t={args.t_start}, floor={args.noise_floor}, "
          f"resample={args.resample_steps}, gamma={args.gamma}")
    print(f"  S2-{args.max_stages}: t={args.t_refine}, "
          f"floor={args.noise_floor_refine}, var_decay={args.var_decay}")

    val_data = dd.get_validation_data()
    print(f"Validation set: {len(val_data)} samples")

    # Resume support
    completed_samples = []
    start_idx = 0
    if not args.fresh and os.path.exists(PT_PATH) and args.quick == 0:
        existing = torch.load(PT_PATH, map_location="cpu", weights_only=False)
        completed_samples = existing.get("samples", [])
        start_idx = len(completed_samples)
        if start_idx >= n_total:
            print(f"Already have {start_idx} samples, nothing to do.")
            return val_indices, eddy_set, noeddy_set
        print(f"Resuming from sample {start_idx + 1} ({start_idx} done)")

    t0_global = time.time()
    print(f"\n{'='*90}")
    print(f"{'#':>4}  {'ValIdx':>7} {'Type':>7}  {'GP MSE':>10}  "
          f"{'FiLM-GP MSE':>12}  {'Ratio':>7}  {'Time':>7}")
    print(f"{'-'*90}")

    for run_i in range(start_idx, n_total):
        t0 = time.time()
        vi = val_indices[run_i]
        is_eddy = vi in eddy_set
        input_image = val_data[vi][0].unsqueeze(0).to(device)
        input_orig = standardizer.unstandardize(
            input_image.squeeze(0)
        ).to(device).unsqueeze(0)

        # Build mask
        land_mask = (input_orig.abs() > 1e-5).float().to(device)
        raw_mask = get_fixed_center_mask(input_image.shape).to(device)
        missing_mask_1ch = raw_mask * land_mask[:, 0:1]
        missing_mask = missing_mask_1ch.expand(-1, 2, -1, -1)

        # ── GP fill (with variance) ──────────────────────────────────
        gp_out, gp_var_map = gp_fill(
            input_orig, missing_mask,
            lengthscale=dd.get_attribute("gp_lengthscale"),
            variance=dd.get_attribute("gp_variance"),
            noise=dd.get_attribute("gp_noise"),
            use_double=True,
            kernel_type=dd.get_attribute("gp_kernel_type"),
            coord_system=dd.get_attribute("gp_coord_system"),
            return_variance=True,
        )

        # GP MSE
        gp_mse = ((gp_out - input_orig) * missing_mask).pow(2).sum() / (
            missing_mask.sum() + 1e-8)

        # Standardize GP output for use as diffusion prior
        gp_std = standardizer(gp_out.squeeze(0)).to(device).unsqueeze(0)

        # ── Multi-stage refinement ────────────────────────────────────
        current_prior = gp_std.clone()
        current_var = gp_var_map.clone()

        for stage in range(1, args.max_stages + 1):
            if stage == 1:
                t_s, nf = args.t_start, args.noise_floor
                seed_s = args.seed + run_i
            else:
                t_s, nf = args.t_refine, args.noise_floor_refine
                seed_s = args.seed + run_i + stage * 10000
                current_var = current_var * args.var_decay

            stage_out = run_stage(
                ddpm, input_image, missing_mask,
                gp_std=current_prior,
                var_map=current_var,
                t_start=t_s, noise_floor=nf,
                seed=seed_s,
                noise_strategy=noise_strategy,
                resample_steps=args.resample_steps,
                gamma=args.gamma,
                device=device,
                project_final=(args.project_final if stage == args.max_stages else 0),
            )
            current_prior = stage_out.clone()

        # Convert to physical space
        film_phys = standardizer.unstandardize(
            stage_out.squeeze(0)
        ).to(device).unsqueeze(0)

        # MSE over missing ocean pixels
        film_mse = ((film_phys - input_orig) * missing_mask).pow(2).sum() / (
            missing_mask.sum() + 1e-8)
        ratio = film_mse.item() / (gp_mse.item() + 1e-12)

        elapsed = time.time() - t0
        tag = "EDDY" if is_eddy else "clean"
        print(f"{run_i+1:>4}  {vi:>7} {tag:>7}  {gp_mse.item():>10.6f}  "
              f"{film_mse.item():>12.6f}  {ratio:>6.3f}x  {elapsed:>6.1f}s")

        completed_samples.append({
            "idx": run_i,
            "val_idx": vi,
            "is_eddy_sample": is_eddy,
            "ground_truth": input_orig.cpu(),
            "gp_output": gp_out.cpu(),
            "film_attn_output": film_phys.cpu(),
            "missing_mask": missing_mask.cpu(),
            "land_mask": land_mask.cpu(),
            "gp_mse": gp_mse.item(),
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
    gp_mses = [s["gp_mse"] for s in completed_samples]
    print(f"\n{'='*90}")
    print(f"Inference done: {n} samples in {elapsed_total:.0f}s "
          f"({elapsed_total/n:.1f}s/sample)")
    print(f"  FiLM-GP  Mean MSE:   {np.mean(mses):.6f}  Median: {np.median(mses):.6f}")
    print(f"  GP       Mean MSE:   {np.mean(gp_mses):.6f}  Median: {np.median(gp_mses):.6f}")
    wins = sum(1 for f, g in zip(mses, gp_mses) if f < g)
    print(f"  FiLM-GP beats GP: {wins}/{n} ({100*wins/n:.0f}%)")
    print(f"{'='*90}")

    return val_indices, eddy_set, noeddy_set


def _save_pt(samples, val_indices, eddy_set, noeddy_set):
    torch.save({
        "samples": samples,
        "n_samples": len(samples),
        "val_indices": val_indices,
        "eddy_indices": sorted(eddy_set),
        "noeddy_indices": sorted(noeddy_set),
        "model": "film_attn_divfree_x0_t250 (epoch 161 raw weights)",
        "inference": f"x0_film_gp_repaint (S{args.max_stages}, t_start={args.t_start}, "
                     f"resample={args.resample_steps}, gamma={args.gamma})",
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
    print("\n" + "=" * 90)
    print("EDDY DETECTION EVALUATION — FiLM+Attn + GP Warm-Start + Adaptive RePaint")
    print("=" * 90)

    data = torch.load(PT_PATH, map_location="cpu", weights_only=False)
    samples = data["samples"]

    eddy_samples = [s for s in samples if s["is_eddy_sample"]]
    noeddy_samples = [s for s in samples if not s["is_eddy_sample"]]
    print(f"Evaluating {len(eddy_samples)} eddy + {len(noeddy_samples)} non-eddy samples")

    gt_total = 0
    film_tp, film_fp = 0, 0
    gp_tp, gp_fp = 0, 0
    all_dists = []

    for s in eddy_samples:
        gt_vel = crop_to_ocean(s["ground_truth"])
        film_vel = crop_to_ocean(s["film_attn_output"])
        gp_vel = crop_to_ocean(s["gp_output"])
        ocean_mask = (s["land_mask"].squeeze(0)[0, :OCEAN_H, :OCEAN_W] > 0.5)

        gt_eddies = run_gamma1(gt_vel, ocean_mask=ocean_mask)
        film_eddies = run_gamma1(film_vel, ocean_mask=ocean_mask)
        gp_eddies = run_gamma1(gp_vel, ocean_mask=ocean_mask)

        gt_total += len(gt_eddies)

        m, fp = match_eddies(gt_eddies, film_eddies, DIST_THRESH)
        film_tp += len(m)
        film_fp += fp
        all_dists.extend([d for _, _, d in m])

        m_gp, fp_gp = match_eddies(gt_eddies, gp_eddies, DIST_THRESH)
        gp_tp += len(m_gp)
        gp_fp += fp_gp

    # Compute metrics
    film_prec = film_tp / (film_tp + film_fp) if (film_tp + film_fp) > 0 else 0
    film_rec = film_tp / gt_total if gt_total > 0 else 0
    film_f1 = (2 * film_prec * film_rec / (film_prec + film_rec)
               if (film_prec + film_rec) > 0 else 0)

    gp_prec = gp_tp / (gp_tp + gp_fp) if (gp_tp + gp_fp) > 0 else 0
    gp_rec = gp_tp / gt_total if gt_total > 0 else 0
    gp_f1 = (2 * gp_prec * gp_rec / (gp_prec + gp_rec)
             if (gp_prec + gp_rec) > 0 else 0)

    # MSE summary
    eddy_mses = [s["film_attn_mse"] for s in eddy_samples]
    noeddy_mses = [s["film_attn_mse"] for s in noeddy_samples]
    all_film_mses = [s["film_attn_mse"] for s in samples]
    all_gp_mses = [s["gp_mse"] for s in samples]

    # Load GP-Diff numbers for comparison
    gpdiff_data = torch.load(GPDIFF_PT, map_location="cpu", weights_only=False)
    gpdiff_samples = gpdiff_data["samples"]
    gpdiff_mses_all = [s["ddpm_mse"] for s in gpdiff_samples]
    gp_ref_mses = [s["gp_mse"] for s in gpdiff_samples]

    n_ran = len(samples)
    gpdiff_sub = gpdiff_mses_all[:n_ran]

    lines = []
    lines.append("=" * 70)
    lines.append("FiLM+Attn + GP Warm-Start + Adaptive RePaint — Evaluation")
    lines.append(f"  Stages: {args.max_stages}, t_start={args.t_start}, "
                 f"t_refine={args.t_refine}")
    lines.append(f"  Resample: {args.resample_steps}, gamma={args.gamma}")
    lines.append(f"  Noise floors: S1={args.noise_floor}, "
                 f"S2+={args.noise_floor_refine}")
    lines.append("=" * 70)

    lines.append(f"\nMSE (missing ocean pixels):")
    lines.append(f"  {'Subset':<15} {'FiLM-GP':>10} {'GP-Diff':>10} {'GP':>10}")
    lines.append(f"  {'-'*50}")

    lines.append(f"  {'All '+str(n_ran):<15} {np.mean(all_film_mses):>10.5f} "
                 f"{np.mean(gpdiff_sub):>10.5f} {np.mean(all_gp_mses):>10.5f}")
    lines.append(f"  {'Median':<15} {np.median(all_film_mses):>10.5f} "
                 f"{np.median(gpdiff_sub):>10.5f} {np.median(all_gp_mses):>10.5f}")

    if eddy_mses:
        gpdiff_eddy = [gpdiff_mses_all[s["idx"]] for s in eddy_samples
                       if s["idx"] < len(gpdiff_mses_all)]
        gp_eddy = [s["gp_mse"] for s in eddy_samples]
        if gpdiff_eddy:
            lines.append(f"  {'Eddy ('+str(len(eddy_mses))+')':<15} "
                         f"{np.mean(eddy_mses):>10.5f} "
                         f"{np.mean(gpdiff_eddy):>10.5f} "
                         f"{np.mean(gp_eddy):>10.5f}")
    if noeddy_mses:
        gpdiff_noeddy = [gpdiff_mses_all[s["idx"]] for s in noeddy_samples
                         if s["idx"] < len(gpdiff_mses_all)]
        gp_noeddy = [s["gp_mse"] for s in noeddy_samples]
        if gpdiff_noeddy:
            lines.append(f"  {'Non-eddy ('+str(len(noeddy_mses))+')':<15} "
                         f"{np.mean(noeddy_mses):>10.5f} "
                         f"{np.mean(gpdiff_noeddy):>10.5f} "
                         f"{np.mean(gp_noeddy):>10.5f}")

    # Win rates
    film_beats_gp = sum(1 for f, g in zip(all_film_mses, all_gp_mses) if f < g)
    film_beats_gpdiff = sum(1 for f, d in zip(all_film_mses, gpdiff_sub) if f < d)

    lines.append(f"\nWin rates (n={n_ran}):")
    lines.append(f"  FiLM-GP beats GP:      {film_beats_gp}/{n_ran} "
                 f"({100*film_beats_gp/n_ran:.1f}%)")
    lines.append(f"  FiLM-GP beats GP-Diff: {film_beats_gpdiff}/{n_ran} "
                 f"({100*film_beats_gpdiff/n_ran:.1f}%)")

    if eddy_samples:
        lines.append(f"\nEddy Detection ({len(eddy_samples)} eddy samples, "
                     f"{gt_total} GT eddies):")
        lines.append(f"  {'Method':<12} {'TP':>4} {'FP':>4} {'Prec':>8} "
                     f"{'Recall':>8} {'F1':>6}")
        lines.append(f"  {'-'*50}")
        lines.append(f"  {'FiLM-GP':<12} {film_tp:>4} {film_fp:>4} "
                     f"{100*film_prec:>7.1f}% {100*film_rec:>7.1f}% "
                     f"{film_f1:>6.3f}")
        lines.append(f"  {'GP':<12} {gp_tp:>4} {gp_fp:>4} "
                     f"{100*gp_prec:>7.1f}% {100*gp_rec:>7.1f}% "
                     f"{gp_f1:>6.3f}")

        # GP-Diff reference numbers from paper
        lines.append(f"  {'GP-Diff*':<12} {'14':>4} {'4':>4} "
                     f"{'77.8':>7}% {'24.6':>7}% {'0.373':>6}")

        if all_dists:
            lines.append(f"\n  FiLM-GP localization:")
            lines.append(f"    Median center dist: {np.median(all_dists):.2f} px")
            lines.append(f"    Mean center dist:   {np.mean(all_dists):.2f} px")

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
        # Quick mode: print MSE summary
        data = torch.load(PT_PATH, map_location="cpu", weights_only=False)
        samples = data["samples"]
        film_mses = [s["film_attn_mse"] for s in samples]
        gp_mses = [s["gp_mse"] for s in samples]
        wins = sum(1 for f, g in zip(film_mses, gp_mses) if f < g)
        print(f"\nQuick summary ({len(samples)} samples):")
        print(f"  FiLM-GP  Mean MSE:   {np.mean(film_mses):.6f}  "
              f"Median: {np.median(film_mses):.6f}")
        print(f"  GP       Mean MSE:   {np.mean(gp_mses):.6f}  "
              f"Median: {np.median(gp_mses):.6f}")
        print(f"  FiLM-GP beats GP: {wins}/{len(samples)}")
        print(f"  Min MSE: {np.min(film_mses):.6f}  Max: {np.max(film_mses):.6f}")
