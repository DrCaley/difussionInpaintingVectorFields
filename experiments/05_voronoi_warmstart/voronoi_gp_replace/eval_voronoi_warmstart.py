#!/usr/bin/env python
"""Voronoi warm-start evaluation: same 100 eddy-balanced samples as GP-Diff.

Replaces the GP posterior mean with a Voronoi tessellation (nearest-neighbour fill)
and the GP posterior variance with distance-from-sensor² as a variance proxy.
Everything else (S6 adaptive RePaint, same DDPM checkpoint, same eval samples)
is identical to the production GP-Diff pipeline.

Usage:
    PYTHONPATH=. python experiments/05_voronoi_warmstart/voronoi_gp_replace/eval_voronoi_warmstart.py
    PYTHONPATH=. python experiments/05_voronoi_warmstart/voronoi_gp_replace/eval_voronoi_warmstart.py --max-stages 1
"""
import argparse, os, sys, time
from pathlib import Path

import torch
import numpy as np
from scipy.spatial import cKDTree

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from data_prep.data_initializer import DDInitializer
from ddpm.neural_networks.ddpm import GaussianDDPM
from ddpm.neural_networks.unets.unet_xl_attn import MyUNet_Attn
from ddpm.helper_functions.masks.border_mask import BorderMaskGenerator
from ddpm.utils.inpainting_utils import repaint_gp_init_adaptive
from ddpm.utils.noise_utils import get_noise_strategy
from ddpm.utils.eddy_detection import detect_eddies_gamma
from collections import defaultdict

# ── args ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--max-stages", type=int, default=6)
parser.add_argument("--t-start", type=int, default=75)
parser.add_argument("--t-refine", type=int, default=50)
parser.add_argument("--resample-steps", type=int, default=5)
parser.add_argument("--noise-floor", type=float, default=0.2)
parser.add_argument("--noise-floor-refine", type=float, default=0.3)
parser.add_argument("--var-decay", type=float, default=0.1)
parser.add_argument("--gamma", type=float, default=3.0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--n-samples", type=int, default=100,
                    help="Number of samples to eval (default: full 100)")
parser.add_argument("--fresh", action="store_true")
parser.add_argument("--eval-only", action="store_true",
                    help="Skip inference, just run eddy eval from saved .pt")
args = parser.parse_args()

# ── paths ────────────────────────────────────────────────────────────
MODEL_CKPT = str(BASE_DIR / (
    "experiments/02_inpaint_algorithm/repaint_gaussian_attn/results/"
    "inpaint_gaussian_t250_best_checkpoint.pt"
))
# Load the SAME sample indices as the GP-Diff balanced eval
GPDIFF_PT = str(BASE_DIR / "results/eddy_balanced_eval/bulk_eval_eddy_balanced_100.pt")
OUT_DIR = str(Path(__file__).resolve().parent / "results")
os.makedirs(OUT_DIR, exist_ok=True)
PT_PATH = os.path.join(OUT_DIR, "voronoi_warmstart_eval.pt")

# ── eddy detection params (same as GP-Diff eval) ────────────────────
RADIUS = 8
GAMMA_THRESH = 0.65
MIN_AREA = 25
SHORE_BUFFER = 2
SMOOTH_SIGMA = 2.0
MIN_SPEED_RATIO = 0.3
MIN_VORTICITY = 0.03
DIST_THRESH = 10.0
OCEAN_H, OCEAN_W = 44, 94


# ── Voronoi tessellation (replaces GP) ──────────────────────────────

def voronoi_fill(vel_phys, obs_mask_2d, ocean_mask_2d):
    """Build Voronoi-tessellated velocity field from sparse observations.

    Args:
        vel_phys: (1, 2, H, W) velocity in physical space
        obs_mask_2d: (H, W) ndarray, 1=known, 0=missing
        ocean_mask_2d: (H, W) ndarray, 1=ocean, 0=land

    Returns:
        voronoi_field: (1, 2, H, W) tensor — Voronoi-filled in physical space
        dist_variance: (1, 2, H, W) tensor — distance² proxy for GP variance
    """
    H, W = obs_mask_2d.shape
    vel_np = vel_phys.squeeze(0).cpu().numpy()  # (2, H, W)

    # Find observed pixel coordinates
    ky, kx = np.where(obs_mask_2d > 0.5)
    if len(ky) == 0:
        return torch.zeros_like(vel_phys), torch.ones(1, 2, H, W)

    # KD-tree for nearest-neighbour lookup
    obs_coords = np.stack([ky, kx], axis=1).astype(np.float64)
    tree = cKDTree(obs_coords)

    gy, gx = np.mgrid[0:H, 0:W]
    grid_coords = np.stack([gy.ravel(), gx.ravel()], axis=1).astype(np.float64)
    dist, idx = tree.query(grid_coords, k=1)

    dist = dist.reshape(H, W)
    idx = idx.reshape(H, W)

    # Voronoi fill: each pixel gets the value of its nearest sensor
    obs_u = vel_np[0, ky, kx]
    obs_v = vel_np[1, ky, kx]
    voronoi_u = obs_u[idx]
    voronoi_v = obs_v[idx]

    # Apply ocean mask
    voronoi_u *= ocean_mask_2d
    voronoi_v *= ocean_mask_2d

    voronoi_field = np.stack([voronoi_u, voronoi_v], axis=0)[np.newaxis]  # (1,2,H,W)
    voronoi_field = torch.tensor(voronoi_field, dtype=torch.float32)

    # Distance-squared as variance proxy (analogous to GP posterior variance:
    # high far from observations, zero at observations)
    dist_sq = (dist ** 2) * ocean_mask_2d
    # Broadcast to (1, 2, H, W) — same variance for both channels
    dist_var = np.stack([dist_sq, dist_sq], axis=0)[np.newaxis]
    dist_variance = torch.tensor(dist_var, dtype=torch.float32)

    return voronoi_field, dist_variance


# ── fixed center mask (same as GP-Diff) ─────────────────────────────
_fixed_mask = None
def get_fixed_center_mask(image_shape):
    global _fixed_mask
    if _fixed_mask is not None:
        return _fixed_mask
    _, _, h, w = image_shape
    area_height, area_width = 44, 94
    mid_row = area_height // 2  # row 22
    mask = np.ones((h, w), dtype=np.float32)
    mask[mid_row:mid_row + 1, 0:area_width] = 0.0  # 0 = known
    mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    border = BorderMaskGenerator().generate_mask(image_shape)
    mask = mask.to(border.device) * border
    _fixed_mask = mask
    return mask


def run_stage(ddpm, input_image, missing_mask, prior_image, var_map,
              t_start, noise_floor, seed, noise_strategy, resample_steps,
              gamma, device):
    torch.manual_seed(seed)
    with torch.no_grad():
        out = repaint_gp_init_adaptive(
            ddpm, input_image, missing_mask,
            gp_image=prior_image,
            gp_variance_map=var_map,
            t_start=t_start,
            noise_floor=noise_floor,
            n_samples=1, device=device,
            noise_strategy=noise_strategy,
            prediction_target="eps",
            resample_steps=resample_steps,
            project_div_free=False,
            anneal_floor=False,
            gamma=gamma,
        )
    return out


# ═══════════════════════════════════════════════════════════════════
#  INFERENCE
# ═══════════════════════════════════════════════════════════════════

def run_inference():
    # Load GP-Diff eval to get identical sample indices
    gpdiff_data = torch.load(GPDIFF_PT, map_location="cpu", weights_only=False)
    val_indices = gpdiff_data["val_indices"]
    eddy_set = set(gpdiff_data.get("eddy_indices", []))
    noeddy_set = set(gpdiff_data.get("noeddy_indices", []))
    n_total = min(args.n_samples, len(val_indices))

    print(f"Using {n_total} samples from GP-Diff balanced eval")
    print(f"  Eddy: {len(eddy_set)}, Non-eddy: {len(noeddy_set)}")

    dd = DDInitializer()
    device = dd.get_device()
    noise_strategy = get_noise_strategy("gaussian")
    standardizer = dd.get_standardizer()

    # Load DDPM (same checkpoint as GP-Diff)
    ckpt = torch.load(MODEL_CKPT, map_location=device, weights_only=False)
    n_steps = ckpt.get("n_steps", 250)
    min_beta = ckpt.get("min_beta", 0.0001)
    max_beta = ckpt.get("max_beta", 0.02)
    epoch = ckpt.get("epoch", "?")
    network = MyUNet_Attn(n_steps=n_steps, time_emb_dim=256)
    ddpm = GaussianDDPM(network, n_steps=n_steps,
                        min_beta=min_beta, max_beta=max_beta, device=device)
    ddpm.load_state_dict(ckpt["model_state_dict"])
    ddpm = ddpm.to(device)
    ddpm.eval()
    print(f"Loaded DDPM checkpoint (epoch {epoch}, n_steps={n_steps})")

    val_data = dd.get_validation_data()
    print(f"Validation set: {len(val_data)} samples")

    # Also load GP-Diff results for direct comparison
    gpdiff_samples = gpdiff_data.get("samples", [])

    # Resume from checkpoint?
    completed_samples = []
    start_idx = 0
    if not args.fresh and os.path.exists(PT_PATH):
        existing = torch.load(PT_PATH, map_location="cpu", weights_only=False)
        completed_samples = existing.get("samples", [])
        start_idx = len(completed_samples)
        if start_idx >= n_total:
            print(f"Already have {start_idx} samples, nothing to do.")
            return
        print(f"Resuming from sample {start_idx + 1}")

    t0_global = time.time()
    print(f"\nVoronoi warm-start eval: S{args.max_stages} adaptive RePaint, gamma={args.gamma}")
    print(f"  S1: t={args.t_start}, floor={args.noise_floor}")
    print(f"  S2+: t={args.t_refine}, floor={args.noise_floor_refine}, "
          f"var_decay={args.var_decay}")
    print(f"{'='*90}")
    print(f"{'#':>4}  {'ValIdx':>7} {'Type':>7}  {'Vor-Diff MSE':>12}  "
          f"{'GP-Diff MSE':>12}  {'GP MSE':>10}  {'Time':>7}")
    print(f"{'-'*90}")

    for run_i in range(start_idx, n_total):
        t0 = time.time()
        vi = val_indices[run_i]
        is_eddy = vi in eddy_set
        input_image = val_data[vi][0].unsqueeze(0).to(device)
        input_orig = standardizer.unstandardize(
            input_image.squeeze(0)
        ).to(device).unsqueeze(0)

        # Mask (same as GP-Diff)
        land_mask = (input_orig.abs() > 1e-5).float().to(device)
        raw_mask = get_fixed_center_mask(input_image.shape).to(device)
        missing_mask = raw_mask * land_mask

        # Build observation mask for Voronoi (1=known, 0=missing)
        # missing_mask is (1,2,H,W) — take max across channels then invert
        miss_2d = missing_mask[0].max(dim=0).values  # (H,W), 1=missing
        obs_mask_2d = (1.0 - miss_2d).cpu().numpy()   # (H,W), 1=known
        ocean_mask_2d = land_mask[0].max(dim=0).values.cpu().numpy()  # (H,W)

        # ── Voronoi fill (replaces GP) ──
        voronoi_field, dist_variance = voronoi_fill(
            input_orig, obs_mask_2d, ocean_mask_2d,
        )

        # Compute Voronoi-only MSE (to compare with GP MSE)
        voronoi_field_dev = voronoi_field.to(device)
        diff_vor = (voronoi_field_dev - input_orig) * missing_mask
        vor_only_mse = (diff_vor ** 2).sum() / (missing_mask.sum() + 1e-8)

        # Standardize Voronoi fill for DDPM input
        vor_std = standardizer(voronoi_field_dev.squeeze(0)).to(device).unsqueeze(0)

        # Multi-stage refinement (identical loop to GP-Diff, but with Voronoi prior)
        current_prior = vor_std.clone()
        current_var = dist_variance.to(device)

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
                prior_image=current_prior,
                var_map=current_var,
                t_start=t_s, noise_floor=nf,
                seed=seed_s,
                noise_strategy=noise_strategy,
                resample_steps=args.resample_steps,
                gamma=args.gamma,
                device=device,
            )
            current_prior = stage_out.clone()

        # Final output in physical space
        vordiff_phys = standardizer.unstandardize(
            stage_out.squeeze(0)
        ).to(device).unsqueeze(0)
        vordiff_mse = ((vordiff_phys - input_orig) * missing_mask).pow(2).sum() / (
            missing_mask.sum() + 1e-8
        )

        # Grab GP-Diff MSE for comparison
        gpdiff_mse = gpdiff_samples[run_i]["gp_mse"] if run_i < len(gpdiff_samples) else float("nan")
        ddpm_mse = gpdiff_samples[run_i]["ddpm_mse"] if run_i < len(gpdiff_samples) else float("nan")

        elapsed = time.time() - t0
        tag = "EDDY" if is_eddy else "clean"
        print(f"{run_i+1:>4}  {vi:>7} {tag:>7}  {vordiff_mse.item():>12.6f}  "
              f"{ddpm_mse:>12.6f}  {gpdiff_mse:>10.6f}  {elapsed:>6.1f}s")

        completed_samples.append({
            "idx": run_i,
            "val_idx": vi,
            "is_eddy_sample": is_eddy,
            "ground_truth": input_orig.cpu(),
            "voronoi_fill": voronoi_field.cpu(),
            "vordiff_output": vordiff_phys.cpu(),
            "missing_mask": missing_mask.cpu(),
            "land_mask": land_mask.cpu(),
            "vor_only_mse": vor_only_mse.item(),
            "vordiff_mse": vordiff_mse.item(),
            "gpdiff_mse": ddpm_mse,
            "gp_mse": gpdiff_mse,
        })

        # Checkpoint every 10 samples
        if (run_i + 1) % 10 == 0 or (run_i + 1) == n_total:
            _save_pt(completed_samples, val_indices, eddy_set, noeddy_set)
            print(f"  [checkpoint saved: {len(completed_samples)} samples]")

    _save_pt(completed_samples, val_indices, eddy_set, noeddy_set)

    elapsed_total = time.time() - t0_global
    n = len(completed_samples)
    print(f"\n{'='*90}")
    print(f"Inference complete: {n} samples in {elapsed_total:.0f}s "
          f"({elapsed_total/n:.1f}s/sample)")
    _print_summary(completed_samples)


def _save_pt(samples, val_indices, eddy_set, noeddy_set):
    torch.save({
        "samples": samples,
        "n_samples": len(samples),
        "val_indices": val_indices,
        "eddy_indices": sorted(eddy_set),
        "noeddy_indices": sorted(noeddy_set),
        "config": {
            "method": "voronoi_warmstart",
            "max_stages": args.max_stages,
            "t_start": args.t_start,
            "t_refine": args.t_refine,
            "resample_steps": args.resample_steps,
            "noise_floor": args.noise_floor,
            "noise_floor_refine": args.noise_floor_refine,
            "var_decay": args.var_decay,
            "gamma": args.gamma,
            "seed": args.seed,
            "variance_proxy": "distance_squared",
        },
    }, PT_PATH)


def _print_summary(samples):
    n = len(samples)
    vordiff_mses = [s["vordiff_mse"] for s in samples]
    gpdiff_mses = [s["gpdiff_mse"] for s in samples]
    gp_mses = [s["gp_mse"] for s in samples]
    vor_only_mses = [s["vor_only_mse"] for s in samples]

    print(f"\n{'='*70}")
    print(f"RECONSTRUCTION MSE COMPARISON ({n} samples)")
    print(f"{'='*70}")
    print(f"  {'Method':<20} {'Mean MSE':>12} {'Median MSE':>12}")
    print(f"  {'-'*44}")
    print(f"  {'GP':<20} {np.mean(gp_mses):>12.6f} {np.median(gp_mses):>12.6f}")
    print(f"  {'GP-Diff (S6)':<20} {np.mean(gpdiff_mses):>12.6f} {np.median(gpdiff_mses):>12.6f}")
    print(f"  {'Voronoi (no diff)':<20} {np.mean(vor_only_mses):>12.6f} {np.median(vor_only_mses):>12.6f}")
    print(f"  {'Vor-Diff (S6)':<20} {np.mean(vordiff_mses):>12.6f} {np.median(vordiff_mses):>12.6f}")

    # Win rates
    vor_beats_gp = sum(1 for v, g in zip(vordiff_mses, gp_mses) if v < g)
    vor_beats_gpdiff = sum(1 for v, d in zip(vordiff_mses, gpdiff_mses) if v < d)
    gpdiff_beats_gp = sum(1 for d, g in zip(gpdiff_mses, gp_mses) if d < g)

    print(f"\n  Win rates:")
    print(f"    Vor-Diff beats GP:      {vor_beats_gp}/{n} ({100*vor_beats_gp/n:.1f}%)")
    print(f"    Vor-Diff beats GP-Diff: {vor_beats_gpdiff}/{n} ({100*vor_beats_gpdiff/n:.1f}%)")
    print(f"    GP-Diff beats GP:       {gpdiff_beats_gp}/{n} ({100*gpdiff_beats_gp/n:.1f}%)")

    # By eddy/non-eddy
    eddy_vd = [s["vordiff_mse"] for s in samples if s["is_eddy_sample"]]
    eddy_gd = [s["gpdiff_mse"] for s in samples if s["is_eddy_sample"]]
    clean_vd = [s["vordiff_mse"] for s in samples if not s["is_eddy_sample"]]
    clean_gd = [s["gpdiff_mse"] for s in samples if not s["is_eddy_sample"]]

    if eddy_vd:
        print(f"\n  Eddy samples ({len(eddy_vd)}):")
        print(f"    Vor-Diff mean: {np.mean(eddy_vd):.6f}  GP-Diff mean: {np.mean(eddy_gd):.6f}")
    if clean_vd:
        print(f"  Non-eddy samples ({len(clean_vd)}):")
        print(f"    Vor-Diff mean: {np.mean(clean_vd):.6f}  GP-Diff mean: {np.mean(clean_gd):.6f}")


# ═══════════════════════════════════════════════════════════════════
#  EDDY DETECTION EVAL (same protocol as GP-Diff eval)
# ═══════════════════════════════════════════════════════════════════

def crop_to_ocean(tensor_4d):
    """(1,2,H,W) → (2, OCEAN_H, OCEAN_W) tensor"""
    return tensor_4d.squeeze(0)[:, :OCEAN_H, :OCEAN_W]


def run_gamma1(vel):
    """vel: (2,H,W) tensor → list of Eddy"""
    vel = torch.nan_to_num(vel, nan=0.0)
    eddies, _, _ = detect_eddies_gamma(
        vel, radius=RADIUS, gamma_threshold=GAMMA_THRESH,
        min_area=MIN_AREA, shore_buffer=SHORE_BUFFER,
        smooth_sigma=SMOOTH_SIGMA, min_mean_speed_ratio=MIN_SPEED_RATIO,
        min_vorticity=MIN_VORTICITY,
    )
    return eddies


def match_eddies(gt_eddies, pred_eddies, dist_thresh):
    """Greedy nearest-neighbour matching (same as production eval)."""
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


def run_eddy_eval():
    data = torch.load(PT_PATH, map_location="cpu", weights_only=False)
    samples = data["samples"]
    eddy_set = set(data.get("eddy_indices", []))

    print(f"\n{'='*70}")
    print(f"EDDY DETECTION EVALUATION (Gamma1)")
    print(f"{'='*70}")

    vd_stats = defaultdict(list)
    vd_noeddy_fp = 0
    n_eddy_samples, n_noeddy_samples = 0, 0

    for s in samples:
        is_eddy = s["is_eddy_sample"]
        gt_vel = crop_to_ocean(s["ground_truth"])
        vd_vel = crop_to_ocean(s["vordiff_output"])

        gt_e = run_gamma1(gt_vel)
        vd_e = run_gamma1(vd_vel)
        n_gt = len(gt_e)

        if n_gt > 0:
            n_eddy_samples += 1
            vd_m, vd_fn, vd_fp = match_eddies(gt_e, vd_e, DIST_THRESH)
            vd_stats["detected"].append(len(vd_m))
            vd_stats["total_gt"].append(n_gt)
            vd_stats["false_pos"].append(len(vd_fp))
            vd_stats["distances"].extend([m[2] for m in vd_m])
            vd_stats["area_ratios"].extend(
                [vd_e[m[1]].area_pixels / gt_e[m[0]].area_pixels for m in vd_m]
            )
        else:
            n_noeddy_samples += 1
            vd_noeddy_fp += len(vd_e)

    total_gt = sum(vd_stats["total_gt"]) if vd_stats["total_gt"] else 0
    tp = sum(vd_stats["detected"])
    fp_eddy = sum(vd_stats["false_pos"])
    fp = fp_eddy + vd_noeddy_fp
    fn = total_gt - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / total_gt if total_gt > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n  Samples:  {len(samples)} total ({n_eddy_samples} eddy, {n_noeddy_samples} non-eddy)")
    print(f"  GT eddies:  {total_gt}")
    print(f"  TP={tp}  FP={fp} ({fp_eddy} on eddy + {vd_noeddy_fp} on clean)  FN={fn}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1:        {f1:.3f}")

    if vd_stats["distances"]:
        dists = vd_stats["distances"]
        print(f"\n  Localization (TP={tp}):")
        print(f"    Median distance: {np.median(dists):.2f} px")
        print(f"    Mean distance:   {np.mean(dists):.2f} px")
        within2 = sum(1 for d in dists if d <= 2)
        print(f"    Within 2px: {within2}/{tp} ({100*within2/tp:.0f}%)")
    if vd_stats["area_ratios"]:
        print(f"    Median area ratio: {np.median(vd_stats['area_ratios']):.2f}")

    # Comparison table with known baselines
    print(f"\n{'='*70}")
    print(f"FULL COMPARISON TABLE")
    print(f"{'='*70}")
    print(f"  {'Method':<20} {'TP':>4} {'FP':>4} {'FN':>4} {'Prec':>7} {'Recall':>7} {'F1':>7}")
    print(f"  {'-'*56}")
    print(f"  {'GP':<20} {'1':>4} {'0':>4} {'56':>4} {'1.000':>7} {'0.018':>7} {'0.034':>7}")
    print(f"  {'GP-Diff (S6)':<20} {'15':>4} {'4':>4} {'42':>4} {'0.789':>7} {'0.263':>7} {'0.395':>7}")
    print(f"  {'Voronoi-CNN':<20} {'28':>4} {'7':>4} {'29':>4} {'0.800':>7} {'0.491':>7} {'0.609':>7}")
    print(f"  {'Vor-Diff (S6)':<20} {tp:>4} {fp:>4} {fn:>4} {precision:>7.3f} {recall:>7.3f} {f1:>7.3f}")


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if args.eval_only:
        if not os.path.exists(PT_PATH):
            print(f"No saved results at {PT_PATH}")
            sys.exit(1)
        data = torch.load(PT_PATH, map_location="cpu", weights_only=False)
        _print_summary(data["samples"])
        run_eddy_eval()
    else:
        run_inference()
        if os.path.exists(PT_PATH):
            run_eddy_eval()
