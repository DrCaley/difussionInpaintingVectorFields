#!/usr/bin/env python
"""Bulk evaluation: 50 eddy + 50 non-eddy samples (fresh, no overlap with prior eval).

Runs GP + S6 adaptive GP-init RePaint (gamma=3) on each sample, then
runs eddy detection and produces a full summary.

Usage:
    PYTHONPATH=. python scripts/bulk_eval_eddy_balanced.py
    PYTHONPATH=. python scripts/bulk_eval_eddy_balanced.py --plot-only
    PYTHONPATH=. python scripts/bulk_eval_eddy_balanced.py --eval-only
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
from ddpm.neural_networks.unets.unet_xl_attn import MyUNet_Attn
from ddpm.helper_functions.masks.border_mask import BorderMaskGenerator
from ddpm.utils.inpainting_utils import repaint_gp_init_adaptive
from ddpm.utils.noise_utils import get_noise_strategy
from ddpm.helper_functions.interpolation_tool import gp_fill
from ddpm.utils.eddy_detection import detect_eddies_gamma
from plots.visualization_tools.standard_plots import plot_inpaint_panels
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
parser.add_argument("--fresh", action="store_true",
                    help="Ignore existing checkpoint and start fresh")
parser.add_argument("--plot-only", action="store_true",
                    help="Skip inference, just regenerate plots + eval from saved .pt")
parser.add_argument("--eval-only", action="store_true",
                    help="Skip inference and plots, just run eddy eval from saved .pt")
parser.add_argument("--skip-plots", action="store_true",
                    help="Skip plot generation")
args = parser.parse_args()

# ── paths ────────────────────────────────────────────────────────────
MODEL_CKPT = (
    "experiments/02_inpaint_algorithm/repaint_gaussian_attn/results/"
    "inpaint_gaussian_t250_best_checkpoint.pt"
)
OUT_DIR = "results/eddy_balanced_eval"
os.makedirs(OUT_DIR, exist_ok=True)
PT_PATH = os.path.join(OUT_DIR, "bulk_eval_eddy_balanced_100.pt")
PLOT_DIR = os.path.join(OUT_DIR, "quiver_plots")
SUMMARY_PATH = os.path.join(OUT_DIR, "eddy_eval_summary.txt")

# ── eddy detection params ───────────────────────────────────────────
RADIUS = 8
GAMMA_THRESH = 0.65
MIN_AREA = 25
SHORE_BUFFER = 2
SMOOTH_SIGMA = 2.0
MIN_SPEED_RATIO = 0.3
MIN_VORTICITY = 0.03
DIST_THRESH = 10.0
OCEAN_H, OCEAN_W = 44, 94


# ── sample selection ─────────────────────────────────────────────────
def select_indices():
    """Pick 50 eddy + 50 non-eddy val indices, disjoint from prior eval."""
    # Load prior eval indices to exclude
    prior_pt = (
        "experiments/02_inpaint_algorithm/repaint_gaussian_attn/results/"
        "bulk_eval_best_100samples.pt"
    )
    prior = torch.load(prior_pt, map_location="cpu", weights_only=False)
    used = set(prior["val_indices"])

    # Load eddy catalogue (already rebuilt with min_vorticity=0.03)
    cat = torch.load("results/val_eddy_catalogue.pt",
                      map_location="cpu", weights_only=False)
    eddy_set = set(cat["eddy_indices"])
    n_val = cat["n_val_total"]

    eddy_avail = sorted(eddy_set - used)
    noeddy_avail = sorted(set(range(n_val)) - eddy_set - used)

    rng = np.random.RandomState(123)  # fixed seed for reproducibility
    eddy50 = sorted(rng.choice(eddy_avail, size=50, replace=False).tolist())
    noeddy50 = sorted(rng.choice(noeddy_avail, size=50, replace=False).tolist())

    # Interleave: eddy first, then non-eddy
    all_indices = eddy50 + noeddy50
    return all_indices, set(eddy50), set(noeddy50)


# ── fixed center mask ───────────────────────────────────────────────
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
    val_indices, eddy_set, noeddy_set = select_indices()
    n_total = len(val_indices)
    print(f"Selected {len(eddy_set)} eddy + {len(noeddy_set)} non-eddy = "
          f"{n_total} fresh samples")

    dd = DDInitializer()
    device = dd.get_device()
    noise_strategy = get_noise_strategy("gaussian")
    standardizer = dd.get_standardizer()

    # Load model
    ckpt = torch.load(MODEL_CKPT, map_location=device, weights_only=False)
    n_steps  = ckpt.get("n_steps", 250)
    min_beta = ckpt.get("min_beta", 0.0001)
    max_beta = ckpt.get("max_beta", 0.02)
    epoch    = ckpt.get("epoch", "?")
    network = MyUNet_Attn(n_steps=n_steps, time_emb_dim=256)
    ddpm = GaussianDDPM(network, n_steps=n_steps,
                        min_beta=min_beta, max_beta=max_beta, device=device)
    ddpm.load_state_dict(ckpt["model_state_dict"])
    ddpm = ddpm.to(device)
    ddpm.eval()
    print(f"Loaded checkpoint (epoch {epoch}, n_steps={n_steps})")

    val_data = dd.get_validation_data()
    print(f"Validation set: {len(val_data)} samples")

    # Resume from checkpoint?
    completed_samples = []
    start_idx = 0
    if not args.fresh and os.path.exists(PT_PATH):
        existing = torch.load(PT_PATH, map_location="cpu", weights_only=False)
        completed_samples = existing.get("samples", [])
        if "val_indices" in existing:
            val_indices = existing["val_indices"]
            eddy_set = set(existing.get("eddy_indices", []))
            noeddy_set = set(existing.get("noeddy_indices", []))
        start_idx = len(completed_samples)
        if start_idx >= n_total:
            print(f"Already have {start_idx} samples in {PT_PATH}, nothing to do.")
            return val_indices, eddy_set, noeddy_set
        print(f"Resuming from sample {start_idx + 1} ({start_idx} already done)")

    t0_global = time.time()
    print(f"\nBulk eval: S{args.max_stages} adaptive GP-init RePaint, gamma={args.gamma}")
    print(f"  S1: t={args.t_start}, floor={args.noise_floor}")
    print(f"  S2+: t={args.t_refine}, floor={args.noise_floor_refine}, "
          f"var_decay={args.var_decay}")
    print(f"{'='*80}")
    print(f"{'#':>4}  {'ValIdx':>7} {'Type':>7}  {'GP MSE':>10}  "
          f"{'DDPM MSE':>10}  {'Ratio':>7}  {'Time':>7}")
    print(f"{'-'*80}")

    for run_i in range(start_idx, n_total):
        t0 = time.time()
        vi = val_indices[run_i]
        is_eddy = vi in eddy_set
        input_image = val_data[vi][0].unsqueeze(0).to(device)
        input_orig = standardizer.unstandardize(
            input_image.squeeze(0)
        ).to(device).unsqueeze(0)

        # Mask
        land_mask = (input_orig.abs() > 1e-5).float().to(device)
        raw_mask = get_fixed_center_mask(input_image.shape).to(device)
        missing_mask = raw_mask * land_mask

        # GP
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
        diff_gp = (gp_out - input_orig) * missing_mask
        gp_mse = (diff_gp ** 2).sum() / (missing_mask.sum() + 1e-8)
        gp_std = standardizer(gp_out.squeeze(0)).to(device).unsqueeze(0)

        # Multi-stage refinement
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
        ddpm_phys = standardizer.unstandardize(
            stage_out.squeeze(0)
        ).to(device).unsqueeze(0)
        ddpm_mse = ((ddpm_phys - input_orig) * missing_mask).pow(2).sum() / (
            missing_mask.sum() + 1e-8
        )
        ratio = ddpm_mse.item() / gp_mse.item()

        elapsed = time.time() - t0
        tag = "EDDY" if is_eddy else "clean"
        print(f"{run_i+1:>4}  {vi:>7} {tag:>7}  {gp_mse.item():>10.6f}  "
              f"{ddpm_mse.item():>10.6f}  {ratio:>6.3f}x  {elapsed:>6.1f}s")

        completed_samples.append({
            "idx": run_i,
            "val_idx": vi,
            "is_eddy_sample": is_eddy,
            "ground_truth": input_orig.cpu(),
            "gp_output": gp_out.cpu(),
            "ddpm_output": ddpm_phys.cpu(),
            "missing_mask": missing_mask.cpu(),
            "land_mask": land_mask.cpu(),
            "gp_mse": gp_mse.item(),
            "ddpm_mse": ddpm_mse.item(),
            "ratio": ratio,
        })

        # Checkpoint every 10 samples
        if (run_i + 1) % 10 == 0 or (run_i + 1) == n_total:
            _save_pt(completed_samples, val_indices, eddy_set, noeddy_set)
            print(f"  [checkpoint saved: {len(completed_samples)} samples]")

    _save_pt(completed_samples, val_indices, eddy_set, noeddy_set)

    elapsed_total = time.time() - t0_global
    n = len(completed_samples)
    ratios = [s["ratio"] for s in completed_samples]
    wins = sum(1 for r in ratios if r < 1.0)
    print(f"\n{'='*80}")
    print(f"Inference complete: {n} samples in {elapsed_total:.0f}s "
          f"({elapsed_total/n:.1f}s/sample)")
    print(f"  DDPM beats GP: {wins}/{n} ({100*wins/n:.1f}%)")
    print(f"  Avg ratio: {np.mean(ratios):.4f}x, "
          f"Median: {np.median(ratios):.4f}x")
    print(f"{'='*80}")

    return val_indices, eddy_set, noeddy_set


def _save_pt(samples, val_indices, eddy_set, noeddy_set):
    torch.save({
        "samples": samples,
        "n_samples": len(samples),
        "val_indices": val_indices,
        "eddy_indices": sorted(eddy_set),
        "noeddy_indices": sorted(noeddy_set),
        "config": {
            "max_stages": args.max_stages,
            "t_start": args.t_start,
            "t_refine": args.t_refine,
            "resample_steps": args.resample_steps,
            "noise_floor": args.noise_floor,
            "noise_floor_refine": args.noise_floor_refine,
            "var_decay": args.var_decay,
            "gamma": args.gamma,
            "seed": args.seed,
        },
        "eddy_detection_params": {
            "radius": RADIUS,
            "gamma_threshold": GAMMA_THRESH,
            "min_area": MIN_AREA,
            "shore_buffer": SHORE_BUFFER,
            "smooth_sigma": SMOOTH_SIGMA,
            "min_speed_ratio": MIN_SPEED_RATIO,
            "min_vorticity": MIN_VORTICITY,
        },
    }, PT_PATH)


# ═══════════════════════════════════════════════════════════════════
#  EDDY EVALUATION
# ═══════════════════════════════════════════════════════════════════

def crop_to_ocean(tensor_4d):
    return tensor_4d.squeeze(0)[:, :OCEAN_H, :OCEAN_W]


def run_gamma1(vel):
    vel = torch.nan_to_num(vel, nan=0.0)
    eddies, _, _ = detect_eddies_gamma(
        vel, radius=RADIUS, gamma_threshold=GAMMA_THRESH,
        min_area=MIN_AREA, shore_buffer=SHORE_BUFFER,
        smooth_sigma=SMOOTH_SIGMA, min_mean_speed_ratio=MIN_SPEED_RATIO,
        min_vorticity=MIN_VORTICITY,
    )
    return eddies


def match_eddies(gt_eddies, pred_eddies, dist_thresh):
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
    """Run eddy detection evaluation on the saved .pt file."""
    print(f"\n{'='*80}")
    print("EDDY DETECTION EVALUATION")
    print(f"{'='*80}\n")

    data = torch.load(PT_PATH, map_location="cpu", weights_only=False)
    samples = data["samples"]
    eddy_set = set(data.get("eddy_indices", []))

    gp_stats = defaultdict(list)
    ddpm_stats = defaultdict(list)
    gp_noeddy_fp, ddpm_noeddy_fp = 0, 0
    gp_noeddy_fp_details, ddpm_noeddy_fp_details = [], []
    n_eddy_samples, n_noeddy_samples = 0, 0

    # Also track MSE by category
    eddy_mses = {"gp": [], "ddpm": []}
    noeddy_mses = {"gp": [], "ddpm": []}

    print(f"{'#':>3} {'ValIdx':>7} {'Type':>6} {'GT':>3}  {'GP':>10}  "
          f"{'DDPM':>10}  {'Ratio':>7}  Notes")
    print("-" * 80)

    for i, s in enumerate(samples):
        vi = s["val_idx"]
        is_eddy = s.get("is_eddy_sample", vi in eddy_set)

        gt_vel = crop_to_ocean(s["ground_truth"])
        gp_vel = crop_to_ocean(s["gp_output"])
        ddpm_vel = crop_to_ocean(s["ddpm_output"])

        gt_e = run_gamma1(gt_vel)
        gp_e = run_gamma1(gp_vel)
        dd_e = run_gamma1(ddpm_vel)

        n_gt = len(gt_e)

        # Track MSE by category
        if is_eddy:
            eddy_mses["gp"].append(s["gp_mse"])
            eddy_mses["ddpm"].append(s["ddpm_mse"])
        else:
            noeddy_mses["gp"].append(s["gp_mse"])
            noeddy_mses["ddpm"].append(s["ddpm_mse"])

        if n_gt > 0:
            n_eddy_samples += 1
            gp_m, gp_fn, gp_fp = match_eddies(gt_e, gp_e, DIST_THRESH)
            dd_m, dd_fn, dd_fp = match_eddies(gt_e, dd_e, DIST_THRESH)

            gp_dists = [m[2] for m in gp_m]
            dd_dists = [m[2] for m in dd_m]
            gp_areas = [gp_e[m[1]].area_pixels / gt_e[m[0]].area_pixels
                        for m in gp_m]
            dd_areas = [dd_e[m[1]].area_pixels / gt_e[m[0]].area_pixels
                        for m in dd_m]

            gp_stats["detected"].append(len(gp_m))
            gp_stats["total_gt"].append(n_gt)
            gp_stats["false_pos"].append(len(gp_fp))
            gp_stats["distances"].extend(gp_dists)
            gp_stats["area_ratios"].extend(gp_areas)
            ddpm_stats["detected"].append(len(dd_m))
            ddpm_stats["total_gt"].append(n_gt)
            ddpm_stats["false_pos"].append(len(dd_fp))
            ddpm_stats["distances"].extend(dd_dists)
            ddpm_stats["area_ratios"].extend(dd_areas)

            gp_str = f"{len(gp_m)}/{n_gt}+{len(gp_fp)}fp"
            dd_str = f"{len(dd_m)}/{n_gt}+{len(dd_fp)}fp"
            dd_d = f"d={np.mean(dd_dists):.1f}" if dd_dists else "MISS"
            notes = f"EDDY  {dd_d}"
        else:
            n_noeddy_samples += 1
            gp_fp_count = len(gp_e)
            ddpm_fp_count = len(dd_e)
            gp_noeddy_fp += gp_fp_count
            ddpm_noeddy_fp += ddpm_fp_count
            if gp_fp_count > 0:
                gp_noeddy_fp_details.append((vi, gp_fp_count))
            if ddpm_fp_count > 0:
                ddpm_noeddy_fp_details.append((vi, ddpm_fp_count))

            gp_str = f"{gp_fp_count}fp" if gp_fp_count else "0fp"
            dd_str = f"{ddpm_fp_count}fp!" if ddpm_fp_count else "0fp"
            notes = "no-eddy" + (" HALLUC" if ddpm_fp_count else "")

        tag = "EDDY" if is_eddy else "clean"
        print(f"{i+1:>3} {vi:>7} {tag:>6} {n_gt:>3}  {gp_str:>10}  "
              f"{dd_str:>10}  {s['ratio']:>6.3f}x  {notes}")

    # ── Aggregate ────────────────────────────────────────────────
    total_gt = sum(gp_stats["total_gt"]) if gp_stats["total_gt"] else 0

    lines = []
    lines.append("=" * 70)
    lines.append("EDDY BALANCED EVALUATION — 50 eddy + 50 non-eddy (FRESH)")
    lines.append("=" * 70)
    lines.append(f"  Total samples:           {len(samples)}")
    lines.append(f"  Eddy samples (GT>0):     {n_eddy_samples}")
    lines.append(f"  Non-eddy samples (GT=0): {n_noeddy_samples}")
    lines.append(f"  Total GT eddies:         {total_gt}")
    lines.append(f"  Gamma1 params:           radius={RADIUS}, thresh={GAMMA_THRESH}, "
                 f"min_area={MIN_AREA}, shore_buffer={SHORE_BUFFER}, "
                 f"min_vorticity={MIN_VORTICITY}")

    # Overall MSE summary
    lines.append("\n" + "─" * 70)
    lines.append("  OVERALL MSE PERFORMANCE")
    lines.append("─" * 70)
    all_gp_mse = [s["gp_mse"] for s in samples]
    all_ddpm_mse = [s["ddpm_mse"] for s in samples]
    all_ratios = [s["ratio"] for s in samples]
    wins = sum(1 for r in all_ratios if r < 1.0)
    lines.append(f"\n  All 100 samples:")
    lines.append(f"    GP MSE:    mean={np.mean(all_gp_mse):.6f}, median={np.median(all_gp_mse):.6f}")
    lines.append(f"    DDPM MSE:  mean={np.mean(all_ddpm_mse):.6f}, median={np.median(all_ddpm_mse):.6f}")
    lines.append(f"    Ratio:     mean={np.mean(all_ratios):.4f}x, median={np.median(all_ratios):.4f}x")
    lines.append(f"    DDPM wins: {wins}/{len(samples)} ({100*wins/len(samples):.1f}%)")

    if eddy_mses["gp"]:
        e_ratios = [d/g for g, d in zip(eddy_mses["gp"], eddy_mses["ddpm"])]
        e_wins = sum(1 for r in e_ratios if r < 1.0)
        lines.append(f"\n  Eddy samples ({len(eddy_mses['gp'])}):")
        lines.append(f"    GP MSE:    mean={np.mean(eddy_mses['gp']):.6f}")
        lines.append(f"    DDPM MSE:  mean={np.mean(eddy_mses['ddpm']):.6f}")
        lines.append(f"    Ratio:     mean={np.mean(e_ratios):.4f}x")
        lines.append(f"    DDPM wins: {e_wins}/{len(e_ratios)} ({100*e_wins/len(e_ratios):.1f}%)")

    if noeddy_mses["gp"]:
        n_ratios = [d/g for g, d in zip(noeddy_mses["gp"], noeddy_mses["ddpm"])]
        n_wins = sum(1 for r in n_ratios if r < 1.0)
        lines.append(f"\n  Non-eddy samples ({len(noeddy_mses['gp'])}):")
        lines.append(f"    GP MSE:    mean={np.mean(noeddy_mses['gp']):.6f}")
        lines.append(f"    DDPM MSE:  mean={np.mean(noeddy_mses['ddpm']):.6f}")
        lines.append(f"    Ratio:     mean={np.mean(n_ratios):.4f}x")
        lines.append(f"    DDPM wins: {n_wins}/{len(n_ratios)} ({100*n_wins/len(n_ratios):.1f}%)")

    # Eddy detection stats
    lines.append("\n" + "─" * 70)
    lines.append("  EDDY DETECTION QUALITY")
    lines.append("─" * 70)

    for method, label, ms in [("gp", "GP", gp_stats), ("ddpm", "DDPM S6", ddpm_stats)]:
        total_det = sum(ms["detected"]) if ms["detected"] else 0
        total_fp = sum(ms["false_pos"]) if ms["false_pos"] else 0
        det_rate = total_det / total_gt if total_gt else 0
        dists = ms["distances"]
        areas = ms["area_ratios"]

        lines.append(f"\n  {label}:")
        lines.append(f"    True positives:    {total_det}/{total_gt} = {det_rate:.1%}")
        lines.append(f"    False positives:   {total_fp} (in {n_eddy_samples} eddy samples)")
        lines.append(f"    False negatives:   {total_gt - total_det}")
        if dists:
            lines.append(f"    Center distance:   mean={np.mean(dists):.2f}px, "
                         f"median={np.median(dists):.2f}px, max={np.max(dists):.2f}px")
        if areas:
            lines.append(f"    Area ratio (P/GT): mean={np.mean(areas):.2f}, "
                         f"median={np.median(areas):.2f}")

        det_rates = [d / t if t > 0 else 0
                     for d, t in zip(ms["detected"], ms["total_gt"])]
        n_perfect = sum(1 for r in det_rates if r >= 1.0)
        n_partial = sum(1 for r in det_rates if 0 < r < 1.0)
        n_miss = sum(1 for r in det_rates if r == 0)
        lines.append(f"    Per-sample:        {n_perfect} perfect, "
                     f"{n_partial} partial, {n_miss} total miss")

    # Non-eddy FPs
    lines.append("\n" + "─" * 70)
    lines.append("  NON-EDDY HALLUCINATIONS")
    lines.append("─" * 70)
    lines.append(f"\n  GP:     {gp_noeddy_fp} across {len(gp_noeddy_fp_details)} "
                 f"of {n_noeddy_samples} non-eddy samples")
    lines.append(f"  DDPM:   {ddpm_noeddy_fp} across {len(ddpm_noeddy_fp_details)} "
                 f"of {n_noeddy_samples} non-eddy samples")
    if ddpm_noeddy_fp_details:
        for vi, cnt in sorted(ddpm_noeddy_fp_details):
            lines.append(f"    val{vi}: {cnt} fake eddy(s)")

    # Precision / Recall
    lines.append("\n" + "─" * 70)
    lines.append("  PRECISION & RECALL")
    lines.append("─" * 70)
    for label, det, fp_e, fp_n in [
        ("GP", sum(gp_stats["detected"]) if gp_stats["detected"] else 0,
         sum(gp_stats["false_pos"]) if gp_stats["false_pos"] else 0, gp_noeddy_fp),
        ("DDPM S6", sum(ddpm_stats["detected"]) if ddpm_stats["detected"] else 0,
         sum(ddpm_stats["false_pos"]) if ddpm_stats["false_pos"] else 0, ddpm_noeddy_fp),
    ]:
        tp = det
        fp = fp_e + fp_n
        fn = total_gt - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        lines.append(f"\n  {label}:")
        lines.append(f"    TP={tp}  FP={fp}  FN={fn}")
        lines.append(f"    Precision: {prec:.1%}")
        lines.append(f"    Recall:    {rec:.1%}")
        lines.append(f"    F1:        {f1:.3f}")

    # Distance distribution
    dists = ddpm_stats["distances"]
    if dists:
        lines.append("\n" + "─" * 70)
        lines.append("  CENTER DISTANCE DISTRIBUTION (DDPM matched)")
        lines.append("─" * 70)
        for lo, hi in [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10)]:
            n = sum(1 for d in dists if lo <= d < hi)
            lines.append(f"    {lo}-{hi}px: {n} ({100*n/len(dists):.0f}%)")

    lines.append("\n" + "=" * 70)

    summary = "\n".join(lines)
    print(f"\n{summary}")

    with open(SUMMARY_PATH, "w") as f:
        f.write(summary + "\n")
    print(f"\nSaved → {SUMMARY_PATH}")


# ═══════════════════════════════════════════════════════════════════
#  PLOTS
# ═══════════════════════════════════════════════════════════════════

def generate_plots():
    os.makedirs(PLOT_DIR, exist_ok=True)
    data = torch.load(PT_PATH, map_location="cpu", weights_only=False)
    samples = data["samples"]
    print(f"\nGenerating plots for {len(samples)} samples → {PLOT_DIR}/")

    for i, s in enumerate(samples):
        vi = s["val_idx"]
        is_eddy = s.get("is_eddy_sample", False)
        mm_inv = 1.0 - s["missing_mask"]
        tag = "eddy" if is_eddy else "clean"
        prefix = f"{tag}_{i+1:03d}_val{vi}"

        plot_inpaint_panels(
            gt=s["ground_truth"],
            missing_mask=mm_inv,
            methods={"GP": s["gp_output"], "DDPM S6": s["ddpm_output"]},
            mse={"GP": s["gp_mse"], "DDPM S6": s["ddpm_mse"]},
            out_dir=PLOT_DIR,
            prefix=prefix,
            extra_titles={"DDPM S6": f"({s['ratio']:.3f}x GP)"},
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
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(samples)} done")

    print(f"All {len(samples)} plots saved → {PLOT_DIR}/")


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if args.eval_only:
        run_eddy_eval()
    elif args.plot_only:
        run_eddy_eval()
        generate_plots()
    else:
        run_inference()
        run_eddy_eval()
        if not args.skip_plots:
            generate_plots()
