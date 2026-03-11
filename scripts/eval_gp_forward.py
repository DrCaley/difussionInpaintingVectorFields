#!/usr/bin/env python
"""Evaluate the GP-forward trained FiLM+Attn model.

This model was trained with gp_forward=True:
  Forward:  x_t = √ᾱ_t · μ_GP + √(1-ᾱ_t) · ε
  Target:   predict x̂₀ = v_GT  (x0-prediction)

We test two inference strategies:
  A) x0_full_reverse_inpaint — full 250-step reverse from pure noise
  B) x0_film_gp_repaint — GP-warm-start with variance-adaptive RePaint

Usage:
    PYTHONPATH=. python scripts/eval_gp_forward.py --quick 5          # 5 samples, both methods
    PYTHONPATH=. python scripts/eval_gp_forward.py --quick 5 --method full  # full-reverse only
    PYTHONPATH=. python scripts/eval_gp_forward.py --quick 5 --method gp    # GP-warm only
    PYTHONPATH=. python scripts/eval_gp_forward.py                    # all 100 samples
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
from ddpm.utils.inpainting_utils import x0_full_reverse_inpaint, x0_film_gp_repaint
from ddpm.utils.noise_utils import get_noise_strategy
from ddpm.helper_functions.interpolation_tool import gp_fill

# ── args ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--quick", type=int, default=0,
                    help="Run only this many samples (0 = all 100)")
parser.add_argument("--method", choices=["both", "full", "gp"], default="both",
                    help="Which inference method: full-reverse, GP-warm, or both")
parser.add_argument("--fresh", action="store_true",
                    help="Ignore existing .pt and start fresh")
# GP-warm hyperparams
parser.add_argument("--t-start", type=int, default=75)
parser.add_argument("--t-refine", type=int, default=50)
parser.add_argument("--max-stages", type=int, default=6)
parser.add_argument("--resample-steps", type=int, default=5)
parser.add_argument("--noise-floor", type=float, default=0.2)
parser.add_argument("--noise-floor-refine", type=float, default=0.3)
parser.add_argument("--var-decay", type=float, default=0.1)
parser.add_argument("--gamma", type=float, default=3.0)
args = parser.parse_args()

# ── paths ─────────────────────────────────────────────────────────────
GP_FWD_WEIGHTS = (
    "experiments/06_gp_forward/gp_finetune/results/"
    "inpaint_forward_diff_div_free_t250_best_weights.pt"
)
GPDIFF_PT = "results/eddy_balanced_eval/bulk_eval_eddy_balanced_100.pt"
OUT_DIR = "results/gp_forward_eval"
os.makedirs(OUT_DIR, exist_ok=True)
PT_PATH = os.path.join(OUT_DIR, "gp_forward_eval.pt")

# ── model config ──────────────────────────────────────────────────────
N_STEPS = 250
MIN_BETA = 0.0001
MAX_BETA = 0.02
NOISE_FN = "forward_diff_div_free"
SHARED_MEAN = -0.05084468695562498
SHARED_STD = 0.11479844598042026
OCEAN_H, OCEAN_W = 44, 94

# ── fixed center mask (row-22 known) ─────────────────────────────────
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
    # Load same 100-sample evaluation set as GP-Diff
    gpdiff_data = torch.load(GPDIFF_PT, map_location="cpu", weights_only=False)
    val_indices = gpdiff_data["val_indices"]
    eddy_set = set(gpdiff_data.get("eddy_indices", []))
    noeddy_set = set(gpdiff_data.get("noeddy_indices", []))
    n_total = len(val_indices)
    if args.quick > 0:
        n_total = min(args.quick, n_total)

    dd = DDInitializer()
    device = dd.get_device()
    standardizer = dd.get_standardizer()
    noise_strategy = get_noise_strategy(NOISE_FN)

    # Build and load GP-forward trained model
    weights_path = os.path.join(BASE_DIR, GP_FWD_WEIGHTS)
    print(f"Loading GP-forward weights from: {weights_path}")
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
    print(f"GP-forward FiLM+Attn: {n_params:.1f}M params, T={N_STEPS}")
    print(f"Training: gp_forward=true, x0-prediction, mask_xt=true")
    print(f"Testing methods: {args.method}")
    if args.method in ("both", "gp"):
        print(f"  GP-warm: S={args.max_stages}, t_start={args.t_start}, "
              f"resample={args.resample_steps}, gamma={args.gamma}")

    val_data = dd.get_validation_data()
    print(f"Validation set: {len(val_data)} samples, running {n_total}")

    do_full = args.method in ("both", "full")
    do_gp = args.method in ("both", "gp")

    # Also load GP-Diff and other baselines for comparison
    gpdiff_mses = [s["ddpm_mse"] for s in gpdiff_data["samples"]]
    gp_ref_mses = [s["gp_mse"] for s in gpdiff_data["samples"]]

    t0_global = time.time()
    completed_samples = []

    header = f"{'#':>4}  {'ValIdx':>7} {'Type':>7}  {'GP MSE':>10}"
    if do_full:
        header += f"  {'Full-Rev':>10}"
    if do_gp:
        header += f"  {'GP-Warm':>10}"
    header += f"  {'GP-Diff*':>10}  {'Time':>7}"
    print(f"\n{'='*len(header)}")
    print(header)
    print(f"{'-'*len(header)}")

    for run_i in range(n_total):
        t0 = time.time()
        vi = val_indices[run_i]
        is_eddy = vi in eddy_set
        tag = "EDDY" if is_eddy else "clean"

        input_image = val_data[vi][0].unsqueeze(0).to(device)
        input_orig = standardizer.unstandardize(
            input_image.squeeze(0)
        ).to(device).unsqueeze(0)

        # Build mask
        land_mask = (input_orig.abs() > 1e-5).float().to(device)
        raw_mask = get_fixed_center_mask(input_image.shape).to(device)
        missing_mask_1ch = raw_mask * land_mask[:, 0:1]
        missing_mask = missing_mask_1ch.expand(-1, 2, -1, -1)

        # GP fill (needed for GP-warm and as baseline)
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
        gp_mse = ((gp_out - input_orig) * missing_mask).pow(2).sum() / (
            missing_mask.sum() + 1e-8)

        sample_result = {
            "idx": run_i,
            "val_idx": vi,
            "is_eddy_sample": is_eddy,
            "ground_truth": input_orig.cpu(),
            "gp_output": gp_out.cpu(),
            "missing_mask": missing_mask.cpu(),
            "land_mask": land_mask.cpu(),
            "gp_mse": gp_mse.item(),
        }

        line = f"{run_i+1:>4}  {vi:>7} {tag:>7}  {gp_mse.item():>10.6f}"

        # === Method A: Full reverse from noise ===
        if do_full:
            torch.manual_seed(args.seed + run_i)
            with torch.no_grad():
                full_std = x0_full_reverse_inpaint(
                    ddpm, input_image, missing_mask,
                    n_samples=1, device=device,
                    noise_strategy=noise_strategy,
                    mask_xt=True,
                )
            full_phys = standardizer.unstandardize(
                full_std.squeeze(0)
            ).to(device).unsqueeze(0)
            full_mse = ((full_phys - input_orig) * missing_mask).pow(2).sum() / (
                missing_mask.sum() + 1e-8)
            sample_result["full_reverse_output"] = full_phys.cpu()
            sample_result["full_reverse_mse"] = full_mse.item()
            line += f"  {full_mse.item():>10.6f}"

        # === Method B: GP-warm-start multi-stage ===
        if do_gp:
            gp_std = standardizer(gp_out.squeeze(0)).to(device).unsqueeze(0)
            current_prior = gp_std.clone()
            current_var = gp_var_map.clone()

            for stage in range(1, args.max_stages + 1):
                if stage == 1:
                    t_s = args.t_start
                    nf = args.noise_floor
                    seed_s = args.seed + run_i
                else:
                    t_s = args.t_refine
                    nf = args.noise_floor_refine
                    seed_s = args.seed + run_i + stage * 10000
                    current_var = current_var * args.var_decay

                torch.manual_seed(seed_s)
                with torch.no_grad():
                    stage_out = x0_film_gp_repaint(
                        ddpm, input_image, missing_mask,
                        gp_image=current_prior,
                        gp_variance_map=current_var,
                        t_start=t_s, noise_floor=nf,
                        n_samples=1, device=device,
                        noise_strategy=noise_strategy,
                        mask_xt=True,
                        resample_steps=args.resample_steps,
                        gamma=args.gamma,
                    )
                current_prior = stage_out.clone()

            gp_warm_phys = standardizer.unstandardize(
                stage_out.squeeze(0)
            ).to(device).unsqueeze(0)
            gp_warm_mse = ((gp_warm_phys - input_orig) * missing_mask).pow(2).sum() / (
                missing_mask.sum() + 1e-8)
            sample_result["gp_warm_output"] = gp_warm_phys.cpu()
            sample_result["gp_warm_mse"] = gp_warm_mse.item()
            line += f"  {gp_warm_mse.item():>10.6f}"

        # GP-Diff reference
        gpdiff_ref = gpdiff_mses[run_i] if run_i < len(gpdiff_mses) else float("nan")
        line += f"  {gpdiff_ref:>10.6f}"

        elapsed = time.time() - t0
        line += f"  {elapsed:>6.1f}s"
        print(line)

        completed_samples.append(sample_result)

    # Save results
    torch.save({
        "samples": completed_samples,
        "n_samples": len(completed_samples),
        "val_indices": val_indices[:n_total],
        "eddy_indices": sorted(eddy_set),
        "noeddy_indices": sorted(noeddy_set),
        "model": "GP-forward FiLM+Attn (epoch 62 best weights)",
        "training": "gp_forward=true, x0-prediction, mask_xt=true, no EMA",
        "methods_tested": args.method,
    }, PT_PATH)

    # Summary
    elapsed_total = time.time() - t0_global
    n = len(completed_samples)
    gp_mses_all = [s["gp_mse"] for s in completed_samples]

    print(f"\n{'='*80}")
    print(f"GP-Forward Model Evaluation — {n} samples in {elapsed_total:.0f}s")
    print(f"{'='*80}")
    print(f"\n{'Method':<20} {'Mean MSE':>10} {'Median':>10} {'vs GP':>8} {'Wins':>8}")
    print(f"{'-'*60}")

    print(f"{'GP baseline':<20} {np.mean(gp_mses_all):>10.6f} "
          f"{np.median(gp_mses_all):>10.6f} {'1.000x':>8} {'---':>8}")

    gpdiff_sub = gpdiff_mses[:n]
    gpdiff_ratio = np.mean(gpdiff_sub) / np.mean(gp_mses_all)
    print(f"{'GP-Diff (ref)':<20} {np.mean(gpdiff_sub):>10.6f} "
          f"{np.median(gpdiff_sub):>10.6f} {gpdiff_ratio:>7.3f}x {'---':>8}")

    if do_full:
        full_mses = [s["full_reverse_mse"] for s in completed_samples]
        ratio = np.mean(full_mses) / np.mean(gp_mses_all)
        wins = sum(1 for f, g in zip(full_mses, gp_mses_all) if f < g)
        print(f"{'Full-reverse':<20} {np.mean(full_mses):>10.6f} "
              f"{np.median(full_mses):>10.6f} {ratio:>7.3f}x "
              f"{wins}/{n}")

    if do_gp:
        gp_warm_mses = [s["gp_warm_mse"] for s in completed_samples]
        ratio = np.mean(gp_warm_mses) / np.mean(gp_mses_all)
        wins = sum(1 for f, g in zip(gp_warm_mses, gp_mses_all) if f < g)
        print(f"{'GP-warm (S6)':<20} {np.mean(gp_warm_mses):>10.6f} "
              f"{np.median(gp_warm_mses):>10.6f} {ratio:>7.3f}x "
              f"{wins}/{n}")

    # Voronoi-CNN reference
    print(f"{'Voronoi-CNN (ref)':<20} {'0.001400':>10} {'---':>10} "
          f"{'0.269x':>8} {'---':>8}")

    print(f"{'='*80}")


if __name__ == "__main__":
    run_inference()
