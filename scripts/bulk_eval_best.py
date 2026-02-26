#!/usr/bin/env python
"""Bulk evaluation of current best setup: S6 adaptive GP-init RePaint (gamma=3).

1. Runs --n-samples through 6-stage refinement, saves all tensors to .pt
2. Generates side-by-side quiver plots (Ground Truth | GP | DDPM S6) for each.

Checkpoints every 10 samples so interrupted runs can resume.

Usage:
    PYTHONPATH=. python scripts/bulk_eval_best.py --n-samples 100
    PYTHONPATH=. python scripts/bulk_eval_best.py --n-samples 100 --fresh
    PYTHONPATH=. python scripts/bulk_eval_best.py --plot-only   # just re-plot from .pt
"""

import argparse, os, sys, time
from pathlib import Path

import torch
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import matplotlib
matplotlib.use('Agg')

from data_prep.data_initializer import DDInitializer
from ddpm.neural_networks.ddpm import GaussianDDPM
from ddpm.neural_networks.unets.unet_xl_attn import MyUNet_Attn
from ddpm.helper_functions.masks.border_mask import BorderMaskGenerator
from ddpm.utils.inpainting_utils import repaint_gp_init_adaptive
from ddpm.utils.noise_utils import get_noise_strategy
from ddpm.helper_functions.interpolation_tool import gp_fill

# ── args ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--n-samples", type=int, default=100)
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
parser.add_argument("--sequential", action="store_true",
                    help="Use sequential val indices 0..N-1 instead of random")
parser.add_argument("--plot-only", action="store_true",
                    help="Skip inference, just regenerate plots from saved .pt")
parser.add_argument("--skip-plots", action="store_true",
                    help="Skip plot generation (inference only)")
args = parser.parse_args()

# ── paths ────────────────────────────────────────────────────────────
MODEL_CKPT = (
    "experiments/02_inpaint_algorithm/repaint_gaussian_attn/results/"
    "inpaint_gaussian_t250_best_checkpoint.pt"
)
OUT_DIR = "experiments/02_inpaint_algorithm/repaint_gaussian_attn/results"
os.makedirs(OUT_DIR, exist_ok=True)
PT_PATH = os.path.join(OUT_DIR, f"bulk_eval_best_{args.n_samples}samples.pt")
PLOT_DIR = os.path.join(OUT_DIR, "quiver_plots")
os.makedirs(PLOT_DIR, exist_ok=True)


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
    """Run one adaptive-noise stage."""
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
#  PLOTTING  (uses standard_plots for consistent style)
# ═══════════════════════════════════════════════════════════════════

from plots.visualization_tools.standard_plots import plot_inpaint_panels


# ═══════════════════════════════════════════════════════════════════
#  INFERENCE
# ═══════════════════════════════════════════════════════════════════

def run_inference():
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
    n_val = len(val_data)
    print(f"Validation set: {n_val} samples")

    # Build index list: random shuffle (default) or sequential
    if args.sequential:
        val_indices = list(range(min(args.n_samples, n_val)))
    else:
        rng = np.random.RandomState(args.seed)
        val_indices = rng.choice(n_val, size=min(args.n_samples, n_val),
                                  replace=False).tolist()
    print(f"Sampling {'sequentially' if args.sequential else 'randomly'} "
          f"({len(val_indices)} indices from {n_val})")

    # Resume from checkpoint?
    completed_samples = []
    start_idx = 0
    if not args.fresh and os.path.exists(PT_PATH):
        existing = torch.load(PT_PATH, map_location="cpu", weights_only=False)
        completed_samples = existing.get("samples", [])
        # Reuse saved indices if resuming
        if "val_indices" in existing:
            val_indices = existing["val_indices"]
        start_idx = len(completed_samples)
        if start_idx >= args.n_samples:
            print(f"Already have {start_idx} samples in {PT_PATH}, nothing to do.")
            return
        print(f"Resuming from sample {start_idx + 1} ({start_idx} already done)")

    t0_global = time.time()
    n_total = args.n_samples

    print(f"\nBulk eval: S{args.max_stages} adaptive GP-init RePaint, gamma={args.gamma}")
    print(f"  S1: t={args.t_start}, floor={args.noise_floor}")
    print(f"  S2+: t={args.t_refine}, floor={args.noise_floor_refine}, "
          f"var_decay={args.var_decay}")
    print(f"{'='*80}")
    print(f"{'#':>4}  {'GP MSE':>10}  {'DDPM MSE':>10}  {'Ratio':>7}  {'Time':>7}")
    print(f"{'-'*80}")

    for run_i in range(start_idx, n_total):
        t0 = time.time()
        vi = val_indices[run_i]
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
                t_s = args.t_start
                nf = args.noise_floor
                seed_s = args.seed + run_i
            else:
                t_s = args.t_refine
                nf = args.noise_floor_refine
                seed_s = args.seed + run_i + stage * 10000
                current_var = current_var * args.var_decay

            stage_out = run_stage(
                ddpm, input_image, missing_mask,
                prior_image=current_prior,
                var_map=current_var,
                t_start=t_s,
                noise_floor=nf,
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
        print(f"{run_i+1:>4}  {gp_mse.item():>10.6f}  {ddpm_mse.item():>10.6f}  "
              f"{ratio:>6.3f}x  {elapsed:>6.1f}s")

        # Store all tensors on CPU for saving
        completed_samples.append({
            "idx": run_i,
            "val_idx": vi,
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
            _save_pt(completed_samples, val_indices)
            print(f"  [checkpoint saved: {len(completed_samples)} samples]")

    # Final save
    _save_pt(completed_samples, val_indices)

    # Summary
    elapsed_total = time.time() - t0_global
    n = len(completed_samples)
    ratios = [s["ratio"] for s in completed_samples]
    avg_ratio = sum(ratios) / n
    wins = sum(1 for r in ratios if r < 1.0)
    median_ratio = sorted(ratios)[n // 2]

    print(f"\n{'='*80}")
    print(f"Summary: {n} samples, S{args.max_stages} gamma={args.gamma}")
    print(f"  Avg ratio DDPM/GP:    {avg_ratio:.4f}x")
    print(f"  Median ratio:         {median_ratio:.4f}x")
    print(f"  Wins vs GP:           {wins}/{n} ({100*wins/n:.1f}%)")
    print(f"  Total time:           {elapsed_total:.0f}s ({elapsed_total/n:.1f}s/sample)")
    print(f"{'='*80}")
    print(f"Saved {PT_PATH}")


def _save_pt(samples, val_indices):
    torch.save({
        "samples": samples,
        "n_samples": len(samples),
        "val_indices": val_indices,
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
    }, PT_PATH)


# ═══════════════════════════════════════════════════════════════════
#  GENERATE PLOTS
# ═══════════════════════════════════════════════════════════════════

def generate_plots():
    print(f"\nLoading {PT_PATH} ...")
    data = torch.load(PT_PATH, map_location="cpu", weights_only=False)
    samples = data["samples"]
    n = len(samples)
    print(f"Generating standard plots for {n} samples → {PLOT_DIR}/")

    for i, s in enumerate(samples):
        # missing_mask convention: saved as 0=missing, 1=known
        # plot_inpaint_panels expects 1=missing, so invert
        mm = s["missing_mask"]
        mm_inv = 1.0 - mm  # 1=missing for the plotting function

        gp_mse = s["gp_mse"]
        ddpm_mse = s["ddpm_mse"]
        ratio = s["ratio"]

        methods = {
            "GP": s["gp_output"],
            "DDPM S6": s["ddpm_output"],
        }
        mse_dict = {
            "GP": gp_mse,
            "DDPM S6": ddpm_mse,
        }
        extra_titles = {
            "DDPM S6": f"({ratio:.3f}x GP)",
        }

        plot_inpaint_panels(
            gt=s["ground_truth"],
            missing_mask=mm_inv,
            methods=methods,
            mse=mse_dict,
            out_dir=PLOT_DIR,
            prefix=f"sample_{i+1:03d}",
            extra_titles=extra_titles,
        )

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n} samples done")

    print(f"All {n} sample plots saved to {PLOT_DIR}/")


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if args.plot_only:
        generate_plots()
    else:
        run_inference()
        if not args.skip_plots:
            generate_plots()
