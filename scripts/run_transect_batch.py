#!/usr/bin/env python
"""Batch evaluation: Adaptive GP-Refined vs GP on transect masks.

Runs N samples with StraightLineMaskGenerator (1 transect, thickness=1),
prints per-sample results and summary statistics.

Usage:
    PYTHONPATH=. python scripts/run_transect_batch.py
    PYTHONPATH=. python scripts/run_transect_batch.py --n-samples 20
"""

import argparse, os, sys, random
from pathlib import Path

import torch

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from data_prep.data_initializer import DDInitializer
from ddpm.neural_networks.ddpm import GaussianDDPM
from ddpm.neural_networks.unets.unet_xl_attn import MyUNet_Attn
from ddpm.helper_functions.masks.straigth_line import StraightLineMaskGenerator
from ddpm.utils.inpainting_utils import repaint_gp_init_adaptive
from ddpm.utils.noise_utils import get_noise_strategy
from ddpm.helper_functions.interpolation_tool import gp_fill
from plots.visualization_tools.standard_plots import plot_inpaint_panels

# ── args ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--n-samples", type=int, default=10)
parser.add_argument("--t-start", type=int, default=75)
parser.add_argument("--resample-steps", type=int, default=5)
parser.add_argument("--noise-floor", type=float, default=0.2)
parser.add_argument("--num-lines", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

# ── paths ────────────────────────────────────────────────────────────
CHECKPOINT = (
    "experiments/02_inpaint_algorithm/repaint_gaussian_attn/results/"
    "inpaint_gaussian_t250_best_checkpoint.pt"
)
OUT_DIR = "experiments/02_inpaint_algorithm/repaint_gaussian_attn/results"
IMG_DIR = os.path.join(OUT_DIR, "transect_batch")
os.makedirs(IMG_DIR, exist_ok=True)

# ── initialise ───────────────────────────────────────────────────────
dd = DDInitializer()
device = dd.get_device()
noise_strategy = get_noise_strategy("gaussian")
standardizer = dd.get_standardizer()

# ── load model ───────────────────────────────────────────────────────
ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
n_steps  = ckpt.get("n_steps", 250)
min_beta = ckpt.get("min_beta", 0.0001)
max_beta = ckpt.get("max_beta", 0.02)
epoch    = ckpt.get("epoch", "?")

network = MyUNet_Attn(n_steps=n_steps, time_emb_dim=256)
ddpm = GaussianDDPM(
    network, n_steps=n_steps,
    min_beta=min_beta, max_beta=max_beta,
    device=device,
)
ddpm.load_state_dict(ckpt["model_state_dict"])
ddpm = ddpm.to(device)
ddpm.eval()
print(f"Loaded checkpoint (epoch {epoch}, n_steps={n_steps})")

# ── validation data ─────────────────────────────────────────────────
val_data = dd.get_validation_data()
n_val = len(val_data)
print(f"Validation set: {n_val} samples")

random.seed(args.seed)
torch.manual_seed(args.seed)

# Deterministic sample indices spread across the dataset
indices = [int(i * n_val / args.n_samples) for i in range(args.n_samples)]

# ── run loop ─────────────────────────────────────────────────────────
results = []
print(f"\n{'Idx':>5} {'Mask%':>6} {'GP_MSE':>10} {'Ada_MSE':>10} {'Ratio':>7} {'Winner':>8}")
print("-" * 55)

for run_i, idx in enumerate(indices):
    input_image = val_data[idx][0].unsqueeze(0).to(device)
    input_orig = standardizer.unstandardize(
        input_image.squeeze(0)
    ).to(device).unsqueeze(0)

    # Mask
    land_mask = (input_orig.abs() > 1e-5).float().to(device)
    raw_mask = StraightLineMaskGenerator(
        num_lines=args.num_lines, line_thickness=1
    ).generate_mask(input_image.shape).to(device)
    missing_mask = raw_mask * land_mask
    mask_pct = (missing_mask[:, 0:1].sum() / (land_mask[:, 0:1].sum() + 1e-8) * 100).item()

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

    # Adaptive GP-Refined
    gp_std = standardizer(gp_out.squeeze(0)).to(device).unsqueeze(0)
    with torch.no_grad():
        adaptive_out = repaint_gp_init_adaptive(
            ddpm, input_image, missing_mask,
            gp_image=gp_std,
            gp_variance_map=gp_var_map,
            t_start=args.t_start,
            noise_floor=args.noise_floor,
            n_samples=1, device=device,
            noise_strategy=noise_strategy,
            prediction_target="eps",
            resample_steps=args.resample_steps,
            project_div_free=False,
        )
    adaptive_phys = standardizer.unstandardize(
        adaptive_out.squeeze(0)
    ).to(device).unsqueeze(0)
    diff_ada = (adaptive_phys - input_orig) * missing_mask
    ada_mse = (diff_ada ** 2).sum() / (missing_mask.sum() + 1e-8)

    ratio = ada_mse.item() / gp_mse.item()
    winner = "Adaptive" if ratio < 1 else "GP"

    results.append({
        "idx": idx, "mask_pct": mask_pct,
        "gp_mse": gp_mse.item(), "ada_mse": ada_mse.item(),
        "ratio": ratio, "winner": winner,
    })

    print(f"{idx:>5} {mask_pct:>5.1f}% {gp_mse.item():>10.6f} {ada_mse.item():>10.6f} {ratio:>7.3f} {winner:>8}")

    # ── save per-sample .pt ──────────────────────────────────────────
    tag = f"{run_i:02d}_idx{idx}"
    torch.save({
        "gt":           input_orig.cpu(),
        "missing_mask": missing_mask.cpu(),
        "gp_out":       gp_out.cpu(),
        "gp_var_map":   gp_var_map.cpu(),
        "adaptive_out": adaptive_phys.cpu(),
        "gp_mse":       gp_mse.item(),
        "ada_mse":      ada_mse.item(),
        "ratio":        ratio,
        "mask_pct":     mask_pct,
        "val_index":    idx,
    }, os.path.join(IMG_DIR, f"transect_{tag}.pt"))

    # ── per-sample plots ─────────────────────────────────────────────
    plot_inpaint_panels(
        gt=input_orig,
        missing_mask=missing_mask,
        methods={"GP": gp_out, "Adaptive": adaptive_phys},
        mse={"GP": gp_mse.item(), "Adaptive": ada_mse.item()},
        out_dir=IMG_DIR,
        prefix=f"transect_{tag}",
        extra_titles={"Adaptive": f"({ratio:.3f}x GP)"},
    )

# ── summary ──────────────────────────────────────────────────────────
n = len(results)
wins = sum(1 for r in results if r["winner"] == "Adaptive")
ratios = [r["ratio"] for r in results]
avg_ratio = sum(ratios) / n
med_ratio = sorted(ratios)[n // 2]
best = min(ratios)
worst = max(ratios)

print(f"\n{'='*55}")
print(f"Transect batch: {n} samples, {args.num_lines} line(s), thickness=1")
print(f"  t_start={args.t_start}, r={args.resample_steps}, noise_floor={args.noise_floor}")
print(f"  Adaptive wins: {wins}/{n} ({100*wins/n:.0f}%)")
print(f"  Avg ratio:     {avg_ratio:.3f}x GP")
print(f"  Median ratio:  {med_ratio:.3f}x GP")
print(f"  Best ratio:    {best:.3f}x GP")
print(f"  Worst ratio:   {worst:.3f}x GP")
print(f"{'='*55}")

# ── save results ─────────────────────────────────────────────────────
pt_path = os.path.join(OUT_DIR, "transect_batch_results.pt")
torch.save({"results": results, "args": vars(args)}, pt_path)
print(f"Saved {pt_path}")
