#!/usr/bin/env python
"""Run a single straight-line path example with Adaptive GP-Refined RePaint.

Produces:
  1. A .pt file with all tensors + metrics
  2. A multi-panel PNG visualization

Usage:
    PYTHONPATH=. python scripts/run_straight_line_adaptive.py
    PYTHONPATH=. python scripts/run_straight_line_adaptive.py --sample-idx 42 --line-width 3
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
parser.add_argument("--sample-idx", type=int, default=100,
                    help="Index into the validation set")
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
os.makedirs(OUT_DIR, exist_ok=True)

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

# ── get sample ───────────────────────────────────────────────────────
val_data = dd.get_validation_data()
print(f"Validation set: {len(val_data)} samples, using index {args.sample_idx}")

random.seed(args.seed)
torch.manual_seed(args.seed)

input_image = val_data[args.sample_idx][0].unsqueeze(0).to(device)
input_orig = standardizer.unstandardize(
    input_image.squeeze(0)
).to(device).unsqueeze(0)

# ── straight-line mask ───────────────────────────────────────────────
land_mask = (input_orig.abs() > 1e-5).float().to(device)
raw_mask = StraightLineMaskGenerator(
    num_lines=args.num_lines, line_thickness=1
).generate_mask(input_image.shape).to(device)
missing_mask = raw_mask * land_mask
mask_pct = missing_mask[:, 0:1].sum() / (land_mask[:, 0:1].sum() + 1e-8) * 100
print(f"Mask coverage: {mask_pct:.1f}% (transect, {args.num_lines} lines, thickness=1)")

# ── GP with posterior variance ───────────────────────────────────────
print("Running GP (with posterior variance)…")
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
print(f"  GP MSE: {gp_mse.item():.6f}")

# ── Adaptive GP-Refined ─────────────────────────────────────────────
gp_std = standardizer(gp_out.squeeze(0)).to(device).unsqueeze(0)

print(f"Running Adaptive GP-Refined (t={args.t_start}, r={args.resample_steps}, "
      f"nf={args.noise_floor})…")
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
diff_adaptive = (adaptive_phys - input_orig) * missing_mask
adaptive_mse = (diff_adaptive ** 2).sum() / (missing_mask.sum() + 1e-8)

ratio = adaptive_mse.item() / gp_mse.item()
print(f"  Adaptive MSE: {adaptive_mse.item():.6f}  ({ratio:.3f}x GP)")
winner = "Adaptive" if ratio < 1 else "GP"
print(f"  Winner: {winner}")

# ── Save .pt ─────────────────────────────────────────────────────────
pt_path = os.path.join(OUT_DIR, "straight_line_adaptive_example.pt")
torch.save({
    "gt":             input_orig.cpu(),
    "missing_mask":   missing_mask.cpu(),
    "gp_out":         gp_out.cpu(),
    "gp_var_map":     gp_var_map.cpu(),
    "adaptive_out":   adaptive_phys.cpu(),
    "gp_mse":         gp_mse.item(),
    "ada_mse":        adaptive_mse.item(),
    "mask_pct":       mask_pct.item(),
    "t_start":        args.t_start,
    "resample_steps": args.resample_steps,
    "noise_floor":    args.noise_floor,
    "line_thickness": 1,
    "num_lines":      args.num_lines,
    "val_index":      args.sample_idx,
    "epoch":          epoch,
}, pt_path)
print(f"Saved {pt_path}")

# ══════════════════════════════════════════════════════════════════════
# Visualization via standard_plots
# ══════════════════════════════════════════════════════════════════════
print("Generating visualizations…")
plot_inpaint_panels(
    gt=input_orig,
    missing_mask=missing_mask,
    methods={"GP": gp_out, "Adaptive": adaptive_phys},
    mse={"GP": gp_mse.item(), "Adaptive": adaptive_mse.item()},
    out_dir=OUT_DIR,
    prefix="straight_line",
    mask_label=f"{args.num_lines} transect(s)",
    extra_titles={"Adaptive": f"({ratio:.3f}x GP)"},
)
print("Done.")
