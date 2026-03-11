#!/usr/bin/env python3
"""Compare 6-stage GP-warm DDPM (multi-mask model) vs V-CNN vs GP baseline.

Uses the 8-channel GP-context multi-mask DDPM with the S6 adaptive GP-init
inference approach, adapted for eps-prediction + 8ch conditioning.

Usage:
    PYTHONPATH=. python tmp_compare_6s_vcnn_gp.py --n-samples 10
    PYTHONPATH=. python tmp_compare_6s_vcnn_gp.py --n-samples 10 --known-frac 0.005
"""
import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import torch
import numpy as np
from scipy.ndimage import distance_transform_edt

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import matplotlib
matplotlib.use("Agg")

from ddpm.neural_networks.ddpm import GaussianDDPM
from ddpm.neural_networks.unets.unet_xl_attn import MyUNet_Attn
from ddpm.helper_functions.standardize_data import ZScoreStandardizer
from ddpm.helper_functions.interpolation_tool import gp_fill
from data_prep.data_initializer import DDInitializer
from scripts.voronoi_cnn_model import VoronoiCNN, build_voronoi_input

# ── Args ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--n-samples", type=int, default=10)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--known-frac", type=float, default=0.005,
                    help="Known fraction of ocean pixels (default 0.5%%)")
parser.add_argument("--max-stages", type=int, default=6)
parser.add_argument("--t-start", type=int, default=75,
                    help="First stage t_start")
parser.add_argument("--t-refine", type=int, default=50,
                    help="Refinement stages t_start")
parser.add_argument("--noise-floor", type=float, default=0.2)
parser.add_argument("--noise-floor-refine", type=float, default=0.3)
parser.add_argument("--var-decay", type=float, default=0.1)
parser.add_argument("--gamma", type=float, default=3.0)
parser.add_argument("--resample-steps", type=int, default=5)
args = parser.parse_args()

# ── Config ────────────────────────────────────────────────────────────
N_STEPS = 250
MIN_BETA = 0.0001
MAX_BETA = 0.02

U_MEAN = -0.06929559429949586
U_STD = 0.1358005549716049
V_MEAN = -0.0323937796117541
V_STD = 0.08899177232117582

OCEAN_H, OCEAN_W = 44, 94
GRID_H, GRID_W = 64, 128
N_OCEAN = OCEAN_H * OCEAN_W  # 4136

GP_PARAMS = {
    "lengthscale": 14.1,
    "variance": 0.0103420345,
    "noise": 1e-8,
    "kernel_type": "rbf_legacy",
    "coord_system": "pixels",
}

# ── Paths ─────────────────────────────────────────────────────────────
DDPM_CKPT = "experiments/09_gp_context/gp_context_multimask/results/inpaint_gaussian_t250_best_checkpoint.pt"
VCNN_CKPT = "results/voronoi_cnn/voronoi_cnn_best.pt"
OUT_DIR = "results/compare_6s_vcnn_gp"
os.makedirs(OUT_DIR, exist_ok=True)

standardizer = ZScoreStandardizer(U_MEAN, U_STD, V_MEAN, V_STD)
norm_mean = np.array([U_MEAN, V_MEAN], dtype=np.float32)
norm_std = np.array([U_STD, V_STD], dtype=np.float32)

# Pre-computed ocean pixel indices
_ocean_rows = np.array([r for r in range(OCEAN_H) for c in range(OCEAN_W)])
_ocean_cols = np.array([c for r in range(OCEAN_H) for c in range(OCEAN_W)])


# ── Mask generation ───────────────────────────────────────────────────
def generate_random_mask(known_frac, rng):
    """Generate random sparse observation mask. 1=missing, 0=known."""
    n_known = max(1, int(round(known_frac * N_OCEAN)))
    chosen = rng.choice(N_OCEAN, size=n_known, replace=False)
    rows = _ocean_rows[chosen]
    cols = _ocean_cols[chosen]
    mask = np.ones((GRID_H, GRID_W), dtype=np.float32)
    mask[rows, cols] = 0.0
    return torch.from_numpy(mask).unsqueeze(0)  # (1, H, W)


def compute_distance_map(mask_1ch):
    """Normalized distance to nearest known pixel."""
    known = (1.0 - mask_1ch.squeeze().numpy())
    if known.max() == 0:
        return torch.zeros(1, GRID_H, GRID_W)
    dist_np = distance_transform_edt(1.0 - known)
    dm = dist_np.max()
    if dm > 0:
        dist_np = dist_np / dm
    return torch.from_numpy(dist_np.astype(np.float32)).unsqueeze(0)


def compute_gp_context(x0_raw, mask_1ch):
    """Compute GP posterior and build 5ch known_context + raw GP output."""
    mask_2ch = mask_1ch.unsqueeze(0).expand(-1, 2, -1, -1)
    gp_raw, gp_var_raw = gp_fill(
        x0_raw, mask_2ch,
        lengthscale=GP_PARAMS["lengthscale"],
        variance=GP_PARAMS["variance"],
        noise=GP_PARAMS["noise"],
        use_double=True,
        kernel_type=GP_PARAMS["kernel_type"],
        coord_system=GP_PARAMS["coord_system"],
        return_variance=True,
    )

    gp_std = standardizer(gp_raw.squeeze(0)).unsqueeze(0)

    gp_var_max = gp_var_raw.max(dim=1, keepdim=True).values
    var_max_val = gp_var_max.max()
    if var_max_val > 0:
        gp_var_max = gp_var_max / var_max_val
    gp_var_max = gp_var_max.float()

    dist = compute_distance_map(mask_1ch)
    ocean = torch.zeros(1, GRID_H, GRID_W)
    ocean[0, :OCEAN_H, :OCEAN_W] = 1.0

    known_context = torch.cat([
        gp_std,               # (1, 2, H, W)
        gp_var_max,           # (1, 1, H, W)
        dist.unsqueeze(0),    # (1, 1, H, W)
        ocean.unsqueeze(0),   # (1, 1, H, W)
    ], dim=1)

    return known_context, gp_raw, gp_var_raw


# ── 6-stage GP-warm reverse (adapted for 8ch eps-prediction) ─────────
def gp_warm_stage(ddpm, x0_std, mask_1ch_dev, known_context, device,
                  gp_std, gp_var_raw, t_start, noise_floor, gamma,
                  resample_steps, seed, mask_xt=True):
    """Single stage of GP-warm-start reverse process.

    Algorithm (adapted from repaint_gp_init_adaptive for 8ch eps model):
    1. Build composite: GT_known + prior_unknown in standardized space
    2. Compute variance-adaptive noise weight
    3. Forward-diffuse composite to t_start with weighted noise
    4. Reverse denoise t_start→0 with 8ch conditioning + RePaint + resampling
    """
    torch.manual_seed(seed)
    ddpm.eval()

    alpha_bars = ddpm.alpha_bars.to(device)
    alphas = ddpm.alphas.to(device)
    betas = ddpm.betas.to(device)

    known_mask_2ch = (1.0 - mask_1ch_dev).expand(-1, 2, -1, -1)
    missing_mask_2ch = mask_1ch_dev.expand(-1, 2, -1, -1)

    # Clamp t_start
    t_start = min(t_start, ddpm.n_steps - 1)

    # Build composite in standardized space: known=GT, unknown=prior(GP)
    composite = x0_std * known_mask_2ch + gp_std * missing_mask_2ch

    # --- Variance-adaptive noise weight ---
    gp_var = gp_var_raw.to(device)
    if gp_var.shape[1] == 1:
        gp_var = gp_var.expand(-1, 2, -1, -1)

    masked_var = gp_var * missing_mask_2ch
    var_max = masked_var.max()
    if (missing_mask_2ch > 0.5).any():
        var_min = masked_var[missing_mask_2ch > 0.5].min()
    else:
        var_min = torch.tensor(0.0)
    var_range = var_max - var_min

    if var_range < 1e-12:
        var_norm = torch.ones_like(missing_mask_2ch)
    else:
        var_norm = ((masked_var - var_min) / var_range).clamp(0, 1)

    noise_weight = noise_floor + (1.0 - noise_floor) * var_norm ** gamma
    noise_weight = noise_weight * missing_mask_2ch + known_mask_2ch

    # --- Forward-diffuse composite to t_start ---
    alpha_bar_t = alpha_bars[t_start]
    noise_init = torch.randn(1, 2, GRID_H, GRID_W, device=device)
    x_t = (alpha_bar_t.sqrt() * composite +
           noise_weight * (1 - alpha_bar_t).sqrt() * noise_init)

    # --- Reverse process with 8ch conditioning + RePaint + resampling ---
    with torch.no_grad():
        for t in range(t_start, -1, -1):
            n_resample = resample_steps if t > 0 else 1

            for r in range(n_resample):
                alpha_t = alphas[t]
                alpha_bar_t = alpha_bars[t]
                beta_t = betas[t]

                # RePaint: paste forward-noised known values
                if t > 0:
                    noise_known = torch.randn_like(x_t)
                    x_known_noised = (alpha_bar_t.sqrt() * x0_std +
                                      (1 - alpha_bar_t).sqrt() * noise_known)
                    x_t = x_known_noised * known_mask_2ch + x_t * missing_mask_2ch

                # Build 8ch input: [x_t(2), mask(1), gp_mean(2), gp_var(1), dist(1), ocean(1)]
                if mask_xt:
                    indep_noise = torch.randn_like(x_t)
                    x_t_input = x_t * mask_1ch_dev + indep_noise * (1.0 - mask_1ch_dev)
                else:
                    x_t_input = x_t

                x_cond = torch.cat([x_t_input, mask_1ch_dev, known_context], dim=1)
                t_tensor = torch.tensor([t], device=device).long().reshape(1, -1)
                pred_eps = ddpm.network(x_cond, t_tensor)

                # eps→mu
                coeff_eps = beta_t / torch.sqrt(1.0 - alpha_bar_t)
                coeff_xt = 1.0 / torch.sqrt(alpha_t)
                mu = coeff_xt * (x_t - coeff_eps * pred_eps)

                if t > 0:
                    sigma = torch.sqrt(beta_t)
                    z = torch.randn_like(x_t)
                    # Variance-adaptive: less noise where GP is confident
                    x_denoised = mu + noise_weight * sigma * z
                else:
                    x_denoised = mu

                x_t = x0_std * known_mask_2ch + x_denoised * missing_mask_2ch

                # Resample (re-noise for next resample iteration)
                if r < n_resample - 1 and t > 0:
                    noise_back = torch.randn_like(x_t)
                    x_t = (alpha_t.sqrt() * x_t +
                           noise_weight * (1 - alpha_t).sqrt() * noise_back)

    # Final paste known pixels
    x_t = x0_std * known_mask_2ch + x_t * missing_mask_2ch
    return x_t


def run_6stage(ddpm, x0_std, mask_1ch_dev, known_context, device,
               gp_std, gp_var_raw, seed):
    """Run full 6-stage GP-warm inference."""
    current_gp_std = gp_std.clone()
    current_var = gp_var_raw.clone()

    for stage in range(1, args.max_stages + 1):
        if stage == 1:
            t_s = args.t_start
            nf = args.noise_floor
            s_seed = seed
        else:
            t_s = args.t_refine
            nf = args.noise_floor_refine
            s_seed = seed + stage * 10000
            current_var = current_var * args.var_decay

        stage_out = gp_warm_stage(
            ddpm, x0_std, mask_1ch_dev, known_context, device,
            gp_std=current_gp_std,
            gp_var_raw=current_var,
            t_start=t_s,
            noise_floor=nf,
            gamma=args.gamma,
            resample_steps=args.resample_steps,
            seed=s_seed,
            mask_xt=True,
        )
        # Feed output as next stage's prior
        current_gp_std = stage_out.clone()

    return stage_out


# ── V-CNN inference ───────────────────────────────────────────────────
def load_vcnn(device):
    """Load V-CNN model from checkpoint."""
    ckpt = torch.load(VCNN_CKPT, map_location="cpu", weights_only=False)
    cfg = ckpt.get("model_config", {})
    model = VoronoiCNN(
        in_channels=cfg.get("in_channels", 5),
        out_channels=cfg.get("out_channels", 2),
        base_ch=cfg.get("base_ch", 32),
        depth=cfg.get("depth", 3),
    )
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()
    return model, ckpt


def run_vcnn(vcnn_model, vel_raw_44x94, obs_mask_44x94, ocean_mask_np, device):
    """Run V-CNN on a sample with the given observation mask.

    Args:
        vel_raw_44x94: (2, 44, 94) numpy, physical velocity
        obs_mask_44x94: (44, 94) numpy, 1=known, 0=missing
        ocean_mask_np: (44, 94) numpy, 1=ocean, 0=land
    Returns:
        pred_full: (1, 2, 64, 128) tensor in physical space
    """
    vel_n = (vel_raw_44x94 - norm_mean[:, None, None]) / norm_std[:, None, None]
    vel_n *= ocean_mask_np[None, :, :]

    voronoi_in = build_voronoi_input(vel_n, obs_mask_44x94, ocean_mask_np)
    voronoi_t = torch.from_numpy(voronoi_in).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_n = vcnn_model(voronoi_t)  # (1, 2, 44, 94)

    mean_t = torch.tensor(norm_mean, dtype=torch.float32).view(1, 2, 1, 1).to(device)
    std_t = torch.tensor(norm_std, dtype=torch.float32).view(1, 2, 1, 1).to(device)
    pred_phys = pred_n * std_t + mean_t

    ocean_t = torch.from_numpy(ocean_mask_np).to(device).unsqueeze(0).unsqueeze(0)
    pred_phys = pred_phys * ocean_t  # (1, 2, 44, 94)

    # Embed in 64x128 grid
    pred_full = torch.zeros(1, 2, GRID_H, GRID_W, device=device)
    pred_full[:, :, :OCEAN_H, :OCEAN_W] = pred_phys
    return pred_full


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    print(f"{'='*80}")
    print("6-Stage GP-Warm DDPM (multi-mask) vs V-CNN vs GP Comparison")
    print(f"{'='*80}")

    # ── Load data ────────────────────────────────────────────────────
    dd = DDInitializer()
    device = dd.get_device()
    dd_std = dd.get_standardizer()

    test_data = dd.get_test_data()
    n_test = len(test_data)
    n_samples = min(args.n_samples, n_test)

    # Load raw data for V-CNN (needs (2, 44, 94) physical arrays)
    with open("data.pickle", "rb") as f:
        _train_np, val_np, _test_np = pickle.load(f)

    def _to_tensor(arr):
        t = torch.from_numpy(np.ascontiguousarray(arr)).float()
        t = t.permute(3, 2, 1, 0)
        t = torch.nan_to_num(t, nan=0.0)
        return t

    val_raw = _to_tensor(val_np)

    # Build ocean mask from data
    gt0 = val_raw[0]  # (2, 44, 94)
    speed0 = (gt0[0]**2 + gt0[1]**2).sqrt()
    ocean_mask_np = (speed0 > 1e-8).numpy().astype(np.float32)

    print(f"  Device: {device}")
    print(f"  Test samples: {n_test}, evaluating: {n_samples}")
    print(f"  Known fraction: {args.known_frac}")
    n_known_px = max(1, int(round(args.known_frac * N_OCEAN)))
    print(f"  → {n_known_px} known pixels out of {N_OCEAN} ({100*args.known_frac:.2f}%)")

    # ── Load DDPM (multi-mask, 8ch) ──────────────────────────────────
    print(f"\nLoading multi-mask DDPM from {DDPM_CKPT}...")
    network = MyUNet_Attn(n_steps=N_STEPS, time_emb_dim=256, in_channels=8)
    ddpm = GaussianDDPM(
        network, n_steps=N_STEPS,
        min_beta=MIN_BETA, max_beta=MAX_BETA, device=device,
    )
    ckpt = torch.load(DDPM_CKPT, map_location=device, weights_only=False)
    ddpm.load_state_dict(ckpt["model_state_dict"])
    ddpm = ddpm.to(device)
    ddpm.eval()
    epoch = ckpt.get("epoch", "?")
    print(f"  Loaded (epoch {epoch}, model weights)")

    # ── Load V-CNN ───────────────────────────────────────────────────
    print(f"Loading V-CNN from {VCNN_CKPT}...")
    vcnn, vcnn_ckpt = load_vcnn(device)
    n_params_vcnn = sum(p.numel() for p in vcnn.parameters()) / 1e6
    print(f"  V-CNN: {n_params_vcnn:.1f}M params")

    # ── S6 config ────────────────────────────────────────────────────
    print(f"\nS6 config: max_stages={args.max_stages}, "
          f"t_start={args.t_start}, t_refine={args.t_refine}")
    print(f"  noise_floor={args.noise_floor}, refine_floor={args.noise_floor_refine}, "
          f"var_decay={args.var_decay}, gamma={args.gamma}")
    print(f"  resample_steps={args.resample_steps}")

    # ── Inference ────────────────────────────────────────────────────
    rng = np.random.default_rng(args.seed)
    completed = []

    gp_mses, ddpm_mses, vcnn_mses = [], [], []

    print(f"\n{'#':>4}  {'Known':>5}  {'GP MSE':>12}  {'S6 MSE':>12}  "
          f"{'VCNN MSE':>12}  {'S6/GP':>7}  {'VCNN/GP':>8}  {'Time':>7}")
    print(f"{'-'*85}")
    t0_global = time.time()

    for i in range(n_samples):
        t0 = time.time()
        sample_seed = args.seed + i

        # -- Ground truth --
        # Use test_data (same as eval_gp_context_multimask.py)
        x0_dd_std = test_data[i][0]  # (2, H, W)
        x0_raw = dd_std.unstandardize(x0_dd_std).unsqueeze(0)  # (1, 2, H, W)
        x0_std = standardizer(x0_raw.squeeze(0)).unsqueeze(0).to(device)
        x0_raw_dev = x0_raw.to(device)

        # -- Generate random mask --
        mask_1ch = generate_random_mask(args.known_frac, rng=rng)  # (1, H, W)
        mask_1ch_dev = mask_1ch.unsqueeze(0).to(device)  # (1, 1, H, W)
        mask_2ch = mask_1ch_dev.expand(-1, 2, -1, -1)
        n_known = int((mask_1ch == 0).sum().item())

        # -- GP context --
        known_context, gp_raw, gp_var_raw = compute_gp_context(x0_raw, mask_1ch)
        known_context = known_context.to(device)
        gp_raw = gp_raw.to(device)
        gp_var_raw = gp_var_raw.to(device)

        # GP baseline MSE
        gp_mse = ((gp_raw - x0_raw_dev) * mask_2ch).pow(2).sum() / (
            mask_2ch.sum() + 1e-8)

        # -- GP standardized for S6 prior --
        gp_std = standardizer(gp_raw.squeeze(0).cpu()).unsqueeze(0).to(device)

        # -- Method 1: 6-stage GP-warm DDPM --
        s6_out_std = run_6stage(
            ddpm, x0_std, mask_1ch_dev, known_context, device,
            gp_std=gp_std, gp_var_raw=gp_var_raw, seed=sample_seed,
        )
        s6_raw = standardizer.unstandardize(
            s6_out_std.squeeze(0).cpu()
        ).unsqueeze(0).to(device)
        s6_mse = ((s6_raw - x0_raw_dev) * mask_2ch).pow(2).sum() / (
            mask_2ch.sum() + 1e-8)

        # -- Method 2: V-CNN --
        # V-CNN needs (2, 44, 94) physical + obs_mask (44x94, 1=known, 0=missing)
        gt_44x94 = x0_raw_dev[0, :, :OCEAN_H, :OCEAN_W].cpu().numpy()  # (2, 44, 94)
        # Convert mask: our mask is 1=missing, vcnn needs 1=known
        obs_mask_44x94 = (1.0 - mask_1ch[0, :OCEAN_H, :OCEAN_W].numpy())
        obs_mask_44x94 *= ocean_mask_np  # zero out land

        vcnn_pred = run_vcnn(vcnn, gt_44x94, obs_mask_44x94, ocean_mask_np, device)
        vcnn_mse = ((vcnn_pred - x0_raw_dev) * mask_2ch).pow(2).sum() / (
            mask_2ch.sum() + 1e-8)

        elapsed = time.time() - t0
        s6_ratio = s6_mse.item() / (gp_mse.item() + 1e-12)
        vcnn_ratio = vcnn_mse.item() / (gp_mse.item() + 1e-12)

        print(f"{i+1:>4}  {n_known:>5}  {gp_mse.item():>12.6f}  "
              f"{s6_mse.item():>12.6f}  {vcnn_mse.item():>12.6f}  "
              f"{s6_ratio:>6.3f}x  {vcnn_ratio:>7.3f}x  {elapsed:>6.1f}s")

        completed.append({
            "idx": i,
            "n_known": n_known,
            "known_frac": args.known_frac,
            "ground_truth": x0_raw_dev.cpu(),
            "gp_output": gp_raw.cpu(),
            "s6_output": s6_raw.cpu(),
            "vcnn_output": vcnn_pred.cpu(),
            "missing_mask": mask_2ch.cpu(),
            "observation_mask": mask_1ch.cpu(),
            "gp_mse": gp_mse.item(),
            "s6_mse": s6_mse.item(),
            "vcnn_mse": vcnn_mse.item(),
            "s6_ratio": s6_ratio,
            "vcnn_ratio": vcnn_ratio,
        })
        gp_mses.append(gp_mse.item())
        ddpm_mses.append(s6_mse.item())
        vcnn_mses.append(vcnn_mse.item())

    total_time = time.time() - t0_global

    # ── Save ─────────────────────────────────────────────────────────
    pt_path = os.path.join(OUT_DIR,
        f"compare_6s_vcnn_gp_n{n_samples}_frac{args.known_frac}.pt")
    torch.save({
        "samples": completed,
        "n_samples": len(completed),
        "known_frac": args.known_frac,
        "s6_config": {
            "max_stages": args.max_stages,
            "t_start": args.t_start,
            "t_refine": args.t_refine,
            "noise_floor": args.noise_floor,
            "noise_floor_refine": args.noise_floor_refine,
            "var_decay": args.var_decay,
            "gamma": args.gamma,
            "resample_steps": args.resample_steps,
        },
        "ddpm_checkpoint": DDPM_CKPT,
        "vcnn_checkpoint": VCNN_CKPT,
    }, pt_path)
    print(f"\nResults saved to {pt_path}")

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"COMPARISON SUMMARY — {n_samples} samples, "
          f"known_frac={args.known_frac} ({n_known_px} pixels)")
    print(f"{'='*80}")
    print(f"  GP baseline  — Mean MSE: {np.mean(gp_mses):.6f}, "
          f"Median: {np.median(gp_mses):.6f}")
    print(f"  S6 GP-warm   — Mean MSE: {np.mean(ddpm_mses):.6f}, "
          f"Median: {np.median(ddpm_mses):.6f}")
    print(f"  V-CNN        — Mean MSE: {np.mean(vcnn_mses):.6f}, "
          f"Median: {np.median(vcnn_mses):.6f}")

    mean_s6_ratio = np.mean(ddpm_mses) / (np.mean(gp_mses) + 1e-12)
    mean_vcnn_ratio = np.mean(vcnn_mses) / (np.mean(gp_mses) + 1e-12)
    print(f"\n  S6/GP ratio:   {mean_s6_ratio:.3f}x")
    print(f"  VCNN/GP ratio: {mean_vcnn_ratio:.3f}x")

    # Per-sample wins
    s6_beats_gp = sum(1 for s, g in zip(ddpm_mses, gp_mses) if s < g)
    vcnn_beats_gp = sum(1 for v, g in zip(vcnn_mses, gp_mses) if v < g)
    s6_beats_vcnn = sum(1 for s, v in zip(ddpm_mses, vcnn_mses) if s < v)
    n = len(completed)
    print(f"\n  S6 beats GP:    {s6_beats_gp}/{n} ({100*s6_beats_gp/n:.0f}%)")
    print(f"  VCNN beats GP:  {vcnn_beats_gp}/{n} ({100*vcnn_beats_gp/n:.0f}%)")
    print(f"  S6 beats VCNN:  {s6_beats_vcnn}/{n} ({100*s6_beats_vcnn/n:.0f}%)")

    # Median per-sample ratios
    s6_ratios = [s["s6_ratio"] for s in completed]
    vcnn_ratios = [s["vcnn_ratio"] for s in completed]
    print(f"\n  Per-sample S6/GP ratios:   mean={np.mean(s6_ratios):.3f}, "
          f"median={np.median(s6_ratios):.3f}")
    print(f"  Per-sample VCNN/GP ratios: mean={np.mean(vcnn_ratios):.3f}, "
          f"median={np.median(vcnn_ratios):.3f}")

    print(f"\n  Total time: {total_time:.0f}s ({total_time/n:.1f}s/sample)")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
