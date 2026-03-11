#!/usr/bin/env python
"""Evaluate the multi-mask GP-context conditioned DDPM (8-channel UNet, eps-prediction).

This model was trained with on-the-fly random masks at various known fractions.
At inference we generate random observation masks and compute GP conditioning
on the fly — exactly matching the training procedure.

Input: [x_t(2), mask(1), gp_mean(2), gp_var(1), distance(1), ocean_mask(1)] = 8ch
Target: predict eps (noise)

Inference methods:
  - Vanilla:   Full 250-step reverse from pure noise
  - RePaint:   Known-pixel pasting during reverse
  - Resample:  RePaint + resample jumps (jump=10, count=3)
  - GP-warm S6: 6-stage GP-warm-start with variance-adaptive noise + resample

Usage:
    PYTHONPATH=. python scripts/eval_gp_context_multimask.py --n-samples 10
    PYTHONPATH=. python scripts/eval_gp_context_multimask.py --known-frac 0.005  # fixed density
    PYTHONPATH=. python scripts/eval_gp_context_multimask.py  # default: 1 sample, random density
"""
import argparse
import os
import sys
import time
from pathlib import Path

import torch
import numpy as np
from scipy.ndimage import distance_transform_edt

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import matplotlib
matplotlib.use("Agg")

from ddpm.neural_networks.ddpm import GaussianDDPM
from ddpm.neural_networks.unets.unet_xl_attn import MyUNet_Attn
from ddpm.helper_functions.standardize_data import ZScoreStandardizer
from ddpm.helper_functions.interpolation_tool import gp_fill
from data_prep.data_initializer import DDInitializer

# ── Args ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--n-samples", type=int, default=1,
                    help="Number of test samples to evaluate")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--known-frac", type=float, default=None,
                    help="Fixed known fraction. If None, sample randomly from training set.")
parser.add_argument("--weights", type=str, default=None,
                    help="Path to checkpoint (default: best checkpoint)")
parser.add_argument("--use-ema", action="store_true", default=True,
                    help="Use EMA weights from checkpoint (default)")
parser.add_argument("--no-ema", dest="use_ema", action="store_false")
parser.add_argument("--mask-xt", action="store_true", default=True)
parser.add_argument("--no-mask-xt", dest="mask_xt", action="store_false")
args = parser.parse_args()

# ── Config (matching training) ────────────────────────────────────────
N_STEPS = 250
MIN_BETA = 0.0001
MAX_BETA = 0.02

# Per-component z-score standardizer (gaussian noise → zscore)
U_MEAN = -0.06929559429949586
U_STD = 0.1358005549716049
V_MEAN = -0.0323937796117541
V_STD = 0.08899177232117582

OCEAN_H, OCEAN_W = 44, 94
GRID_H, GRID_W = 64, 128
N_OCEAN = OCEAN_H * OCEAN_W  # 4136

# Known fractions used during training
TRAINING_KNOWN_FRACS = [0.001, 0.002, 0.005, 0.01]

# GP parameters (from data.yaml)
GP_PARAMS = {
    "lengthscale": 14.1,
    "variance": 0.0103420345,
    "noise": 1e-8,
    "kernel_type": "rbf_legacy",
    "coord_system": "pixels",
}

# ── Paths ─────────────────────────────────────────────────────────────
EXP_DIR = "experiments/09_gp_context/gp_context_multimask/results"
if args.weights:
    CKPT_PATH = args.weights
else:
    CKPT_PATH = os.path.join(EXP_DIR, "inpaint_gaussian_t250_best_checkpoint.pt")

OUT_DIR = "results/gp_context_multimask_eval"
os.makedirs(OUT_DIR, exist_ok=True)
PT_PATH = os.path.join(OUT_DIR, "gp_context_multimask_eval.pt")


# ── Helper: Generate random mask ──────────────────────────────────────
# Pre-compute ocean pixel indices
_ocean_rows = np.array([r for r in range(OCEAN_H) for c in range(OCEAN_W)])
_ocean_cols = np.array([c for r in range(OCEAN_H) for c in range(OCEAN_W)])


def generate_random_mask(known_frac, rng=None):
    """Generate a random sparse observation mask.

    Args:
        known_frac: fraction of ocean pixels that are known (observed)
        rng: numpy RandomState for reproducibility

    Returns:
        mask_1ch: (1, H, W) float tensor. 1=missing, 0=known.
    """
    if rng is None:
        rng = np.random.default_rng()
    n_known = max(1, int(round(known_frac * N_OCEAN)))
    chosen = rng.choice(N_OCEAN, size=n_known, replace=False)
    rows = _ocean_rows[chosen]
    cols = _ocean_cols[chosen]

    mask = np.ones((GRID_H, GRID_W), dtype=np.float32)
    mask[rows, cols] = 0.0
    return torch.from_numpy(mask).unsqueeze(0)  # (1, H, W)


def compute_distance_map(mask_1ch):
    """Compute normalized distance to nearest known pixel."""
    known_binary = (1.0 - mask_1ch.squeeze().numpy())
    if known_binary.max() == 0:
        return torch.zeros(1, GRID_H, GRID_W)
    dist_np = distance_transform_edt(1.0 - known_binary)
    dist_max = dist_np.max()
    if dist_max > 0:
        dist_np = dist_np / dist_max
    return torch.from_numpy(dist_np.astype(np.float32)).unsqueeze(0)


def compute_gp_context(x0_raw, mask_1ch, standardizer):
    """Compute GP posterior and build the 5-channel known_context.

    Args:
        x0_raw: (1, 2, H, W) raw velocity field (with known values)
        mask_1ch: (1, H, W) mask (1=missing, 0=known)
        standardizer: ZScoreStandardizer

    Returns:
        known_context: (1, 5, H, W) [gp_mean_u, gp_mean_v, gp_var, dist, ocean]
        gp_raw: (1, 2, H, W) raw GP output (for MSE baseline)
    """
    mask_2ch = mask_1ch.unsqueeze(0).expand(-1, 2, -1, -1)  # (1, 2, H, W)

    gp_raw, gp_var_raw = gp_fill(
        x0_raw,
        mask_2ch,
        lengthscale=GP_PARAMS["lengthscale"],
        variance=GP_PARAMS["variance"],
        noise=GP_PARAMS["noise"],
        use_double=True,  # better numerical stability at inference
        kernel_type=GP_PARAMS["kernel_type"],
        coord_system=GP_PARAMS["coord_system"],
        return_variance=True,
    )
    # gp_raw: (1, 2, H, W), gp_var_raw: (1, 2, H, W)

    # Standardize GP mean
    gp_std = standardizer(gp_raw.squeeze(0)).unsqueeze(0)  # (1, 2, H, W)

    # GP variance: max over u,v → normalize
    gp_var_max = gp_var_raw.max(dim=1, keepdim=True).values  # (1, 1, H, W)
    var_max_val = gp_var_max.max()
    if var_max_val > 0:
        gp_var_max = gp_var_max / var_max_val
    gp_var_max = gp_var_max.float()

    # Distance map
    dist = compute_distance_map(mask_1ch)  # (1, H, W)

    # Ocean mask
    ocean = torch.zeros(1, GRID_H, GRID_W)
    ocean[0, :OCEAN_H, :OCEAN_W] = 1.0

    known_context = torch.cat([
        gp_std,                        # (1, 2, H, W)
        gp_var_max,                    # (1, 1, H, W)
        dist.unsqueeze(0),             # (1, 1, H, W)
        ocean.unsqueeze(0),            # (1, 1, H, W)
    ], dim=1)  # (1, 5, H, W)

    return known_context, gp_raw


# ── Reverse process (copied from eval_gp_context.py) ─────────────────
def gp_context_reverse(ddpm, x_T, mask_1ch, known_context, device,
                        mask_xt=True, seed=42, x0_known_std=None,
                        repaint=False, resample_jumps=10, resample_count=3):
    """Run DDPM reverse with 8ch GP-context conditioning + optional RePaint."""
    torch.manual_seed(seed)
    ddpm.eval()

    n_steps = ddpm.n_steps
    alpha_bars = ddpm.alpha_bars.to(device)
    alphas = ddpm.alphas.to(device)
    betas = ddpm.betas.to(device)

    known_mask_2ch = (1.0 - mask_1ch).expand(-1, 2, -1, -1)
    missing_mask_2ch = mask_1ch.expand(-1, 2, -1, -1)

    x_t = x_T.clone().to(device)

    if repaint and resample_jumps > 0:
        schedule = _build_repaint_schedule(n_steps, resample_jumps, resample_count)
    else:
        schedule = list(reversed(range(n_steps)))

    with torch.no_grad():
        prev_t = n_steps
        for t_idx in schedule:
            if repaint and t_idx >= prev_t and prev_t < n_steps:
                alpha_bar_target = alpha_bars[t_idx]
                alpha_bar_prev = alpha_bars[max(prev_t - 1, 0)]
                ratio = alpha_bar_target / (alpha_bar_prev + 1e-12)
                noise = torch.randn_like(x_t)
                x_t = torch.sqrt(ratio) * x_t + torch.sqrt(1.0 - ratio) * noise

            t_tensor = torch.tensor([t_idx], device=device).long()

            if repaint and x0_known_std is not None and t_idx > 0:
                alpha_bar_t = alpha_bars[t_idx]
                noise = torch.randn_like(x_t)
                x_known_noised = (torch.sqrt(alpha_bar_t) * x0_known_std +
                                  torch.sqrt(1.0 - alpha_bar_t) * noise)
                x_t = x_known_noised * known_mask_2ch + x_t * missing_mask_2ch

            if mask_xt:
                indep_noise = torch.randn_like(x_t)
                x_t_input = x_t * mask_1ch + indep_noise * (1.0 - mask_1ch)
            else:
                x_t_input = x_t

            x_cond = torch.cat([x_t_input, mask_1ch, known_context], dim=1)
            pred_eps = ddpm.network(x_cond, t_tensor.reshape(1, -1))

            alpha_t = alphas[t_idx]
            alpha_bar_t = alpha_bars[t_idx]
            beta_t = betas[t_idx]

            coeff_eps = beta_t / torch.sqrt(1.0 - alpha_bar_t)
            coeff_xt = 1.0 / torch.sqrt(alpha_t)
            mu = coeff_xt * (x_t - coeff_eps * pred_eps)

            if t_idx > 0:
                sigma = torch.sqrt(beta_t)
                z = torch.randn_like(x_t)
                x_t = mu + sigma * z
            else:
                x_t = mu

            prev_t = t_idx

    if repaint and x0_known_std is not None:
        x_t = x0_known_std * known_mask_2ch + x_t * missing_mask_2ch

    return x_t


def _build_repaint_schedule(n_steps, jump_size=10, n_resample=3):
    """Build RePaint time schedule with resample jumps."""
    schedule = []
    t = n_steps - 1
    while t >= 0:
        steps_this_seg = min(jump_size, t + 1)
        for resample_idx in range(n_resample):
            for s in range(steps_this_seg):
                schedule.append(t - s)
            if resample_idx < n_resample - 1 and t - steps_this_seg + 1 > 0:
                schedule.append(t)
        t -= steps_this_seg
    return schedule


# ── Main ──────────────────────────────────────────────────────────────
def run_inference():
    print(f"{'='*70}")
    print(f"Multi-Mask GP-Context DDPM Inference")
    print(f"{'='*70}")

    # ── Load data ─────────────────────────────────────────────────────
    dd = DDInitializer()
    device = dd.get_device()
    standardizer = ZScoreStandardizer(U_MEAN, U_STD, V_MEAN, V_STD)

    # Get raw test data for GP computation
    test_data = dd.get_test_data()
    dd_std = dd.get_standardizer()
    n_test = len(test_data)
    n_samples = min(args.n_samples, n_test)

    print(f"  Device: {device}")
    print(f"  Test samples: {n_test}, evaluating: {n_samples}")
    if args.known_frac is not None:
        print(f"  Fixed known fraction: {args.known_frac}")
        n_known_pixels = max(1, int(round(args.known_frac * N_OCEAN)))
        print(f"  → {n_known_pixels} known pixels out of {N_OCEAN} ocean pixels "
              f"({100*args.known_frac:.2f}%)")
    else:
        print(f"  Random known fractions from: {TRAINING_KNOWN_FRACS}")

    # ── Load model ────────────────────────────────────────────────────
    print(f"\nLoading checkpoint from {CKPT_PATH}...")
    network = MyUNet_Attn(n_steps=N_STEPS, time_emb_dim=256, in_channels=8)
    ddpm = GaussianDDPM(
        network, n_steps=N_STEPS,
        min_beta=MIN_BETA, max_beta=MAX_BETA, device=device,
    )

    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    if args.use_ema and "ema_state" in ckpt:
        print("  Using EMA shadow weights from checkpoint")
        shadow = ckpt["ema_state"]["shadow"]
        # Shadow keys use GaussianDDPM format (network.xxx)
        # Load into ddpm, allowing 1 missing buffer key
        missing, unexpected = ddpm.load_state_dict(shadow, strict=False)
        if missing:
            print(f"  EMA missing keys ({len(missing)}): {missing[:5]}")
        if unexpected:
            print(f"  EMA unexpected keys ({len(unexpected)}): {unexpected[:5]}")
    elif args.use_ema and "ema_state_dict" in ckpt:
        print("  Using EMA weights from checkpoint")
        ddpm.load_state_dict(ckpt["ema_state_dict"])
    elif "model_state_dict" in ckpt:
        print("  Using model weights from checkpoint")
        ddpm.load_state_dict(ckpt["model_state_dict"])
    else:
        # Try loading as raw state dict
        ddpm.load_state_dict(ckpt)

    ddpm = ddpm.to(device)
    ddpm.eval()

    n_params = sum(p.numel() for p in ddpm.parameters()) / 1e6
    print(f"  Model: MyUNet_Attn(in_channels=8), {n_params:.1f}M params")
    print(f"  Prediction: eps, T={N_STEPS}, mask_xt={args.mask_xt}")

    # ── Inference loop ────────────────────────────────────────────────
    completed = []
    gp_mses = []
    ddpm_mses = []
    repaint_mses = []
    resample_mses = []
    known_fracs_used = []

    print(f"\n{'#':>4}  {'Frac':>6}  {'Known':>5}  {'GP MSE':>12}  {'Vanilla MSE':>12}  "
          f"{'RePaint MSE':>12}  {'Resample MSE':>12}  {'Time':>7}")
    print(f"{'-'*90}")
    t0_global = time.time()

    rng = np.random.default_rng(args.seed)

    for i in range(n_samples):
        t0 = time.time()
        sample_seed = args.seed + i

        # Get ground truth
        x0_dd_std = test_data[i][0]  # (2, H, W)
        x0_raw = dd_std.unstandardize(x0_dd_std).unsqueeze(0)  # (1, 2, H, W)
        x0_std = standardizer(x0_raw.squeeze(0)).unsqueeze(0).to(device)

        # Choose known fraction
        if args.known_frac is not None:
            kf = args.known_frac
        else:
            kf = rng.choice(TRAINING_KNOWN_FRACS)
        known_fracs_used.append(kf)

        # Generate random mask
        mask_1ch = generate_random_mask(kf, rng=rng)  # (1, H, W)
        mask_1ch_dev = mask_1ch.unsqueeze(0).to(device)  # (1, 1, H, W)
        mask_2ch = mask_1ch_dev.expand(-1, 2, -1, -1)

        n_known = int((mask_1ch == 0).sum().item())

        # Compute GP context on the fly
        known_context, gp_raw = compute_gp_context(x0_raw, mask_1ch, standardizer)
        known_context = known_context.to(device)
        gp_raw = gp_raw.to(device)
        x0_raw_dev = x0_raw.to(device)

        # GP baseline MSE (missing region only)
        gp_mse = ((gp_raw - x0_raw_dev) * mask_2ch).pow(2).sum() / (
            mask_2ch.sum() + 1e-8)

        # Shared noise
        torch.manual_seed(sample_seed)
        H, W = GRID_H, GRID_W
        x_T = torch.randn(1, 2, H, W, device=device)

        # Method 1: Vanilla DDPM
        x0_vanilla_std = gp_context_reverse(
            ddpm, x_T, mask_1ch_dev, known_context,
            device=device, mask_xt=args.mask_xt, seed=sample_seed,
            repaint=False,
        )
        x0_vanilla_raw = standardizer.unstandardize(
            x0_vanilla_std.squeeze(0).cpu()
        ).unsqueeze(0).to(device)
        vanilla_mse = ((x0_vanilla_raw - x0_raw_dev) * mask_2ch).pow(2).sum() / (
            mask_2ch.sum() + 1e-8)

        # Method 2: RePaint (no resample)
        x0_repaint_std = gp_context_reverse(
            ddpm, x_T.clone(), mask_1ch_dev, known_context,
            device=device, mask_xt=args.mask_xt, seed=sample_seed,
            x0_known_std=x0_std, repaint=True,
            resample_jumps=0, resample_count=1,
        )
        x0_repaint_raw = standardizer.unstandardize(
            x0_repaint_std.squeeze(0).cpu()
        ).unsqueeze(0).to(device)
        repaint_mse = ((x0_repaint_raw - x0_raw_dev) * mask_2ch).pow(2).sum() / (
            mask_2ch.sum() + 1e-8)

        # Method 3: RePaint + resample jumps
        x0_resample_std = gp_context_reverse(
            ddpm, x_T.clone(), mask_1ch_dev, known_context,
            device=device, mask_xt=args.mask_xt, seed=sample_seed,
            x0_known_std=x0_std, repaint=True,
            resample_jumps=10, resample_count=3,
        )
        x0_resample_raw = standardizer.unstandardize(
            x0_resample_std.squeeze(0).cpu()
        ).unsqueeze(0).to(device)
        resample_mse = ((x0_resample_raw - x0_raw_dev) * mask_2ch).pow(2).sum() / (
            mask_2ch.sum() + 1e-8)

        elapsed = time.time() - t0

        print(f"{i+1:>4}  {kf:>6.3f}  {n_known:>5}  {gp_mse.item():>12.6f}  "
              f"{vanilla_mse.item():>12.6f}  {repaint_mse.item():>12.6f}  "
              f"{resample_mse.item():>12.6f}  {elapsed:>6.1f}s")

        completed.append({
            "idx": i,
            "known_frac": kf,
            "n_known": n_known,
            "ground_truth": x0_raw_dev.cpu(),
            "gp_output": gp_raw.cpu(),
            "ddpm_vanilla": x0_vanilla_raw.cpu(),
            "ddpm_repaint": x0_repaint_raw.cpu(),
            "ddpm_resample": x0_resample_raw.cpu(),
            "missing_mask": mask_2ch.cpu(),
            "observation_mask": mask_1ch.cpu(),
            "gp_mse": gp_mse.item(),
            "vanilla_mse": vanilla_mse.item(),
            "repaint_mse": repaint_mse.item(),
            "resample_mse": resample_mse.item(),
        })
        gp_mses.append(gp_mse.item())
        ddpm_mses.append(vanilla_mse.item())
        repaint_mses.append(repaint_mse.item())
        resample_mses.append(resample_mse.item())

    total_time = time.time() - t0_global

    # ── Save results ──────────────────────────────────────────────────
    torch.save({
        "samples": completed,
        "n_samples": len(completed),
        "model": "Multi-mask GP-context MyUNet_Attn(in_channels=8), eps-prediction",
        "checkpoint": CKPT_PATH,
        "mask_xt": args.mask_xt,
        "use_ema": args.use_ema,
        "seed": args.seed,
        "known_fracs_used": known_fracs_used,
        "gp_params": GP_PARAMS,
        "methods": ["gp", "vanilla", "repaint", "resample"],
    }, PT_PATH)
    print(f"\nResults saved to {PT_PATH}")

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"Multi-Mask GP-Context DDPM — {len(completed)} samples, {total_time:.0f}s")
    print(f"{'='*70}")
    print(f"  GP baseline  — Mean MSE: {np.mean(gp_mses):.6f}, "
          f"Median: {np.median(gp_mses):.6f}")
    print(f"  Vanilla DDPM — Mean MSE: {np.mean(ddpm_mses):.6f}, "
          f"Median: {np.median(ddpm_mses):.6f}")
    print(f"  RePaint      — Mean MSE: {np.mean(repaint_mses):.6f}, "
          f"Median: {np.median(repaint_mses):.6f}")
    print(f"  Resample     — Mean MSE: {np.mean(resample_mses):.6f}, "
          f"Median: {np.median(resample_mses):.6f}")
    print(f"\n  Vanilla/GP ratio: {np.mean(ddpm_mses)/(np.mean(gp_mses)+1e-12):.3f}x")
    print(f"  RePaint/GP ratio: {np.mean(repaint_mses)/(np.mean(gp_mses)+1e-12):.3f}x")
    print(f"  Resample/GP ratio: {np.mean(resample_mses)/(np.mean(gp_mses)+1e-12):.3f}x")
    rp_wins = sum(1 for r, g in zip(repaint_mses, gp_mses) if r < g)
    rs_wins = sum(1 for r, g in zip(resample_mses, gp_mses) if r < g)
    van_wins = sum(1 for d, g in zip(ddpm_mses, gp_mses) if d < g)
    n = len(completed)
    print(f"\n  vs GP:  Vanilla {van_wins}/{n}, RePaint {rp_wins}/{n}, Resample {rs_wins}/{n}")

    # Per-density breakdown if using random fracs
    if args.known_frac is None and len(set(known_fracs_used)) > 1:
        print(f"\n  Per-density breakdown:")
        for kf in sorted(set(known_fracs_used)):
            idxs = [j for j, f in enumerate(known_fracs_used) if f == kf]
            gp_sub = [gp_mses[j] for j in idxs]
            rs_sub = [resample_mses[j] for j in idxs]
            ratio = np.mean(rs_sub) / (np.mean(gp_sub) + 1e-12)
            wins = sum(1 for r, g in zip(rs_sub, gp_sub) if r < g)
            print(f"    frac={kf:.3f} ({len(idxs)} samples): "
                  f"GP={np.mean(gp_sub):.6f}, Resample={np.mean(rs_sub):.6f}, "
                  f"ratio={ratio:.3f}x, wins={wins}/{len(idxs)}")

    print(f"{'='*70}")


if __name__ == "__main__":
    run_inference()
