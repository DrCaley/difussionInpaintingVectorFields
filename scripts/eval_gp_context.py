#!/usr/bin/env python
"""Evaluate the GP-context conditioned DDPM (8-channel UNet, eps-prediction).

This model uses dense spatial conditioning:
  Input: [x_t(2), mask(1), gp_mean(2), gp_var(1), distance(1), ocean_mask(1)] = 8ch
  Target: predict eps (noise)

Inference: standard DDPM reverse process with eps-prediction, building the
8-channel conditioning at each reverse step.

Usage:
    PYTHONPATH=. python scripts/eval_gp_context.py --n-samples 5
    PYTHONPATH=. python scripts/eval_gp_context.py --n-samples 20
    PYTHONPATH=. python scripts/eval_gp_context.py  # default: 1 sample
"""
import argparse
import os
import sys
import time
from pathlib import Path

import torch
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import matplotlib
matplotlib.use("Agg")

from ddpm.neural_networks.ddpm import GaussianDDPM
from ddpm.neural_networks.unets.unet_xl_attn import MyUNet_Attn
from ddpm.helper_functions.standardize_data import ZScoreStandardizer
from data_prep.data_initializer import DDInitializer

# ── Args ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--n-samples", type=int, default=1,
                    help="Number of test samples to evaluate")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--weights", type=str, default=None,
                    help="Path to model weights (default: best EMA weights)")
parser.add_argument("--use-ema", action="store_true", default=True,
                    help="Use EMA weights (default)")
parser.add_argument("--no-ema", dest="use_ema", action="store_false",
                    help="Use raw (non-EMA) weights")
parser.add_argument("--mask-xt", action="store_true", default=True,
                    help="Mask known region of x_t with independent noise (matches training)")
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

# ── Paths ─────────────────────────────────────────────────────────────
EXP_DIR = "experiments/09_gp_context/gp_context_eps/results"
if args.weights:
    WEIGHTS_PATH = args.weights
elif args.use_ema:
    WEIGHTS_PATH = os.path.join(EXP_DIR, "inpaint_gaussian_t250_best_ema_weights.pt")
else:
    WEIGHTS_PATH = os.path.join(EXP_DIR, "inpaint_gaussian_t250_best_weights.pt")

GP_CACHE_PATH = "data/rams_head/gp_precomputed.pt"
OUT_DIR = "results/gp_context_eval"
os.makedirs(OUT_DIR, exist_ok=True)
PT_PATH = os.path.join(OUT_DIR, "gp_context_eval.pt")


def gp_context_reverse(ddpm, x_T, mask_1ch, known_context, device,
                        mask_xt=True, seed=42, x0_known_std=None,
                        repaint=False, resample_jumps=10, resample_count=3):
    """Run DDPM reverse with 8ch GP-context conditioning + optional RePaint.

    This replicates the training forward pass at each reverse step:
        x_cond = [x_t(2), mask(1), known_context(5)]  → 8ch → UNet → pred_eps

    When repaint=True, applies RePaint-style data consistency:
      1) At each reverse step, splice q(x_t|x_0) noised ground truth into
         known pixels (known-region replacement).
      2) Resample jumps: after denoising to t, re-noise back to t+jump,
         then re-denoise (improves coherence at boundary).

    Args:
        ddpm: GaussianDDPM with 8-channel UNet
        x_T: (1, 2, H, W) initial noise
        mask_1ch: (1, 1, H, W) observation mask (1=missing, 0=known)
        known_context: (1, 5, H, W) [gp_mean_u, gp_mean_v, gp_var, dist, ocean]
        device: torch device
        mask_xt: if True, replace known region of x_t with independent noise
        seed: random seed for reproducibility
        x0_known_std: (1, 2, H, W) standardized ground truth for RePaint
        repaint: if True, enable RePaint known-region replacement + resample
        resample_jumps: jump size for resample schedule (default 10)
        resample_count: number of resample iterations per jump point (default 3)

    Returns:
        x_0: (1, 2, H, W) denoised output
    """
    torch.manual_seed(seed)
    ddpm.eval()

    n_steps = ddpm.n_steps
    alpha_bars = ddpm.alpha_bars.to(device)
    alphas = ddpm.alphas.to(device)
    betas = ddpm.betas.to(device)

    known_mask_2ch = (1.0 - mask_1ch).expand(-1, 2, -1, -1)  # 1 where known
    missing_mask_2ch = mask_1ch.expand(-1, 2, -1, -1)         # 1 where missing

    x_t = x_T.clone().to(device)

    # Build RePaint time schedule with resample jumps
    # Standard: [T-1, T-2, ..., 0], with jumps we revisit higher t's
    if repaint and resample_jumps > 0:
        schedule = _build_repaint_schedule(n_steps, resample_jumps, resample_count)
    else:
        schedule = list(reversed(range(n_steps)))

    with torch.no_grad():
        prev_t = n_steps  # track for detecting forward jumps
        for t_idx in schedule:
            # If t_idx increased (resample jump), re-noise x_t forward
            if repaint and t_idx >= prev_t and prev_t < n_steps:
                # Forward diffuse from prev_t to t_idx
                # q(x_{t_idx} | x_{prev_t}) using the ratio of alpha_bars
                alpha_bar_target = alpha_bars[t_idx]
                alpha_bar_prev = alpha_bars[max(prev_t - 1, 0)]
                # noise fraction needed
                ratio = alpha_bar_target / (alpha_bar_prev + 1e-12)
                noise = torch.randn_like(x_t)
                x_t = torch.sqrt(ratio) * x_t + torch.sqrt(1.0 - ratio) * noise

            t_tensor = torch.tensor([t_idx], device=device).long()

            # RePaint: known-region replacement — splice noised GT into known pixels
            if repaint and x0_known_std is not None and t_idx > 0:
                alpha_bar_t = alpha_bars[t_idx]
                noise = torch.randn_like(x_t)
                # q(x_t | x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
                x_known_noised = (torch.sqrt(alpha_bar_t) * x0_known_std +
                                  torch.sqrt(1.0 - alpha_bar_t) * noise)
                # Splice: known pixels from noised GT, missing pixels from x_t
                x_t = x_known_noised * known_mask_2ch + x_t * missing_mask_2ch

            # Build UNet input
            if mask_xt:
                indep_noise = torch.randn_like(x_t)
                x_t_input = x_t * mask_1ch + indep_noise * (1.0 - mask_1ch)
            else:
                x_t_input = x_t

            # Build 8-channel input: [x_t(2), mask(1), known_context(5)]
            x_cond = torch.cat([x_t_input, mask_1ch, known_context], dim=1)

            # UNet predicts epsilon
            pred_eps = ddpm.network(x_cond, t_tensor.reshape(1, -1))

            # Standard DDPM reverse step (eps-prediction)
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

    # Final splice: put exact known values at t=0
    if repaint and x0_known_std is not None:
        x_t = x0_known_std * known_mask_2ch + x_t * missing_mask_2ch

    return x_t


def _build_repaint_schedule(n_steps, jump_size=10, n_resample=3):
    """Build RePaint time schedule with resample jumps.

    At every `jump_size` steps, go back `jump_size` steps and re-denoise
    `n_resample` times. This improves coherence between known/unknown regions.

    Returns list of timestep indices to process.
    """
    schedule = []
    t = n_steps - 1
    while t >= 0:
        # Denoise for jump_size steps (or remaining)
        steps_this_seg = min(jump_size, t + 1)
        for _ in range(n_resample):
            # Forward: denoise these steps
            for s in range(steps_this_seg):
                schedule.append(t - s)
            # If not last resample, jump back up
            if _ < n_resample - 1 and t - steps_this_seg + 1 > 0:
                # Re-noise: add the jump-back timestep
                schedule.append(t)  # signals forward jump
        t -= steps_this_seg
    return schedule


def run_inference():
    print(f"{'='*70}")
    print(f"GP-Context DDPM Inference")
    print(f"{'='*70}")

    # ── Load data ─────────────────────────────────────────────────────
    dd = DDInitializer()
    device = dd.get_device()
    standardizer = ZScoreStandardizer(U_MEAN, U_STD, V_MEAN, V_STD)

    # Load GP cache
    print(f"Loading GP cache from {GP_CACHE_PATH}...")
    gp_data = torch.load(GP_CACHE_PATH, map_location="cpu", weights_only=False)
    gp_test_raw = gp_data["gp_test"]       # (N_test, 2, H, W) raw space
    var_test_raw = gp_data["var_test"]      # (N_test, 2, H, W) raw space
    mask_1ch = gp_data["mask_1ch"]          # (1, 1, H, W)
    dist_map = gp_data["dist_map"]          # (1, H, W) or (1, 1, H, W)

    # Normalize mask dimensions
    if mask_1ch.dim() == 4:
        mask_1ch = mask_1ch.squeeze(0)      # (1, H, W)
    if dist_map.dim() == 4:
        dist_map = dist_map.squeeze(0)      # (1, H, W)

    # Standardize GP means
    gp_test_std = standardizer(gp_test_raw)  # (N_test, 2, H, W)

    # GP variance: max over u,v → normalize to [0, 1]
    var_max = var_test_raw.max(dim=1, keepdim=True).values  # (N_test, 1, H, W)
    # Use global max from TRAINING data for consistency
    # (the dataset uses global max across train+test, but at inference we need
    #  the same normalization — use the max from the test set as approximation,
    #  or ideally the same normalizer as training)
    gp_train_var = gp_data["var_train"]
    var_train_max = gp_train_var.max(dim=1, keepdim=True).values
    var_global_max = max(var_train_max.max().item(), var_max.max().item())
    if var_global_max > 0:
        var_test_norm = var_max / var_global_max  # (N_test, 1, H, W)
    else:
        var_test_norm = var_max

    # Ocean mask
    H, W = mask_1ch.shape[-2:]
    ocean_mask = torch.zeros(1, H, W)
    ocean_mask[0, :OCEAN_H, :OCEAN_W] = 1.0

    print(f"  GP test: {gp_test_raw.shape}, var: {var_test_raw.shape}")
    print(f"  Mask shape: {mask_1ch.shape}, dist: {dist_map.shape}")
    print(f"  var_global_max: {var_global_max:.6f}")
    n_known = (mask_1ch == 0).sum().item()
    n_total_ocean = OCEAN_H * OCEAN_W
    print(f"  Known pixels: {n_known}, ocean area: {n_total_ocean}, "
          f"mask coverage: {100*(1 - n_known/n_total_ocean):.1f}%")

    # ── Load model ────────────────────────────────────────────────────
    print(f"\nLoading model weights from {WEIGHTS_PATH}...")
    network = MyUNet_Attn(n_steps=N_STEPS, time_emb_dim=256, in_channels=8)
    ddpm = GaussianDDPM(
        network, n_steps=N_STEPS,
        min_beta=MIN_BETA, max_beta=MAX_BETA, device=device,
    )
    state = torch.load(WEIGHTS_PATH, map_location=device, weights_only=False)
    ddpm.load_state_dict(state)
    ddpm = ddpm.to(device)
    ddpm.eval()

    n_params = sum(p.numel() for p in ddpm.parameters()) / 1e6
    print(f"  Model: MyUNet_Attn(in_channels=8), {n_params:.1f}M params")
    print(f"  Prediction target: eps, T={N_STEPS}")
    print(f"  mask_xt: {args.mask_xt}")

    # ── Get test data ─────────────────────────────────────────────────
    test_data = dd.get_test_data()
    n_test = len(test_data)
    n_samples = min(args.n_samples, n_test)
    print(f"\nRunning inference on {n_samples} test samples...")

    # ── Inference loop ────────────────────────────────────────────────
    mask_1ch_dev = mask_1ch.unsqueeze(0).to(device)  # (1, 1, H, W)
    mask_2ch = mask_1ch_dev.expand(-1, 2, -1, -1)    # (1, 2, H, W)
    dist_dev = dist_map.unsqueeze(0).to(device)       # (1, 1, H, W)
    ocean_dev = ocean_mask.unsqueeze(0).to(device)    # (1, 1, H, W)

    completed = []
    gp_mses = []
    ddpm_mses = []
    repaint_mses = []
    resample_mses = []

    print(f"\n{'#':>4}  {'GP MSE':>12}  {'Vanilla MSE':>12}  {'RePaint MSE':>12}  {'Resample MSE':>12}  {'Time':>7}")
    print(f"{'-'*80}")
    t0_global = time.time()

    for i in range(n_samples):
        t0 = time.time()
        torch.manual_seed(args.seed + i)

        # Get ground truth (standardized by DDInitializer)
        x0_dd_std = test_data[i][0]  # (2, H, W) standardized
        dd_std = dd.get_standardizer()
        x0_raw = dd_std.unstandardize(x0_dd_std).unsqueeze(0)  # (1, 2, H, W) raw

        # Re-standardize with per-component z-score
        x0_std = standardizer(x0_raw.squeeze(0)).unsqueeze(0).to(device)  # (1, 2, H, W)

        # GP fields for this sample (already loaded from cache)
        gp_mean_std = gp_test_std[i].unsqueeze(0).to(device)   # (1, 2, H, W)
        gp_var_i = var_test_norm[i].unsqueeze(0).to(device)     # (1, 1, H, W)

        # Build known_context: [gp_mean_u, gp_mean_v, gp_var, dist, ocean]
        known_context = torch.cat([
            gp_mean_std,   # (1, 2, H, W)
            gp_var_i,      # (1, 1, H, W)
            dist_dev,      # (1, 1, H, W)
            ocean_dev,     # (1, 1, H, W)
        ], dim=1)  # (1, 5, H, W)

        # GP baseline MSE (in raw space, missing region only)
        gp_raw = gp_test_raw[i].unsqueeze(0).to(device)  # (1, 2, H, W)
        x0_raw_dev = x0_raw.to(device)
        gp_mse = ((gp_raw - x0_raw_dev) * mask_2ch).pow(2).sum() / (
            mask_2ch.sum() + 1e-8)

        # Shared initial noise for fair comparison
        x_T = torch.randn(1, 2, H, W, device=device)

        # Method 1: Vanilla DDPM (no RePaint)
        x0_vanilla_std = gp_context_reverse(
            ddpm, x_T, mask_1ch_dev, known_context,
            device=device, mask_xt=args.mask_xt,
            seed=args.seed + i,
            repaint=False,
        )
        x0_vanilla_raw = standardizer.unstandardize(
            x0_vanilla_std.squeeze(0).cpu()
        ).unsqueeze(0).to(device)
        vanilla_mse = ((x0_vanilla_raw - x0_raw_dev) * mask_2ch).pow(2).sum() / (
            mask_2ch.sum() + 1e-8)

        # Method 2: RePaint known-region replacement (no resample jumps)
        x0_repaint_std = gp_context_reverse(
            ddpm, x_T.clone(), mask_1ch_dev, known_context,
            device=device, mask_xt=args.mask_xt,
            seed=args.seed + i,
            x0_known_std=x0_std,
            repaint=True,
            resample_jumps=0, resample_count=1,
        )
        x0_repaint_raw = standardizer.unstandardize(
            x0_repaint_std.squeeze(0).cpu()
        ).unsqueeze(0).to(device)
        repaint_mse = ((x0_repaint_raw - x0_raw_dev) * mask_2ch).pow(2).sum() / (
            mask_2ch.sum() + 1e-8)

        # Method 3: RePaint + resample jumps (full RePaint)
        x0_resample_std = gp_context_reverse(
            ddpm, x_T.clone(), mask_1ch_dev, known_context,
            device=device, mask_xt=args.mask_xt,
            seed=args.seed + i,
            x0_known_std=x0_std,
            repaint=True,
            resample_jumps=10, resample_count=3,
        )
        x0_resample_raw = standardizer.unstandardize(
            x0_resample_std.squeeze(0).cpu()
        ).unsqueeze(0).to(device)
        resample_mse = ((x0_resample_raw - x0_raw_dev) * mask_2ch).pow(2).sum() / (
            mask_2ch.sum() + 1e-8)

        elapsed = time.time() - t0

        print(f"{i+1:>4}  {gp_mse.item():>12.6f}  {vanilla_mse.item():>12.6f}  "
              f"{repaint_mse.item():>12.6f}  {resample_mse.item():>12.6f}  {elapsed:>6.1f}s")

        completed.append({
            "idx": i,
            "ground_truth": x0_raw_dev.cpu(),
            "gp_output": gp_raw.cpu(),
            "ddpm_vanilla": x0_vanilla_raw.cpu(),
            "ddpm_repaint": x0_repaint_raw.cpu(),
            "ddpm_resample": x0_resample_raw.cpu(),
            "missing_mask": mask_2ch.cpu(),
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
        "model": "GP-context MyUNet_Attn(in_channels=8), eps-prediction",
        "weights": WEIGHTS_PATH,
        "mask_xt": args.mask_xt,
        "use_ema": args.use_ema,
        "seed": args.seed,
        "methods": ["gp", "vanilla", "repaint", "resample"],
    }, PT_PATH)
    print(f"\nResults saved to {PT_PATH}")

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"GP-Context DDPM — {len(completed)} samples, {total_time:.0f}s total")
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
    print(f"{'='*70}")


if __name__ == "__main__":
    run_inference()
