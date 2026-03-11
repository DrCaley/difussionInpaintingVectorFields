#!/usr/bin/env python3
"""
GP-CNN → DDPM Hybrid Ensemble Evaluation

1. Run GP-CNN to get initial reconstruction (best deterministic baseline)
2. Use DDPM single-step ensemble (N members at timestep t) for:
   - Refined mean prediction (potentially better MSE)
   - Per-pixel uncertainty (ensemble std)
3. Also retrieve GP posterior variance per pixel
4. Compare DDPM-ensemble uncertainty vs GP variance:
   - Which better predicts actual reconstruction error?
   - Pixel-wise correlation with |error|
   - Calibration curves
   - Oracle ranking (quintile test)

Uses same 100 balanced samples as all other evaluations.

Usage:
    PYTHONPATH=. python scripts/gp_cnn_hybrid_eval.py [--n-samples 100] [--n-ensemble 10] [--timestep 100]
"""

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from scripts.voronoi_cnn_model import VoronoiCNN
from ddpm.neural_networks.ddpm import GaussianDDPM
from ddpm.neural_networks.unets.unet_film_attn import MyUNet_FiLM_Attn
from ddpm.helper_functions.standardize_data import ZScoreStandardizer
from ddpm.helper_functions.interpolation_tool import gp_fill

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OCEAN_H, OCEAN_W = 44, 94
FULL_H, FULL_W = 64, 128
OBS_ROW = 22
N_STEPS = 250

# Per-component stats (used by GP-CNN for normalization)
U_MEAN, U_STD = -0.06929559429949586, 0.1358005549716049
V_MEAN, V_STD = -0.0323937796117541, 0.08899177232117582

# DDPM uses per-component ZScoreStandardizer (same as in tmp_vcnn_ddpm_hybrid.py)
ddpm_standardizer = ZScoreStandardizer(U_MEAN, U_STD, V_MEAN, V_STD)

GP_PARAMS = {
    "lengthscale": 14.1,
    "variance": 0.0103420345,
    "noise": 1e-8,
    "kernel_type": "rbf_legacy",
    "coord_system": "pixels",
}

DDPM_WEIGHT_PATH = "experiments/06_gp_forward/gp_conditioned/results/inpaint_gaussian_t250_best_ema_weights.pt"
GP_CNN_CKPT = "results/gp_cnn/gp_cnn_best.pt"
DDPM_EVAL_PT = "results/eddy_balanced_eval/bulk_eval_eddy_balanced_100.pt"
OUT_DIR = Path("results/gp_cnn_hybrid_eval")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_pickle_data(path="data.pickle"):
    with open(path, "rb") as f:
        train_np, val_np, _test_np = pickle.load(f)

    def to_tensor(arr):
        t = torch.from_numpy(np.ascontiguousarray(arr)).float()
        t = t.permute(3, 2, 1, 0)
        t = torch.nan_to_num(t, nan=0.0)
        return t

    return to_tensor(train_np), to_tensor(val_np)


# ---------------------------------------------------------------------------
# GP computation (with variance)
# ---------------------------------------------------------------------------

def compute_gp(vel_phys, obs_mask, ocean_mask):
    """
    Compute GP posterior mean AND variance for a single sample.

    Returns:
        gp_mean: (2, OCEAN_H, OCEAN_W) GP posterior mean in physical space
        gp_var:  (2, OCEAN_H, OCEAN_W) GP posterior variance
    """
    vel_full = np.zeros((1, 2, FULL_H, FULL_W), dtype=np.float32)
    vel_full[0, :, :OCEAN_H, :OCEAN_W] = vel_phys * ocean_mask[None]

    gp_mask = np.ones((FULL_H, FULL_W), dtype=np.float32)
    gp_mask[:OCEAN_H, :OCEAN_W] = 1.0 - obs_mask
    gp_mask[:OCEAN_H, :OCEAN_W] *= ocean_mask
    gp_mask[OCEAN_H:, :] = 0.0
    gp_mask[:, OCEAN_W:] = 0.0

    gp_mask_t = torch.from_numpy(gp_mask).unsqueeze(0).unsqueeze(0).expand(1, 2, -1, -1).clone()
    vel_t = torch.from_numpy(vel_full)

    gp_mean, gp_var = gp_fill(
        vel_t, gp_mask_t,
        lengthscale=GP_PARAMS["lengthscale"],
        variance=GP_PARAMS["variance"],
        noise=GP_PARAMS["noise"],
        use_double=True,
        kernel_type=GP_PARAMS["kernel_type"],
        coord_system=GP_PARAMS["coord_system"],
        return_variance=True,
    )

    gp_mean_ocean = gp_mean[0, :, :OCEAN_H, :OCEAN_W].numpy()
    gp_var_ocean = gp_var[0, :, :OCEAN_H, :OCEAN_W].numpy()

    return gp_mean_ocean, gp_var_ocean


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_ddpm(device):
    network = MyUNet_FiLM_Attn(n_steps=N_STEPS, time_emb_dim=256, in_channels=5)
    ddpm = GaussianDDPM(network, n_steps=N_STEPS, min_beta=0.0001, max_beta=0.02, device=device)
    ddpm.load_state_dict(torch.load(DDPM_WEIGHT_PATH, map_location="cpu", weights_only=False))
    ddpm = ddpm.to(device)
    ddpm.eval()
    return ddpm


def load_gp_cnn(device):
    ckpt = torch.load(GP_CNN_CKPT, map_location="cpu", weights_only=False)
    cfg = ckpt["model_config"]
    model = VoronoiCNN(
        in_channels=cfg["in_channels"],
        out_channels=cfg["out_channels"],
        base_ch=cfg.get("base_ch", 32),
        depth=cfg.get("depth", 3),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


# ---------------------------------------------------------------------------
# GP-CNN inference (returns both physical and DDPM-standardized)
# ---------------------------------------------------------------------------

def gp_cnn_predict(model, gp_mean_raw, obs_mask, ocean_mask, norm_mean, norm_std,
                   gp_std_map, device):
    """
    Run GP-CNN inference.

    gp_mean_raw: (2, H, W) GP posterior mean in physical space
    Returns:
        pred_phys: (1, 2, 64, 128) physical-space prediction
        pred_ddpm_std: (1, 2, 64, 128) in DDPM standardized space
    """
    # Normalize GP mean
    gp_mean_norm = (gp_mean_raw - norm_mean[:, None, None]) / norm_std[:, None, None]
    gp_mean_norm *= ocean_mask[None, :, :]

    # 6-channel input
    gp_mean_t = torch.from_numpy(gp_mean_norm.astype(np.float32))
    gp_std_t = torch.from_numpy(gp_std_map.astype(np.float32))
    sensor_t = torch.from_numpy(obs_mask.astype(np.float32)).unsqueeze(0)
    ocean_ch = torch.from_numpy(ocean_mask.astype(np.float32)).unsqueeze(0)

    gp_input = torch.cat([gp_mean_t, gp_std_t, sensor_t, ocean_ch], dim=0)
    gp_input = gp_input.unsqueeze(0).to(device)

    with torch.no_grad():
        pred_n = model(gp_input)  # (1, 2, 44, 94) in per-component z-score

    # Un-normalize to physical
    mean_t = torch.tensor(norm_mean, dtype=torch.float32).view(1, 2, 1, 1).to(device)
    std_t = torch.tensor(norm_std, dtype=torch.float32).view(1, 2, 1, 1).to(device)
    pred_phys_small = pred_n * std_t + mean_t

    # Zero land
    ocean_t = torch.from_numpy(ocean_mask).float().to(device).unsqueeze(0).unsqueeze(0)
    pred_phys_small = pred_phys_small * ocean_t

    # Embed in 64x128 grid
    pred_phys = torch.zeros(1, 2, FULL_H, FULL_W, device=device)
    pred_phys[:, :, :OCEAN_H, :OCEAN_W] = pred_phys_small

    # Convert to DDPM standardized space
    pred_ddpm_std = ddpm_standardizer(pred_phys.squeeze(0)).unsqueeze(0)

    return pred_phys, pred_ddpm_std


# ---------------------------------------------------------------------------
# DDPM ensemble refinement
# ---------------------------------------------------------------------------

def ddpm_single_step(ddpm, cond_std_field, missing_mask_1ch, t_val, seed, device):
    """Single-step DDPM x0 prediction: noise cond field → predict x0."""
    torch.manual_seed(seed)
    alpha_bar = ddpm.alpha_bars[t_val].to(device)
    noise = torch.randn_like(cond_std_field)
    noisy = alpha_bar.sqrt() * cond_std_field + (1 - alpha_bar).sqrt() * noise
    # 5-channel input: [noisy(2ch), missing_mask(1ch), conditioning(2ch)]
    x_cond = torch.cat([noisy, missing_mask_1ch, cond_std_field], dim=1)
    time_tensor = torch.full((1, 1), t_val, device=device, dtype=torch.long)
    with torch.no_grad():
        return ddpm.network(x_cond, time_tensor)  # predicted x0 in DDPM-standardized space


def ddpm_ensemble(ddpm, cond_std_field, missing_mask_1ch, t_val, n_ens, device):
    """Run N ensemble members with different seeds. Returns (mean, std) in DDPM-std space."""
    preds = []
    for k in range(n_ens):
        pred = ddpm_single_step(ddpm, cond_std_field, missing_mask_1ch, t_val,
                                seed=42 + k * 1000, device=device)
        preds.append(pred)
    stack = torch.stack(preds, dim=0)  # (N, 1, 2, 64, 128)
    return stack.mean(dim=0), stack.std(dim=0)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_mse_phys(pred_phys, gt_phys, missing_mask):
    """MSE over missing ocean pixels in physical space. Both (1,2,64,128)."""
    return ((pred_phys - gt_phys) * missing_mask).pow(2).sum() / (missing_mask.sum() + 1e-8)


def unstd_ddpm(t):
    """Convert DDPM-standardized (1,2,H,W) to physical space."""
    return ddpm_standardizer.unstandardize(t.squeeze(0)).unsqueeze(0)


# ---------------------------------------------------------------------------
# Uncertainty evaluation functions
# ---------------------------------------------------------------------------

def pixel_wise_correlation(errors, uncertainties, mask):
    """
    Computes correlation between |error| and uncertainty across pixels.
    errors: (2, H, W), uncertainties: (2, H, W), mask: (H, W) boolean
    Returns: correlation coefficient
    """
    err_flat = errors[:, mask].flatten().numpy()
    unc_flat = uncertainties[:, mask].flatten().numpy()
    if len(err_flat) < 10:
        return 0.0
    return float(np.corrcoef(err_flat, unc_flat)[0, 1])


def oracle_quintile_test(errors, uncertainties, mask):
    """
    Sort pixels by uncertainty into 5 bins. Check if higher-uncertainty bins
    have higher error (monotonic relationship = good calibration).
    Returns: list of 5 mean-error values, one per quintile (low→high uncertainty).
    """
    err_flat = errors[:, mask].flatten().numpy()
    unc_flat = uncertainties[:, mask].flatten().numpy()
    n = len(err_flat)
    if n < 50:
        return [0.0] * 5
    idx = np.argsort(unc_flat)
    q_size = n // 5
    return [float(err_flat[idx[i * q_size:(i + 1) * q_size]].mean()) for i in range(5)]


def calibration_curve(errors_all, uncertainties_all, n_bins=10):
    """
    For z-scores: |error| / uncertainty. Fraction within k std should be ≈ expected.
    Returns: (expected_coverage, actual_coverage) for k=0.5,1.0,...,3.0
    """
    from scipy.stats import norm as sp_norm
    z = errors_all / (uncertainties_all + 1e-10)
    thresholds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    expected = [2 * sp_norm.cdf(k) - 1 for k in thresholds]
    actual = [float((z < k).mean()) for k in thresholds]
    return thresholds, expected, actual


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--n-ensemble", type=int, default=10)
    parser.add_argument("--timestep", type=int, default=100)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Load models
    print("Loading DDPM...")
    ddpm = load_ddpm(device)
    print("Loading GP-CNN...")
    gpcnn, ckpt = load_gp_cnn(device)

    norm_mean = ckpt["norm_mean"].numpy()
    norm_std = ckpt["norm_std"].numpy()
    ocean_mask = ckpt["ocean_mask"]
    gp_std_map = ckpt["gp_std_map"]

    # Load raw validation data
    _, val_vel = load_pickle_data()

    # Observation mask
    row22_mask = np.zeros((OCEAN_H, OCEAN_W), dtype=np.float32)
    row22_mask[OBS_ROW, :] = 1.0
    row22_mask *= ocean_mask

    # Build missing mask for DDPM (64x128 grid: 1=known, 0=missing)
    # In DDPM convention: missing_mask_1ch has 1 where KNOWN
    from ddpm.helper_functions.masks.border_mask import BorderMaskGenerator
    border_gen = BorderMaskGenerator()

    # Load eval indices
    ddpm_data = torch.load(DDPM_EVAL_PT, map_location="cpu", weights_only=False)
    ddpm_samples = ddpm_data["samples"]
    val_indices = [s["val_idx"] for s in ddpm_samples]
    eddy_set = set(ddpm_data.get("eddy_indices", []))
    n_total = min(args.n_samples, len(val_indices))
    val_indices = val_indices[:n_total]

    print(f"\nRunning GP-CNN → DDPM Ens{args.n_ensemble}@t={args.timestep}")
    print(f"on {n_total} balanced samples")
    print(f"{'=' * 80}")

    results = []
    t0_global = time.time()

    for run_i, vi in enumerate(val_indices):
        t0 = time.time()
        gt_vel_small = val_vel[vi].numpy()  # (2, 44, 94) physical

        # Embed GT in 64x128
        gt_full = torch.zeros(1, 2, FULL_H, FULL_W)
        gt_full[0, :, :OCEAN_H, :OCEAN_W] = torch.from_numpy(gt_vel_small)

        # ── Step 1: GP with variance ──
        gp_mean, gp_var = compute_gp(gt_vel_small, row22_mask, ocean_mask)
        # gp_mean: (2, 44, 94), gp_var: (2, 44, 94)

        # Embed GP in 64x128 for MSE computation
        gp_full = torch.zeros(1, 2, FULL_H, FULL_W)
        gp_full[0, :, :OCEAN_H, :OCEAN_W] = torch.from_numpy(gp_mean)

        # ── Step 2: GP-CNN prediction ──
        gpcnn_phys, gpcnn_ddpm_std = gp_cnn_predict(
            gpcnn, gp_mean, row22_mask, ocean_mask,
            norm_mean, norm_std, gp_std_map, device
        )
        gpcnn_phys_cpu = gpcnn_phys.cpu()

        # ── Step 3: Build DDPM masks ──
        # missing_mask: 1=known, 0=missing (DDPM convention)
        land_mask = (gt_full.abs() > 1e-5).float()
        raw_mask = torch.ones(1, 1, FULL_H, FULL_W)
        raw_mask[0, 0, OBS_ROW, :OCEAN_W] = 0.0  # Row 22 is observed → 0 means "this is known" in missing_mask? 
        # Actually let me reconsider. In the DDPM hybrid code, missing_mask has 1=known.
        # The border mask sets edges to 0, and the known (observed) pixels are set to 0 in the raw_mask.
        # Wait, let me re-read the existing code...

        # From tmp_vcnn_ddpm_hybrid.py prepare_sample():
        # raw_mask starts as all 1s, then sets row22 to 0 (known),
        # then multiplies by border and land_mask.
        # So: missing_mask_1ch = 1 where missing ocean, 0 where known or land or border
        # BUT in ddpm_refine_vcnn, missing_mask is used but not actually applied to the output.
        # The DDPM just sees [noisy, missing_mask_1ch, conditioning] as 5 channels.
        # The missing_mask_1ch tells the network WHERE the missing region is.

        border = border_gen.generate_mask(torch.Size([1, 2, FULL_H, FULL_W])).cpu()
        raw_mask = raw_mask * border
        missing_mask_1ch = raw_mask * land_mask[:, 0:1].cpu()
        missing_mask = missing_mask_1ch.expand(-1, 2, -1, -1)

        missing_mask_1ch_dev = missing_mask_1ch.to(device)
        missing_mask_dev = missing_mask.to(device)

        # ── Step 4: DDPM ensemble refinement of GP-CNN ──
        gpcnn_ddpm_std_dev = gpcnn_ddpm_std.to(device)
        ens_mean_std, ens_std_std = ddpm_ensemble(
            ddpm, gpcnn_ddpm_std_dev, missing_mask_1ch_dev,
            args.timestep, args.n_ensemble, device
        )

        # Convert ensemble mean back to physical space
        ens_mean_phys = unstd_ddpm(ens_mean_std.cpu())
        ens_std_phys = ens_std_std.cpu()  # Still in DDPM-std space

        # Convert ensemble std to physical space (scale by per-component std)
        # DDPM-std = (phys - mean) / std → std_phys = std_ddpm * std_component
        ens_std_phys_scaled = torch.zeros_like(ens_std_phys)
        ens_std_phys_scaled[:, 0:1] = ens_std_phys[:, 0:1] * U_STD
        ens_std_phys_scaled[:, 1:2] = ens_std_phys[:, 1:2] * V_STD

        # ── Step 5: Compute errors and metrics ──
        # Missing ocean mask (boolean, 44x94)
        known_mask = torch.from_numpy(row22_mask).bool()  # (44, 94)
        ocean_bool = torch.from_numpy(ocean_mask).bool()   # (44, 94)
        missing_ocean = ocean_bool & ~known_mask            # (44, 94)

        gt_small_t = torch.from_numpy(gt_vel_small)  # (2, 44, 94)

        # GP-CNN absolute error
        gpcnn_err = (gpcnn_phys_cpu[0, :, :OCEAN_H, :OCEAN_W] - gt_small_t).abs()

        # Ensemble mean absolute error
        ens_err = (ens_mean_phys[0, :, :OCEAN_H, :OCEAN_W] - gt_small_t).abs()

        # GP-only absolute error
        gp_err = (torch.from_numpy(gp_mean) - gt_small_t).abs()

        # Uncertainties to compare:
        # 1) GP posterior variance → std (sqrt of variance)
        gp_std = np.sqrt(np.clip(gp_var, 0, None))  # (2, 44, 94)
        gp_std_t = torch.from_numpy(gp_std.astype(np.float32))

        # 2) DDPM ensemble std (per pixel) - already in physical space
        ens_std_small = ens_std_phys_scaled[0, :, :OCEAN_H, :OCEAN_W]

        # MSEs
        gp_mse = gp_err[:, missing_ocean].pow(2).mean().item()
        gpcnn_mse = gpcnn_err[:, missing_ocean].pow(2).mean().item()
        ens_mse = ens_err[:, missing_ocean].pow(2).mean().item()

        # Correlations (how well does uncertainty predict |error|?)
        # Use GP-CNN error (since GP std is for GP predictions)
        # Actually: GP variance reflects uncertainty of GP predictions
        #           DDPM ensemble std reflects uncertainty of the ensemble-refined predictions
        # For fair comparison, we correlate each uncertainty with its own method's error:
        #   GP_std ↔ |GP_error|
        #   DDPM_ens_std ↔ |ensemble_error|
        # AND also cross-compare GP_std with GP-CNN error (since GP-CNN starts from GP)

        corr_gp_vs_gperr = pixel_wise_correlation(gp_err, gp_std_t, missing_ocean)
        corr_ens_vs_enserr = pixel_wise_correlation(ens_err, ens_std_small, missing_ocean)
        corr_gp_vs_gpcnnerr = pixel_wise_correlation(gpcnn_err, gp_std_t, missing_ocean)
        corr_ens_vs_gpcnnerr = pixel_wise_correlation(gpcnn_err, ens_std_small, missing_ocean)

        # Quintile tests
        quint_gp = oracle_quintile_test(gp_err, gp_std_t, missing_ocean)
        quint_ens = oracle_quintile_test(ens_err, ens_std_small, missing_ocean)

        elapsed = time.time() - t0
        tag = "EDDY" if vi in eddy_set else "clean"
        print(f"  [{run_i+1:3d}/{n_total}] vi={vi:5d} {tag:>5}  "
              f"GP={gp_mse:.6f}  CNN={gpcnn_mse:.6f}  Ens={ens_mse:.6f}  "
              f"corr(GP)={corr_gp_vs_gperr:.3f}  corr(Ens)={corr_ens_vs_enserr:.3f}  "
              f"({elapsed:.1f}s)")

        results.append({
            "val_idx": vi,
            "is_eddy": vi in eddy_set,
            # MSEs
            "gp_mse": gp_mse,
            "gpcnn_mse": gpcnn_mse,
            "ens_mse": ens_mse,
            # Correlations
            "corr_gp_vs_gperr": corr_gp_vs_gperr,
            "corr_ens_vs_enserr": corr_ens_vs_enserr,
            "corr_gp_vs_gpcnnerr": corr_gp_vs_gpcnnerr,
            "corr_ens_vs_gpcnnerr": corr_ens_vs_gpcnnerr,
            # Quintiles
            "quintiles_gp": quint_gp,
            "quintiles_ens": quint_ens,
            # Mean uncertainty values
            "mean_gp_std": float(gp_std_t[:, missing_ocean].mean()),
            "mean_ens_std": float(ens_std_small[:, missing_ocean].mean()),
            # Per-pixel data for calibration (aggregate later)
            "gp_err_flat": gp_err[:, missing_ocean].flatten().numpy(),
            "gp_std_flat": gp_std_t[:, missing_ocean].flatten().numpy(),
            "ens_err_flat": ens_err[:, missing_ocean].flatten().numpy(),
            "ens_std_flat": ens_std_small[:, missing_ocean].flatten().numpy(),
        })

    elapsed_total = time.time() - t0_global
    print(f"\nDone: {n_total} samples in {elapsed_total:.1f}s "
          f"({elapsed_total/n_total:.2f}s/sample)")

    # ==================================================================
    # Aggregate results
    # ==================================================================
    print(f"\n{'=' * 80}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 80}")

    gp_mses = [r["gp_mse"] for r in results]
    gpcnn_mses = [r["gpcnn_mse"] for r in results]
    ens_mses = [r["ens_mse"] for r in results]

    mean_gp = np.mean(gp_mses)
    mean_gpcnn = np.mean(gpcnn_mses)
    mean_ens = np.mean(ens_mses)

    print(f"\n--- MSE Comparison ---")
    print(f"  {'Method':<30} {'Mean MSE':>12} {'Ratio vs GP':>12}")
    print(f"  {'-' * 56}")
    print(f"  {'GP baseline':<30} {mean_gp:>12.6f} {'1.000x':>12}")
    print(f"  {'GP-CNN':<30} {mean_gpcnn:>12.6f} {mean_gpcnn/mean_gp:>11.3f}x")
    print(f"  {f'GP-CNN→Ens{args.n_ensemble}@t={args.timestep}':<30} {mean_ens:>12.6f} {mean_ens/mean_gp:>11.3f}x")

    # Win rates
    wins_ens_gp = sum(1 for e, g in zip(ens_mses, gp_mses) if e < g)
    wins_ens_cnn = sum(1 for e, c in zip(ens_mses, gpcnn_mses) if e < c)
    wins_cnn_gp = sum(1 for c, g in zip(gpcnn_mses, gp_mses) if c < g)
    print(f"\n  Win rates:")
    print(f"    GP-CNN beats GP:         {wins_cnn_gp}/{n_total}")
    print(f"    Ensemble beats GP:       {wins_ens_gp}/{n_total}")
    print(f"    Ensemble beats GP-CNN:   {wins_ens_cnn}/{n_total}")

    # ==================================================================
    # Uncertainty comparison
    # ==================================================================
    print(f"\n{'=' * 80}")
    print("UNCERTAINTY COMPARISON: DDPM Ensemble Std vs GP Posterior Std")
    print(f"{'=' * 80}")

    # 1) Mean pixel-wise correlation
    corrs_gp = [r["corr_gp_vs_gperr"] for r in results]
    corrs_ens = [r["corr_ens_vs_enserr"] for r in results]
    corrs_gp_cnn = [r["corr_gp_vs_gpcnnerr"] for r in results]
    corrs_ens_cnn = [r["corr_ens_vs_gpcnnerr"] for r in results]

    print(f"\n--- Test 1: Pixel-wise Correlation(uncertainty, |error|) ---")
    print(f"  Higher = uncertainty better tracks actual error")
    print(f"  {'Uncertainty → Error':<40} {'Mean r':>8} {'Median r':>10}")
    print(f"  {'-' * 60}")
    print(f"  {'GP_std → |GP_error|':<40} {np.mean(corrs_gp):>8.4f} {np.median(corrs_gp):>10.4f}")
    print(f"  {'Ens_std → |Ens_error|':<40} {np.mean(corrs_ens):>8.4f} {np.median(corrs_ens):>10.4f}")
    print(f"  {'GP_std → |GP-CNN_error|':<40} {np.mean(corrs_gp_cnn):>8.4f} {np.median(corrs_gp_cnn):>10.4f}")
    print(f"  {'Ens_std → |GP-CNN_error|':<40} {np.mean(corrs_ens_cnn):>8.4f} {np.median(corrs_ens_cnn):>10.4f}")

    # 2) Oracle quintile test (average across samples)
    print(f"\n--- Test 2: Oracle Quintile Test ---")
    print(f"  Pixels sorted by uncertainty into 5 bins (low→high).")
    print(f"  Good calibration = monotonically increasing error.")
    all_quint_gp = np.array([r["quintiles_gp"] for r in results])
    all_quint_ens = np.array([r["quintiles_ens"] for r in results])
    mean_quint_gp = all_quint_gp.mean(axis=0)
    mean_quint_ens = all_quint_ens.mean(axis=0)
    print(f"  {'Quintile':<10} {'GP_std → |GP_err|':>20} {'Ens_std → |Ens_err|':>22}")
    print(f"  {'-' * 54}")
    for i in range(5):
        label = ["Q1 (low)", "Q2", "Q3", "Q4", "Q5 (high)"][i]
        print(f"  {label:<10} {mean_quint_gp[i]:>20.6f} {mean_quint_ens[i]:>22.6f}")

    gp_monotonic = all(mean_quint_gp[i] <= mean_quint_gp[i+1] for i in range(4))
    ens_monotonic = all(mean_quint_ens[i] <= mean_quint_ens[i+1] for i in range(4))
    gp_ratio = mean_quint_gp[4] / (mean_quint_gp[0] + 1e-10)
    ens_ratio = mean_quint_ens[4] / (mean_quint_ens[0] + 1e-10)
    print(f"\n  GP monotonic:  {'YES ✓' if gp_monotonic else 'NO ✗'}  (Q5/Q1 = {gp_ratio:.2f}x)")
    print(f"  Ens monotonic: {'YES ✓' if ens_monotonic else 'NO ✗'}  (Q5/Q1 = {ens_ratio:.2f}x)")

    # 3) Calibration curve (z-score based)
    print(f"\n--- Test 3: Calibration Curve (z-score) ---")
    print(f"  What fraction of errors fall within k·σ? Ideal: matches Gaussian CDF.")

    # Aggregate all pixel-level data
    all_gp_err = np.concatenate([r["gp_err_flat"] for r in results])
    all_gp_std = np.concatenate([r["gp_std_flat"] for r in results])
    all_ens_err = np.concatenate([r["ens_err_flat"] for r in results])
    all_ens_std = np.concatenate([r["ens_std_flat"] for r in results])

    from scipy.stats import norm as sp_norm

    thresholds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    expected = [2 * sp_norm.cdf(k) - 1 for k in thresholds]

    gp_z = all_gp_err / (all_gp_std + 1e-10)
    ens_z = all_ens_err / (all_ens_std + 1e-10)

    gp_actual = [float((gp_z < k).mean()) for k in thresholds]
    ens_actual = [float((ens_z < k).mean()) for k in thresholds]

    print(f"  {'k·σ':<6} {'Expected':>10} {'GP actual':>12} {'Ens actual':>12} {'GP gap':>10} {'Ens gap':>10}")
    print(f"  {'-' * 62}")
    for k, exp, ga, ea in zip(thresholds, expected, gp_actual, ens_actual):
        print(f"  {k:<6.1f} {exp:>10.3f} {ga:>12.3f} {ea:>12.3f} "
              f"{ga - exp:>+10.3f} {ea - exp:>+10.3f}")

    gp_cal_err = np.mean([abs(a - e) for a, e in zip(gp_actual, expected)])
    ens_cal_err = np.mean([abs(a - e) for a, e in zip(ens_actual, expected)])
    print(f"\n  Mean absolute calibration error:")
    print(f"    GP:       {gp_cal_err:.4f}")
    print(f"    Ensemble: {ens_cal_err:.4f}")
    print(f"    {'→ GP better calibrated' if gp_cal_err < ens_cal_err else '→ Ensemble better calibrated'}")

    # 4) Sample-level correlation (does higher mean uncertainty → higher MSE?)
    print(f"\n--- Test 4: Sample-level Uncertainty Ranking ---")
    print(f"  Does higher sample-mean uncertainty predict higher sample MSE?")

    mean_gp_stds = [r["mean_gp_std"] for r in results]
    mean_ens_stds = [r["mean_ens_std"] for r in results]

    from scipy.stats import spearmanr
    rho_gp, p_gp = spearmanr(mean_gp_stds, gp_mses)
    rho_ens, p_ens = spearmanr(mean_ens_stds, ens_mses)
    rho_gp_cnn, p_gp_cnn = spearmanr(mean_gp_stds, gpcnn_mses)

    print(f"  {'Uncertainty → MSE':<35} {'Spearman ρ':>12} {'p-value':>12}")
    print(f"  {'-' * 61}")
    print(f"  {'GP_std → GP_MSE':<35} {rho_gp:>12.4f} {p_gp:>12.2e}")
    print(f"  {'Ens_std → Ens_MSE':<35} {rho_ens:>12.4f} {p_ens:>12.2e}")
    print(f"  {'GP_std → GP-CNN_MSE':<35} {rho_gp_cnn:>12.4f} {p_gp_cnn:>12.2e}")

    # 5) Overall winner
    print(f"\n{'=' * 80}")
    print("SUMMARY: Which uncertainty estimate is better?")
    print(f"{'=' * 80}")
    scores = {"GP posterior std": 0, "DDPM ensemble std": 0}

    # Correlation
    if np.mean(corrs_gp) > np.mean(corrs_ens):
        scores["GP posterior std"] += 1
        print(f"  [Pixel correlation]   GP wins ({np.mean(corrs_gp):.4f} vs {np.mean(corrs_ens):.4f})")
    else:
        scores["DDPM ensemble std"] += 1
        print(f"  [Pixel correlation]   Ensemble wins ({np.mean(corrs_ens):.4f} vs {np.mean(corrs_gp):.4f})")

    # Quintile monotonicity + ratio
    if gp_monotonic and not ens_monotonic:
        scores["GP posterior std"] += 1
        print(f"  [Quintile ordering]   GP wins (monotonic={gp_monotonic}, ratio={gp_ratio:.2f}x)")
    elif ens_monotonic and not gp_monotonic:
        scores["DDPM ensemble std"] += 1
        print(f"  [Quintile ordering]   Ensemble wins (monotonic={ens_monotonic}, ratio={ens_ratio:.2f}x)")
    elif gp_ratio > ens_ratio:
        scores["GP posterior std"] += 1
        print(f"  [Quintile ordering]   GP wins (ratio={gp_ratio:.2f}x vs {ens_ratio:.2f}x)")
    else:
        scores["DDPM ensemble std"] += 1
        print(f"  [Quintile ordering]   Ensemble wins (ratio={ens_ratio:.2f}x vs {gp_ratio:.2f}x)")

    # Calibration
    if gp_cal_err < ens_cal_err:
        scores["GP posterior std"] += 1
        print(f"  [Calibration]         GP wins (err={gp_cal_err:.4f} vs {ens_cal_err:.4f})")
    else:
        scores["DDPM ensemble std"] += 1
        print(f"  [Calibration]         Ensemble wins (err={ens_cal_err:.4f} vs {gp_cal_err:.4f})")

    # Sample ranking
    if abs(rho_gp) > abs(rho_ens):
        scores["GP posterior std"] += 1
        print(f"  [Sample ranking]      GP wins (ρ={rho_gp:.4f} vs {rho_ens:.4f})")
    else:
        scores["DDPM ensemble std"] += 1
        print(f"  [Sample ranking]      Ensemble wins (ρ={rho_ens:.4f} vs {rho_gp:.4f})")

    print(f"\n  FINAL SCORE:  GP = {scores['GP posterior std']}  |  Ensemble = {scores['DDPM ensemble std']}")
    winner = max(scores, key=scores.get)
    if scores["GP posterior std"] == scores["DDPM ensemble std"]:
        print(f"  → TIE")
    else:
        print(f"  → Winner: {winner}")

    # Save results
    save_data = {
        "results": results,
        "config": {
            "n_samples": n_total,
            "n_ensemble": args.n_ensemble,
            "timestep": args.timestep,
        },
        "summary": {
            "gp_mean_mse": mean_gp,
            "gpcnn_mean_mse": mean_gpcnn,
            "ens_mean_mse": mean_ens,
            "corr_gp": float(np.mean(corrs_gp)),
            "corr_ens": float(np.mean(corrs_ens)),
            "gp_cal_err": gp_cal_err,
            "ens_cal_err": ens_cal_err,
            "gp_monotonic": bool(gp_monotonic),
            "ens_monotonic": bool(ens_monotonic),
            "winner": winner if scores["GP posterior std"] != scores["DDPM ensemble std"] else "tie",
        },
    }
    save_path = OUT_DIR / f"hybrid_eval_ens{args.n_ensemble}_t{args.timestep}.pt"
    torch.save(save_data, save_path)
    print(f"\nSaved to {save_path}")


if __name__ == "__main__":
    main()
