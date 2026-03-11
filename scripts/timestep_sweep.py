#!/usr/bin/env python3
"""
Timestep sweep for GP-CNN → DDPM ensemble refinement.

For each timestep t in {25, 50, 75, 100, 150, 200}:
  1. Raw ensemble mean (current approach)
  2. Variance-weighted composite: blend = (1-w)*CNN + w*EnsembleMean
     where w = σ²_GP_norm (normalized GP posterior variance, 0→1)
     Near sensors: w≈0 → keep GP-CNN.  Far from sensors: w≈1 → use ensemble.

Finds optimal timestep for each strategy.

Usage:
    PYTHONPATH=. python scripts/timestep_sweep.py [--n-samples 100] [--n-ensemble 10]
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

U_MEAN, U_STD = -0.06929559429949586, 0.1358005549716049
V_MEAN, V_STD = -0.0323937796117541, 0.08899177232117582

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
OUT_DIR = Path("results/timestep_sweep")

TIMESTEPS = [25, 50, 75, 100, 150, 200]


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
# GP computation
# ---------------------------------------------------------------------------

def compute_gp(vel_phys, obs_mask, ocean_mask):
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

    return gp_mean[0, :, :OCEAN_H, :OCEAN_W].numpy(), gp_var[0, :, :OCEAN_H, :OCEAN_W].numpy()


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
# GP-CNN inference
# ---------------------------------------------------------------------------

def gp_cnn_predict(model, gp_mean_raw, obs_mask, ocean_mask, norm_mean, norm_std,
                   gp_std_map, device):
    gp_mean_norm = (gp_mean_raw - norm_mean[:, None, None]) / norm_std[:, None, None]
    gp_mean_norm *= ocean_mask[None, :, :]

    gp_mean_t = torch.from_numpy(gp_mean_norm.astype(np.float32))
    gp_std_t = torch.from_numpy(gp_std_map.astype(np.float32))
    sensor_t = torch.from_numpy(obs_mask.astype(np.float32)).unsqueeze(0)
    ocean_ch = torch.from_numpy(ocean_mask.astype(np.float32)).unsqueeze(0)

    gp_input = torch.cat([gp_mean_t, gp_std_t, sensor_t, ocean_ch], dim=0)
    gp_input = gp_input.unsqueeze(0).to(device)

    with torch.no_grad():
        pred_n = model(gp_input)

    mean_t = torch.tensor(norm_mean, dtype=torch.float32).view(1, 2, 1, 1).to(device)
    std_t = torch.tensor(norm_std, dtype=torch.float32).view(1, 2, 1, 1).to(device)
    pred_phys_small = pred_n * std_t + mean_t

    ocean_t = torch.from_numpy(ocean_mask).float().to(device).unsqueeze(0).unsqueeze(0)
    pred_phys_small = pred_phys_small * ocean_t

    pred_phys = torch.zeros(1, 2, FULL_H, FULL_W, device=device)
    pred_phys[:, :, :OCEAN_H, :OCEAN_W] = pred_phys_small

    pred_ddpm_std = ddpm_standardizer(pred_phys.squeeze(0)).unsqueeze(0)

    return pred_phys, pred_ddpm_std


# ---------------------------------------------------------------------------
# DDPM single-step and ensemble
# ---------------------------------------------------------------------------

def ddpm_single_step(ddpm, cond_std_field, missing_mask_1ch, t_val, seed, device):
    torch.manual_seed(seed)
    alpha_bar = ddpm.alpha_bars[t_val].to(device)
    noise = torch.randn_like(cond_std_field)
    noisy = alpha_bar.sqrt() * cond_std_field + (1 - alpha_bar).sqrt() * noise
    x_cond = torch.cat([noisy, missing_mask_1ch, cond_std_field], dim=1)
    time_tensor = torch.full((1, 1), t_val, device=device, dtype=torch.long)
    with torch.no_grad():
        return ddpm.network(x_cond, time_tensor)


def ddpm_ensemble(ddpm, cond_std_field, missing_mask_1ch, t_val, n_ens, device):
    preds = []
    for k in range(n_ens):
        pred = ddpm_single_step(ddpm, cond_std_field, missing_mask_1ch, t_val,
                                seed=42 + k * 1000, device=device)
        preds.append(pred)
    stack = torch.stack(preds, dim=0)
    return stack.mean(dim=0), stack.std(dim=0)


def unstd_ddpm(t):
    return ddpm_standardizer.unstandardize(t.squeeze(0)).unsqueeze(0)


# ---------------------------------------------------------------------------
# Variance-weighted compositing
# ---------------------------------------------------------------------------

def compute_gp_var_weight(gp_var, ocean_mask):
    """
    Compute per-pixel weight w ∈ [0, 1] from GP posterior variance.
    w=0 means keep CNN (low GP uncertainty),  w=1 means use ensemble (high GP uncertainty).

    We normalize variance to [0, 1] across ocean pixels.
    Shape: (2, OCEAN_H, OCEAN_W) → (1, 2, FULL_H, FULL_W) weight map.
    """
    ocean_bool = ocean_mask.astype(bool)
    gp_std = np.sqrt(np.clip(gp_var, 0, None))

    # Normalize per-component to [0, 1] across ocean pixels
    w = np.zeros_like(gp_std)
    for c in range(2):
        vals = gp_std[c][ocean_bool]
        vmin, vmax = vals.min(), vals.max()
        if vmax > vmin:
            w[c] = (gp_std[c] - vmin) / (vmax - vmin)
        w[c] *= ocean_mask

    # Embed in full grid
    w_full = np.zeros((1, 2, FULL_H, FULL_W), dtype=np.float32)
    w_full[0, :, :OCEAN_H, :OCEAN_W] = w

    return torch.from_numpy(w_full)


def composite(cnn_phys, ens_phys, weight):
    """
    Variance-weighted composite: keep CNN where confidence is high.
    result = (1 - w) * cnn + w * ensemble
    All inputs: (1, 2, 64, 128)
    """
    return (1.0 - weight) * cnn_phys + weight * ens_phys


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--n-ensemble", type=int, default=10)
    parser.add_argument("--lambdas", nargs="+", type=float, default=None,
                        help="Additional softmax-style lambda values for w = σ²/(σ²+λ) compositing")
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

    # Load models once
    print("Loading DDPM...")
    ddpm = load_ddpm(device)
    print("Loading GP-CNN...")
    gpcnn, ckpt = load_gp_cnn(device)

    norm_mean = ckpt["norm_mean"].numpy()
    norm_std = ckpt["norm_std"].numpy()
    ocean_mask = ckpt["ocean_mask"]
    gp_std_map = ckpt["gp_std_map"]

    _, val_vel = load_pickle_data()

    row22_mask = np.zeros((OCEAN_H, OCEAN_W), dtype=np.float32)
    row22_mask[OBS_ROW, :] = 1.0
    row22_mask *= ocean_mask

    # Build DDPM masks
    from ddpm.helper_functions.masks.border_mask import BorderMaskGenerator
    border_gen = BorderMaskGenerator()

    ddpm_data = torch.load(DDPM_EVAL_PT, map_location="cpu", weights_only=False)
    ddpm_samples = ddpm_data["samples"]
    val_indices = [s["val_idx"] for s in ddpm_samples]
    eddy_set = set(ddpm_data.get("eddy_indices", []))
    n_total = min(args.n_samples, len(val_indices))
    val_indices = val_indices[:n_total]

    print(f"\nTimestep sweep: t ∈ {TIMESTEPS}")
    print(f"Ensemble size N={args.n_ensemble}, samples={n_total}")
    print(f"{'=' * 90}")

    # Storage: per-timestep, per-sample MSEs
    # Keys:  t -> list of dicts
    sweep_results = {t: [] for t in TIMESTEPS}

    t0_global = time.time()

    for run_i, vi in enumerate(val_indices):
        t0 = time.time()
        gt_vel_small = val_vel[vi].numpy()

        gt_full = torch.zeros(1, 2, FULL_H, FULL_W)
        gt_full[0, :, :OCEAN_H, :OCEAN_W] = torch.from_numpy(gt_vel_small)

        # ── GP with variance ──
        gp_mean, gp_var = compute_gp(gt_vel_small, row22_mask, ocean_mask)

        # ── GP-CNN prediction ──
        gpcnn_phys, gpcnn_ddpm_std = gp_cnn_predict(
            gpcnn, gp_mean, row22_mask, ocean_mask,
            norm_mean, norm_std, gp_std_map, device
        )
        gpcnn_phys_cpu = gpcnn_phys.cpu()

        # ── DDPM masks ──
        land_mask = (gt_full.abs() > 1e-5).float()
        raw_mask = torch.ones(1, 1, FULL_H, FULL_W)
        raw_mask[0, 0, OBS_ROW, :OCEAN_W] = 0.0
        border = border_gen.generate_mask(torch.Size([1, 2, FULL_H, FULL_W])).cpu()
        raw_mask = raw_mask * border
        missing_mask_1ch = raw_mask * land_mask[:, 0:1].cpu()
        missing_mask_1ch_dev = missing_mask_1ch.to(device)

        # ── Variance weight for compositing ──
        w = compute_gp_var_weight(gp_var, ocean_mask)

        # ── Missing ocean mask for MSE ──
        known_mask = torch.from_numpy(row22_mask).bool()
        ocean_bool = torch.from_numpy(ocean_mask).bool()
        missing_ocean = ocean_bool & ~known_mask
        gt_small_t = torch.from_numpy(gt_vel_small)

        # GP baseline MSE
        gp_full = torch.zeros(1, 2, FULL_H, FULL_W)
        gp_full[0, :, :OCEAN_H, :OCEAN_W] = torch.from_numpy(gp_mean)
        gp_mse = (torch.from_numpy(gp_mean) - gt_small_t)[:, missing_ocean].pow(2).mean().item()

        # GP-CNN MSE
        gpcnn_mse = (gpcnn_phys_cpu[0, :, :OCEAN_H, :OCEAN_W] - gt_small_t)[:, missing_ocean].pow(2).mean().item()

        gpcnn_ddpm_std_dev = gpcnn_ddpm_std.to(device)

        # ── Sweep over timesteps ──
        for t_val in TIMESTEPS:
            ens_mean_std, ens_std_std = ddpm_ensemble(
                ddpm, gpcnn_ddpm_std_dev, missing_mask_1ch_dev,
                t_val, args.n_ensemble, device
            )

            # Raw ensemble mean in physical space
            ens_mean_phys = unstd_ddpm(ens_mean_std.cpu())

            # Variance-weighted composite
            comp_phys = composite(gpcnn_phys_cpu, ens_mean_phys, w)

            # MSEs over missing ocean
            ens_mse = (ens_mean_phys[0, :, :OCEAN_H, :OCEAN_W] - gt_small_t)[:, missing_ocean].pow(2).mean().item()
            comp_mse = (comp_phys[0, :, :OCEAN_H, :OCEAN_W] - gt_small_t)[:, missing_ocean].pow(2).mean().item()

            # Ensemble std in physical space for uncertainty
            ens_std_phys = torch.zeros_like(ens_std_std.cpu())
            ens_std_phys[:, 0:1] = ens_std_std.cpu()[:, 0:1] * U_STD
            ens_std_phys[:, 1:2] = ens_std_std.cpu()[:, 1:2] * V_STD
            ens_std_small = ens_std_phys[0, :, :OCEAN_H, :OCEAN_W]

            # Composite error for uncertainty correlation
            comp_err = (comp_phys[0, :, :OCEAN_H, :OCEAN_W] - gt_small_t).abs()
            ens_err = (ens_mean_phys[0, :, :OCEAN_H, :OCEAN_W] - gt_small_t).abs()

            # Pixel-wise correlation: ensemble std vs |composite error|
            unc_flat = ens_std_small[:, missing_ocean].flatten().numpy()
            err_flat = comp_err[:, missing_ocean].flatten().numpy()
            corr_comp = float(np.corrcoef(err_flat, unc_flat)[0, 1]) if len(err_flat) > 10 else 0.0

            # Spearman: sample mean uncertainty vs MSE
            mean_ens_std_val = float(ens_std_small[:, missing_ocean].mean())

            sweep_results[t_val].append({
                "val_idx": vi,
                "is_eddy": vi in eddy_set,
                "gp_mse": gp_mse,
                "gpcnn_mse": gpcnn_mse,
                "ens_mse": ens_mse,
                "comp_mse": comp_mse,
                "corr_comp": corr_comp,
                "mean_ens_std": mean_ens_std_val,
            })

        elapsed = time.time() - t0
        tag = "EDDY" if vi in eddy_set else "clean"
        # Quick per-timestep summary for this sample
        mse_strs = [f"t{t}={sweep_results[t][-1]['comp_mse']:.6f}" for t in TIMESTEPS]
        print(f"  [{run_i+1:3d}/{n_total}] vi={vi:5d} {tag:>5}  "
              f"GP={gp_mse:.6f}  CNN={gpcnn_mse:.6f}  {' '.join(mse_strs[:3])}...  ({elapsed:.1f}s)")

    elapsed_total = time.time() - t0_global
    print(f"\nDone: {n_total} samples × {len(TIMESTEPS)} timesteps in {elapsed_total:.1f}s")

    # ==================================================================
    # Aggregate
    # ==================================================================
    from scipy.stats import spearmanr

    print(f"\n{'=' * 100}")
    print(f"{'TIMESTEP SWEEP RESULTS':^100}")
    print(f"{'=' * 100}")

    mean_gp_mse = np.mean([sweep_results[TIMESTEPS[0]][i]["gp_mse"] for i in range(n_total)])
    mean_cnn_mse = np.mean([sweep_results[TIMESTEPS[0]][i]["gpcnn_mse"] for i in range(n_total)])

    print(f"\n  GP baseline:  MSE = {mean_gp_mse:.6f}  (1.000x)")
    print(f"  GP-CNN alone: MSE = {mean_cnn_mse:.6f}  ({mean_cnn_mse/mean_gp_mse:.3f}x)")

    print(f"\n  {'Timestep':>8}  {'Ens Mean MSE':>13} {'Ratio':>8}  "
          f"{'Composite MSE':>14} {'Ratio':>8}  "
          f"{'Ens beats CNN':>14}  {'Comp beats CNN':>15}  "
          f"{'Mean Ens Std':>13}  {'Spearman ρ':>11}")
    print(f"  {'-' * 112}")

    best_t_ens = None
    best_mse_ens = float('inf')
    best_t_comp = None
    best_mse_comp = float('inf')

    for t_val in TIMESTEPS:
        recs = sweep_results[t_val]
        ens_mses = [r["ens_mse"] for r in recs]
        comp_mses = [r["comp_mse"] for r in recs]
        cnn_mses = [r["gpcnn_mse"] for r in recs]

        m_ens = np.mean(ens_mses)
        m_comp = np.mean(comp_mses)

        wins_ens = sum(1 for e, c in zip(ens_mses, cnn_mses) if e < c)
        wins_comp = sum(1 for co, c in zip(comp_mses, cnn_mses) if co < c)

        mean_std = np.mean([r["mean_ens_std"] for r in recs])

        # Sample-level Spearman: mean ensemble std vs composite MSE
        stds = [r["mean_ens_std"] for r in recs]
        mses_s = comp_mses
        rho, _ = spearmanr(stds, mses_s)

        print(f"  t={t_val:>5}  {m_ens:>13.6f} {m_ens/mean_gp_mse:>7.3f}x  "
              f"{m_comp:>14.6f} {m_comp/mean_gp_mse:>7.3f}x  "
              f"{wins_ens:>10}/{n_total}     {wins_comp:>10}/{n_total}    "
              f"{mean_std:>13.6f}  {rho:>11.4f}")

        if m_ens < best_mse_ens:
            best_mse_ens = m_ens
            best_t_ens = t_val
        if m_comp < best_mse_comp:
            best_mse_comp = m_comp
            best_t_comp = t_val

    print(f"\n  Best raw ensemble:        t={best_t_ens}, MSE={best_mse_ens:.6f} ({best_mse_ens/mean_gp_mse:.3f}x)")
    print(f"  Best variance-composite:  t={best_t_comp}, MSE={best_mse_comp:.6f} ({best_mse_comp/mean_gp_mse:.3f}x)")
    print(f"  GP-CNN alone:             MSE={mean_cnn_mse:.6f} ({mean_cnn_mse/mean_gp_mse:.3f}x)")

    improvement_ens = (mean_cnn_mse - best_mse_ens) / mean_cnn_mse * 100
    improvement_comp = (mean_cnn_mse - best_mse_comp) / mean_cnn_mse * 100
    print(f"\n  Best ensemble vs GP-CNN:   {improvement_ens:+.2f}% MSE change")
    print(f"  Best composite vs GP-CNN:  {improvement_comp:+.2f}% MSE change")

    # Eddy vs non-eddy breakdown for best composite timestep
    recs_best = sweep_results[best_t_comp]
    eddy_comp = [r["comp_mse"] for r in recs_best if r["is_eddy"]]
    clean_comp = [r["comp_mse"] for r in recs_best if not r["is_eddy"]]
    eddy_cnn = [r["gpcnn_mse"] for r in recs_best if r["is_eddy"]]
    clean_cnn = [r["gpcnn_mse"] for r in recs_best if not r["is_eddy"]]

    print(f"\n  Eddy/Non-eddy breakdown (best composite t={best_t_comp}):")
    print(f"    Eddy:     CNN={np.mean(eddy_cnn):.6f}  Comp={np.mean(eddy_comp):.6f}  "
          f"({(np.mean(eddy_cnn)-np.mean(eddy_comp))/np.mean(eddy_cnn)*100:+.2f}%)")
    print(f"    Non-eddy: CNN={np.mean(clean_cnn):.6f}  Comp={np.mean(clean_comp):.6f}  "
          f"({(np.mean(clean_cnn)-np.mean(clean_comp))/np.mean(clean_cnn)*100:+.2f}%)")

    # Uncertainty quality at best composite timestep
    print(f"\n  Uncertainty quality at t={best_t_comp}:")
    corrs = [r["corr_comp"] for r in recs_best]
    print(f"    Mean pixel correlation (ens_std ↔ |comp_error|): r = {np.mean(corrs):.4f}")
    stds = [r["mean_ens_std"] for r in recs_best]
    mses_s = [r["comp_mse"] for r in recs_best]
    rho, p = spearmanr(stds, mses_s)
    print(f"    Sample-level Spearman (mean_ens_std ↔ comp_MSE): ρ = {rho:.4f}, p = {p:.2e}")

    # Save
    save_data = {
        "sweep_results": sweep_results,
        "config": {
            "timesteps": TIMESTEPS,
            "n_samples": n_total,
            "n_ensemble": args.n_ensemble,
        },
        "summary": {
            "gp_mean_mse": float(mean_gp_mse),
            "gpcnn_mean_mse": float(mean_cnn_mse),
            "best_t_ens": best_t_ens,
            "best_mse_ens": float(best_mse_ens),
            "best_t_comp": best_t_comp,
            "best_mse_comp": float(best_mse_comp),
        },
    }
    save_path = OUT_DIR / f"sweep_ens{args.n_ensemble}.pt"
    torch.save(save_data, save_path)
    print(f"\nSaved to {save_path}")


if __name__ == "__main__":
    main()
