#!/usr/bin/env python3
"""
Multi-step DDPM reverse sweep for GP-CNN → DDPM refinement.

Compares:
  A) Single-step x₀ prediction (current best: t≈150, composited)
  B) Multi-step reverse: full DDPM chain from t→0 (cond-only, no paste-back)
  C) Multi-step reverse + variance-weighted compositing

Sweeps over t ∈ {10, 25, 50, 75, 100} for multi-step (low t are cheap,
high t are expensive: t steps × n_ensemble).

Usage:
    PYTHONPATH=. python scripts/multistep_sweep.py [--n-samples 100] [--n-ensemble 10]
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
# Constants (shared with timestep_sweep.py)
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
OUT_DIR = Path("results/multistep_sweep")

# Multi-step timesteps (low t to keep cost reasonable)
MULTI_TIMESTEPS = [10, 25, 50, 75, 100]
# Single-step reference at the best composite timestep from prior sweep
SINGLE_REF_T = 150


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
# DDPM inference: single-step x₀ and multi-step reverse
# ---------------------------------------------------------------------------

def ddpm_single_step(ddpm, cond_std_field, missing_mask_1ch, t_val, seed, device):
    """Single-step: noise at t, predict x₀ in one shot."""
    torch.manual_seed(seed)
    alpha_bar = ddpm.alpha_bars[t_val].to(device)
    noise = torch.randn_like(cond_std_field)
    noisy = alpha_bar.sqrt() * cond_std_field + (1 - alpha_bar).sqrt() * noise
    x_cond = torch.cat([noisy, missing_mask_1ch, cond_std_field], dim=1)
    time_tensor = torch.full((1, 1), t_val, device=device, dtype=torch.long)
    with torch.no_grad():
        return ddpm.network(x_cond, time_tensor)


def ddpm_multi_step(ddpm, cond_std_field, missing_mask_1ch, t_start, seed, device):
    """
    Multi-step DDPM reverse from t_start → 0 (conditioning-only, no paste-back).

    At each step t:
      1. Build [x_t, mask, conditioning] → predict x̂₀
      2. DDPM posterior: q(x_{t-1} | x_t, x̂₀) → sample x_{t-1}
      3. At t=0: return x̂₀ directly

    The conditioning channels stay fixed (GP-CNN output) throughout.
    """
    torch.manual_seed(seed)

    # Forward diffuse conditioning field to t_start
    alpha_bar_start = ddpm.alpha_bars[t_start].to(device)
    noise_init = torch.randn_like(cond_std_field)
    x = alpha_bar_start.sqrt() * cond_std_field + (1 - alpha_bar_start).sqrt() * noise_init

    with torch.no_grad():
        for t in range(t_start, -1, -1):
            alpha_t = ddpm.alphas[t].to(device)
            alpha_bar_t = ddpm.alpha_bars[t].to(device)
            beta_t = ddpm.betas[t].to(device)

            time_tensor = torch.full((1, 1), t, device=device, dtype=torch.long)

            # 5-channel input: [x_t, mask, conditioning]
            x_cond = torch.cat([x, missing_mask_1ch, cond_std_field], dim=1)

            # Network predicts x̂₀
            x0_pred = ddpm.network(x_cond, time_tensor)

            if t > 0:
                alpha_bar_prev = ddpm.alpha_bars[t - 1].to(device)

                # Posterior mean
                coeff_x0 = (alpha_bar_prev.sqrt() * beta_t) / (1 - alpha_bar_t)
                coeff_xt = (alpha_t.sqrt() * (1 - alpha_bar_prev)) / (1 - alpha_bar_t)
                mu = coeff_x0 * x0_pred + coeff_xt * x

                # Posterior variance
                beta_tilde = ((1 - alpha_bar_prev) / (1 - alpha_bar_t)) * beta_t
                sigma_t = beta_tilde.sqrt()

                z = torch.randn_like(x)
                x = mu + sigma_t * z
            else:
                x = x0_pred

    return x


def ensemble_single_step(ddpm, cond_std_field, missing_mask_1ch, t_val, n_ens, device):
    """Ensemble of single-step x₀ predictions."""
    preds = []
    for k in range(n_ens):
        pred = ddpm_single_step(ddpm, cond_std_field, missing_mask_1ch, t_val,
                                seed=42 + k * 1000, device=device)
        preds.append(pred)
    stack = torch.stack(preds, dim=0)
    return stack.mean(dim=0), stack.std(dim=0)


def ensemble_multi_step(ddpm, cond_std_field, missing_mask_1ch, t_start, n_ens, device):
    """Ensemble of full multi-step reverse chains."""
    preds = []
    for k in range(n_ens):
        pred = ddpm_multi_step(ddpm, cond_std_field, missing_mask_1ch, t_start,
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
    """w ∈ [0,1] from GP posterior variance: 0=keep CNN, 1=use ensemble."""
    ocean_bool = ocean_mask.astype(bool)
    gp_std = np.sqrt(np.clip(gp_var, 0, None))
    w = np.zeros_like(gp_std)
    for c in range(2):
        vals = gp_std[c][ocean_bool]
        vmin, vmax = vals.min(), vals.max()
        if vmax > vmin:
            w[c] = (gp_std[c] - vmin) / (vmax - vmin)
        w[c] *= ocean_mask
    w_full = np.zeros((1, 2, FULL_H, FULL_W), dtype=np.float32)
    w_full[0, :, :OCEAN_H, :OCEAN_W] = w
    return torch.from_numpy(w_full)


def composite(cnn_phys, ens_phys, weight):
    return (1.0 - weight) * cnn_phys + weight * ens_phys


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--n-ensemble", type=int, default=10)
    parser.add_argument("--multi-timesteps", nargs="+", type=int, default=None,
                        help="Override multi-step timesteps (default: 10 25 50 75 100)")
    args = parser.parse_args()

    multi_ts = args.multi_timesteps or MULTI_TIMESTEPS
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

    _, val_vel = load_pickle_data()

    row22_mask = np.zeros((OCEAN_H, OCEAN_W), dtype=np.float32)
    row22_mask[OBS_ROW, :] = 1.0
    row22_mask *= ocean_mask

    from ddpm.helper_functions.masks.border_mask import BorderMaskGenerator
    border_gen = BorderMaskGenerator()

    ddpm_data = torch.load(DDPM_EVAL_PT, map_location="cpu", weights_only=False)
    ddpm_samples = ddpm_data["samples"]
    val_indices = [s["val_idx"] for s in ddpm_samples]
    eddy_set = set(ddpm_data.get("eddy_indices", []))
    n_total = min(args.n_samples, len(val_indices))
    val_indices = val_indices[:n_total]

    # All methods to test
    all_methods = (
        [("single", SINGLE_REF_T)] +
        [("multi", t) for t in multi_ts]
    )

    print(f"\nMulti-step sweep:")
    print(f"  Single-step reference:  t={SINGLE_REF_T}")
    print(f"  Multi-step timesteps:   {multi_ts}")
    print(f"  Ensemble N={args.n_ensemble}, samples={n_total}")
    print(f"  Total method variants:  {len(all_methods)}")

    # Estimate cost: multi-step t=100 costs 100× compared to single-step
    est_calls = n_total * args.n_ensemble * (1 + sum(multi_ts))
    print(f"  Estimated network calls: ~{est_calls:,}")
    print(f"{'=' * 90}")

    # Storage: method_key -> list of per-sample dicts
    results = {f"{mode}_t{t}": [] for mode, t in all_methods}

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

        # GP & CNN baseline MSEs
        gp_mse = (torch.from_numpy(gp_mean) - gt_small_t)[:, missing_ocean].pow(2).mean().item()
        gpcnn_mse = (gpcnn_phys_cpu[0, :, :OCEAN_H, :OCEAN_W] - gt_small_t)[:, missing_ocean].pow(2).mean().item()

        gpcnn_ddpm_std_dev = gpcnn_ddpm_std.to(device)

        # ── Run all methods ──
        for mode, t_val in all_methods:
            key = f"{mode}_t{t_val}"

            if mode == "single":
                ens_mean_std, ens_std_std = ensemble_single_step(
                    ddpm, gpcnn_ddpm_std_dev, missing_mask_1ch_dev,
                    t_val, args.n_ensemble, device
                )
            else:
                ens_mean_std, ens_std_std = ensemble_multi_step(
                    ddpm, gpcnn_ddpm_std_dev, missing_mask_1ch_dev,
                    t_val, args.n_ensemble, device
                )

            ens_mean_phys = unstd_ddpm(ens_mean_std.cpu())
            comp_phys = composite(gpcnn_phys_cpu, ens_mean_phys, w)

            ens_mse = (ens_mean_phys[0, :, :OCEAN_H, :OCEAN_W] - gt_small_t)[:, missing_ocean].pow(2).mean().item()
            comp_mse = (comp_phys[0, :, :OCEAN_H, :OCEAN_W] - gt_small_t)[:, missing_ocean].pow(2).mean().item()

            # Ensemble std for uncertainty correlation
            ens_std_phys = torch.zeros_like(ens_std_std.cpu())
            ens_std_phys[:, 0:1] = ens_std_std.cpu()[:, 0:1] * U_STD
            ens_std_phys[:, 1:2] = ens_std_std.cpu()[:, 1:2] * V_STD
            ens_std_small = ens_std_phys[0, :, :OCEAN_H, :OCEAN_W]
            mean_ens_std_val = float(ens_std_small[:, missing_ocean].mean())

            results[key].append({
                "val_idx": vi,
                "is_eddy": vi in eddy_set,
                "gp_mse": gp_mse,
                "gpcnn_mse": gpcnn_mse,
                "ens_mse": ens_mse,
                "comp_mse": comp_mse,
                "mean_ens_std": mean_ens_std_val,
            })

        elapsed = time.time() - t0
        tag = "EDDY" if vi in eddy_set else "clean"
        # Show key MSEs
        single_comp = results[f"single_t{SINGLE_REF_T}"][-1]["comp_mse"]
        multi_comps = " ".join(
            f"m{t}={results[f'multi_t{t}'][-1]['comp_mse']:.6f}"
            for t in multi_ts[:3]
        )
        print(f"  [{run_i+1:3d}/{n_total}] vi={vi:5d} {tag:>5}  "
              f"CNN={gpcnn_mse:.6f}  s{SINGLE_REF_T}={single_comp:.6f}  "
              f"{multi_comps}  ({elapsed:.1f}s)")

    elapsed_total = time.time() - t0_global
    print(f"\nDone: {n_total} samples × {len(all_methods)} methods in {elapsed_total:.1f}s")

    # ==================================================================
    # Aggregate
    # ==================================================================
    print(f"\n{'=' * 110}")
    print(f"{'MULTI-STEP vs SINGLE-STEP SWEEP RESULTS':^110}")
    print(f"{'=' * 110}")

    mean_gp_mse = np.mean([results[f"single_t{SINGLE_REF_T}"][i]["gp_mse"] for i in range(n_total)])
    mean_cnn_mse = np.mean([results[f"single_t{SINGLE_REF_T}"][i]["gpcnn_mse"] for i in range(n_total)])

    print(f"\n  GP baseline:  MSE = {mean_gp_mse:.6f}  (1.000x)")
    print(f"  GP-CNN alone: MSE = {mean_cnn_mse:.6f}  ({mean_cnn_mse/mean_gp_mse:.3f}x)")

    print(f"\n  {'Method':>15}  {'Raw Ens MSE':>12} {'Ens/GP':>7}  "
          f"{'Comp MSE':>12} {'Comp/GP':>8}  "
          f"{'Comp<CNN':>9}  {'Mean Std':>10}  {'Steps':>6}")
    print(f"  {'-' * 95}")

    for mode, t_val in all_methods:
        key = f"{mode}_t{t_val}"
        recs = results[key]
        ens_mses = [r["ens_mse"] for r in recs]
        comp_mses = [r["comp_mse"] for r in recs]
        cnn_mses = [r["gpcnn_mse"] for r in recs]

        m_ens = np.mean(ens_mses)
        m_comp = np.mean(comp_mses)
        wins = sum(1 for co, c in zip(comp_mses, cnn_mses) if co < c)
        mean_std = np.mean([r["mean_ens_std"] for r in recs])
        n_steps = 1 if mode == "single" else t_val + 1

        label = f"{'1-step' if mode == 'single' else 'multi'} t={t_val}"
        print(f"  {label:>15}  {m_ens:>12.6f} {m_ens/mean_gp_mse:>6.3f}x  "
              f"{m_comp:>12.6f} {m_comp/mean_gp_mse:>7.3f}x  "
              f"{wins:>6}/{n_total}  {mean_std:>10.6f}  {n_steps:>6}")

    # Eddy vs non-eddy breakdown
    print(f"\n  Eddy/Non-eddy breakdown (composite MSE):")
    print(f"  {'Method':>15}  {'Eddy Comp':>12}  {'NE Comp':>12}  "
          f"{'E wins':>7}  {'NE wins':>8}")
    print(f"  {'-' * 65}")

    for mode, t_val in all_methods:
        key = f"{mode}_t{t_val}"
        recs = results[key]
        eddy = [r for r in recs if r["is_eddy"]]
        non_eddy = [r for r in recs if not r["is_eddy"]]
        e_comp = np.mean([r["comp_mse"] for r in eddy]) if eddy else 0
        ne_comp = np.mean([r["comp_mse"] for r in non_eddy]) if non_eddy else 0
        e_wins = sum(1 for r in eddy if r["comp_mse"] < r["gpcnn_mse"])
        ne_wins = sum(1 for r in non_eddy if r["comp_mse"] < r["gpcnn_mse"])
        n_e = len(eddy)
        n_ne = len(non_eddy)

        label = f"{'1-step' if mode == 'single' else 'multi'} t={t_val}"
        print(f"  {label:>15}  {e_comp:>12.6f}  {ne_comp:>12.6f}  "
              f"{e_wins:>4}/{n_e}  {ne_wins:>5}/{n_ne}")

    # Save
    save_data = {
        "results": results,
        "config": {
            "single_ref_t": SINGLE_REF_T,
            "multi_timesteps": multi_ts,
            "n_samples": n_total,
            "n_ensemble": args.n_ensemble,
        },
        "summary": {
            "gp_mean_mse": float(mean_gp_mse),
            "gpcnn_mean_mse": float(mean_cnn_mse),
        },
    }
    save_path = OUT_DIR / f"multistep_sweep_ens{args.n_ensemble}.pt"
    torch.save(save_data, save_path)
    print(f"\nSaved to {save_path}")


if __name__ == "__main__":
    main()
