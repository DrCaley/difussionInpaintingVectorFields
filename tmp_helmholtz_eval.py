#!/usr/bin/env python3
"""Evaluate Helmholtz dual-head UNet baseline — single-step x0 prediction.

Sweeps t_val (noise level) and reports MSE vs V-CNN at 0.5% coverage.

Usage:
    PYTHONPATH=. python3 tmp_helmholtz_eval.py
    PYTHONPATH=. python3 tmp_helmholtz_eval.py --n-samples 10 --coverage 0.5
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

BASE_DIR = Path(__file__).resolve().parent

from ddpm.neural_networks.ddpm import GaussianDDPM
from ddpm.neural_networks.unets.unet_helmholtz import MyUNet_Helmholtz
from ddpm.helper_functions.standardize_data import ZScoreStandardizer
from scripts.voronoi_cnn_model import VoronoiCNN, build_voronoi_input

# ── Constants ────────────────────────────────────────────────────────
OCEAN_H, OCEAN_W = 44, 94
FULL_H, FULL_W = 64, 128
N_STEPS = 250

U_MEAN, U_STD = -0.06929559429949586, 0.1358005549716049
V_MEAN, V_STD = -0.0323937796117541, 0.08899177232117582
NM = np.array([U_MEAN, V_MEAN], dtype=np.float32)
NS = np.array([U_STD, V_STD], dtype=np.float32)
standardizer = ZScoreStandardizer(U_MEAN, U_STD, V_MEAN, V_STD)

HELMHOLTZ_WEIGHTS = (
    "experiments/12_helmholtz_dual_head/helmholtz_baseline/results/"
    "inpaint_gaussian_t250_best_weights.pt"
)
VCNN_CKPT = "results/voronoi_cnn/voronoi_cnn_best.pt"

# ── Args ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--coverage", type=float, default=0.5,
                    help="Percent of ocean cells observed (0.5 = 0.5%%)")
parser.add_argument("--n-samples", type=int, default=5)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--t-vals", type=int, nargs="+",
                    default=[25, 50, 75, 100, 125],
                    help="Noise timesteps to sweep for single-step")
args = parser.parse_args()

# ── Device ───────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Device: {device}")


# ── Data loading ─────────────────────────────────────────────────────
def load_val_data():
    import pickle
    with open(str(BASE_DIR / "data.pickle"), "rb") as f:
        _, _, test_raw = pickle.load(f)
    test = np.transpose(test_raw, (3, 2, 1, 0)).astype(np.float32)
    test = np.nan_to_num(test, nan=0.0)
    return torch.from_numpy(test)


def get_ocean_mask(val_tensor):
    return (val_tensor[0].abs().sum(dim=0) > 1e-7).float().numpy()


def make_border_mask(shape, dev):
    m = torch.zeros(1, 1, shape[2], shape[3], device=dev)
    m[:, :, :OCEAN_H, :OCEAN_W] = 1.0
    return m


def random_obs_mask(ocean_mask, pct, rng):
    idx = np.argwhere(ocean_mask > 0.5)
    n = max(1, round(len(idx) * pct / 100.0))
    sel = rng.choice(len(idx), size=n, replace=False)
    m = np.zeros((OCEAN_H, OCEAN_W), dtype=np.float32)
    for i in sel:
        m[idx[i][0], idx[i][1]] = 1.0
    return m


def voronoi_fill(vel_obs, obs_mask, ocean_mask):
    ky, kx = np.where(obs_mask > 0.5)
    if len(ky) == 0:
        return np.zeros_like(vel_obs)
    tree = cKDTree(np.stack([ky, kx], axis=1).astype(np.float64))
    gy, gx = np.mgrid[0:OCEAN_H, 0:OCEAN_W]
    _, idx = tree.query(np.stack([gy.ravel(), gx.ravel()], axis=1).astype(np.float64))
    idx = idx.reshape(OCEAN_H, OCEAN_W)
    filled = np.stack([vel_obs[0, ky, kx][idx], vel_obs[1, ky, kx][idx]], axis=0)
    return filled * ocean_mask


# ── V-CNN ────────────────────────────────────────────────────────────
def load_vcnn(dev):
    ck = torch.load(str(BASE_DIR / VCNN_CKPT), map_location="cpu", weights_only=False)
    m = VoronoiCNN(**ck["model_config"]).to(dev)
    m.load_state_dict(ck["model_state"])
    m.eval()
    return m


def predict_vcnn(model, vel_obs, obs_mask, ocean_mask, dev):
    vel_n = ((vel_obs - NM[:, None, None]) / NS[:, None, None]) * ocean_mask[None]
    vi = build_voronoi_input(vel_n, obs_mask, ocean_mask)
    with torch.no_grad():
        p = model(torch.from_numpy(vi).unsqueeze(0).to(dev))
    mt = torch.tensor(NM).view(1, 2, 1, 1).to(dev)
    st = torch.tensor(NS).view(1, 2, 1, 1).to(dev)
    om = torch.from_numpy(ocean_mask).float().to(dev).view(1, 1, OCEAN_H, OCEAN_W)
    phys = (p * st + mt) * om
    return phys.squeeze(0).cpu().numpy()


# ── Model loading ────────────────────────────────────────────────────
def load_helmholtz():
    net = MyUNet_Helmholtz(n_steps=N_STEPS, time_emb_dim=256,
                           n_stage_tokens=0, self_cond_channels=0)
    ddpm = GaussianDDPM(net, n_steps=N_STEPS,
                        min_beta=0.0001, max_beta=0.02, device=device)
    state = torch.load(str(BASE_DIR / HELMHOLTZ_WEIGHTS), map_location="cpu",
                       weights_only=False)
    ddpm.load_state_dict(state)
    ddpm.to(device)
    ddpm.eval()
    return ddpm


# ── Single-step inference ────────────────────────────────────────────
def run_single_step(ddpm, gt_ocean, obs_mask, ocean_mask, seed, t_val):
    """Forward-noise voronoi fill at t_val, one network pass → x0_pred."""
    # Build full-size tensors
    gt_full = np.zeros((1, 2, FULL_H, FULL_W), dtype=np.float32)
    gt_full[0, :, :OCEAN_H, :OCEAN_W] = gt_ocean * ocean_mask[None]
    known_std = standardizer(
        torch.from_numpy(gt_full).squeeze(0)).unsqueeze(0).to(device)

    border = make_border_mask((1, 2, FULL_H, FULL_W), device)
    land_mask = (torch.from_numpy(gt_full).abs() > 1e-5).float().to(device)
    raw_miss = np.ones((FULL_H, FULL_W), dtype=np.float32)
    raw_miss[:OCEAN_H, :OCEAN_W] -= obs_mask
    raw_miss[:OCEAN_H, :OCEAN_W] *= ocean_mask
    raw_miss[OCEAN_H:, :] = 0.0
    raw_miss[:, OCEAN_W:] = 0.0
    miss_mask = (torch.from_numpy(raw_miss).unsqueeze(0).unsqueeze(0).to(device)
                 * border * land_mask)
    known_mask = 1.0 - miss_mask

    vel_obs = gt_ocean * obs_mask[None]
    vor_fill = voronoi_fill(vel_obs, obs_mask, ocean_mask)
    vor_full = np.zeros((1, 2, FULL_H, FULL_W), dtype=np.float32)
    vor_full[0, :, :OCEAN_H, :OCEAN_W] = vor_fill
    vor_std = standardizer(
        torch.from_numpy(vor_full).squeeze(0)).unsqueeze(0).to(device)
    current = known_std * known_mask + vor_std * miss_mask

    torch.manual_seed(seed)
    with torch.no_grad():
        time_tensor = torch.full((1, 1), t_val, device=device, dtype=torch.long)
        alpha_bar_t = ddpm.alpha_bars[t_val]
        eps = torch.randn_like(current)
        x_t = alpha_bar_t.sqrt() * current + (1 - alpha_bar_t).sqrt() * eps
        x0_pred = ddpm.network(x_t, time_tensor)
        result = known_std * known_mask + x0_pred * miss_mask

    result_phys = standardizer.unstandardize(result.squeeze(0).cpu())
    return result_phys[:, :OCEAN_H, :OCEAN_W].numpy()


# ── Reverse chain inference ──────────────────────────────────────────
def run_reverse_chain(ddpm, gt_ocean, obs_mask, ocean_mask, seed, t_start,
                      resample_steps=3):
    """Full DDPM reverse chain from t_start to 0."""
    gt_full = np.zeros((1, 2, FULL_H, FULL_W), dtype=np.float32)
    gt_full[0, :, :OCEAN_H, :OCEAN_W] = gt_ocean * ocean_mask[None]
    known_std = standardizer(
        torch.from_numpy(gt_full).squeeze(0)).unsqueeze(0).to(device)

    border = make_border_mask((1, 2, FULL_H, FULL_W), device)
    land_mask = (torch.from_numpy(gt_full).abs() > 1e-5).float().to(device)
    raw_miss = np.ones((FULL_H, FULL_W), dtype=np.float32)
    raw_miss[:OCEAN_H, :OCEAN_W] -= obs_mask
    raw_miss[:OCEAN_H, :OCEAN_W] *= ocean_mask
    raw_miss[OCEAN_H:, :] = 0.0
    raw_miss[:, OCEAN_W:] = 0.0
    miss_mask = (torch.from_numpy(raw_miss).unsqueeze(0).unsqueeze(0).to(device)
                 * border * land_mask)
    known_mask = 1.0 - miss_mask

    vel_obs = gt_ocean * obs_mask[None]
    vor_fill = voronoi_fill(vel_obs, obs_mask, ocean_mask)
    vor_full = np.zeros((1, 2, FULL_H, FULL_W), dtype=np.float32)
    vor_full[0, :, :OCEAN_H, :OCEAN_W] = vor_fill
    vor_std = standardizer(
        torch.from_numpy(vor_full).squeeze(0)).unsqueeze(0).to(device)
    current = known_std * known_mask + vor_std * miss_mask

    torch.manual_seed(seed)
    with torch.no_grad():
        # Forward noise from voronoi fill
        alpha_bar_start = ddpm.alpha_bars[t_start]
        eps = torch.randn_like(current)
        x = alpha_bar_start.sqrt() * current + (1 - alpha_bar_start).sqrt() * eps

        for t in range(t_start, -1, -1):
            n_resample = resample_steps if t > 0 else 1
            for r in range(n_resample):
                time_tensor = torch.full((1, 1), t, device=device, dtype=torch.long)

                # x0 prediction model
                x0_pred = ddpm.network(x, time_tensor)

                if t > 0:
                    alpha_bar_t = ddpm.alpha_bars[t]
                    alpha_bar_prev = ddpm.alpha_bars[t - 1]
                    alpha_t = ddpm.alphas[t]
                    beta_t = ddpm.betas[t]

                    # Posterior mean: reconstruct x_{t-1} from x0_pred
                    coef1 = (alpha_bar_prev.sqrt() * beta_t) / (1 - alpha_bar_t)
                    coef2 = (alpha_t.sqrt() * (1 - alpha_bar_prev)) / (1 - alpha_bar_t)
                    mean = coef1 * x0_pred + coef2 * x
                    var = (beta_t * (1 - alpha_bar_prev) / (1 - alpha_bar_t))
                    noise = torch.randn_like(x)
                    x_denoised = mean + var.sqrt() * noise
                else:
                    x_denoised = x0_pred

                # Paste known values
                x = known_std * known_mask + x_denoised * miss_mask

                # RePaint re-noise for resample
                if r < n_resample - 1 and t > 0:
                    noise_back = torch.randn_like(x)
                    x = alpha_t.sqrt() * x + (1 - alpha_t).sqrt() * noise_back

    result_phys = standardizer.unstandardize(x.squeeze(0).cpu())
    return result_phys[:, :OCEAN_H, :OCEAN_W].numpy()


# ── Metrics ──────────────────────────────────────────────────────────
def ocean_mse(pred, gt, ocean_mask):
    ocean_b = ocean_mask.astype(bool)
    diff = pred[:, ocean_b] - gt[:, ocean_b]
    return float((diff ** 2).mean())


# ══════════════════════════════════════════════════════════════════════
def main():
    coverage = args.coverage
    n_samples = args.n_samples
    t_vals = args.t_vals

    print(f"\n{'='*72}")
    print(f"HELMHOLTZ DUAL-HEAD BASELINE — EVALUATION")
    print(f"Coverage: {coverage}%  |  Samples: {n_samples}  |  "
          f"t_vals: {t_vals}")
    print(f"{'='*72}")

    val = load_val_data()
    ocean_mask = get_ocean_mask(val)
    n_ocean = int(ocean_mask.sum())
    n_obs = max(1, round(n_ocean * coverage / 100.0))
    print(f"  Ocean cells: {n_ocean}, Observations: {n_obs}")

    rng = np.random.default_rng(seed=7777)
    val_indices = rng.choice(val.shape[0], size=min(n_samples, val.shape[0]),
                             replace=False)
    val_indices.sort()

    # Load models
    print("\nLoading models...")
    ddpm = load_helmholtz()
    print(f"  Helmholtz UNet loaded from {HELMHOLTZ_WEIGHTS}")
    vcnn = load_vcnn(device)
    print("  V-CNN loaded")

    # Method names: Voronoi, V-CNN, then 1S@t for each t_val, then reverse chain
    method_names = ["Voronoi", "V-CNN"]
    for tv in t_vals:
        method_names.append(f"1S@t={tv}")
    method_names.append("RevChain")
    results = {m: [] for m in method_names}

    # Use best t_val for reverse chain start
    rev_t_start = 75  # moderate noise level

    print(f"\n{'#':>3} {'Idx':>5}", end="")
    for m in method_names:
        print(f" {m:>13}", end="")
    print(f" {'Time':>7}")
    print("-" * (10 + 14 * len(method_names) + 8))

    t0_global = time.time()
    for i, vi in enumerate(val_indices):
        gt = val[vi, :, :OCEAN_H, :OCEAN_W].numpy()
        seed = args.seed + vi
        obs_mask = random_obs_mask(ocean_mask, coverage,
                                   np.random.default_rng(seed=seed))

        t0 = time.time()

        # Voronoi baseline
        vel_obs = gt * obs_mask[None]
        vor = voronoi_fill(vel_obs, obs_mask, ocean_mask)
        results["Voronoi"].append(ocean_mse(vor, gt, ocean_mask))

        # V-CNN
        vcnn_pred = predict_vcnn(vcnn, vel_obs, obs_mask, ocean_mask, device)
        results["V-CNN"].append(ocean_mse(vcnn_pred, gt, ocean_mask))

        # Single-step at each t_val
        for tv in t_vals:
            ss = run_single_step(ddpm, gt, obs_mask, ocean_mask, seed, tv)
            results[f"1S@t={tv}"].append(ocean_mse(ss, gt, ocean_mask))

        # Reverse chain
        rc = run_reverse_chain(ddpm, gt, obs_mask, ocean_mask, seed,
                               t_start=rev_t_start, resample_steps=3)
        results["RevChain"].append(ocean_mse(rc, gt, ocean_mask))

        elapsed = time.time() - t0
        row = f"{i+1:>3} {vi:>5}"
        for m in method_names:
            row += f" {results[m][-1]:>13.6f}"
        row += f" {elapsed:>6.1f}s"
        print(row)

    total = time.time() - t0_global

    # Summary
    print(f"\n{'='*72}")
    print(f"SUMMARY — {coverage}% coverage, {len(val_indices)} samples, "
          f"{total:.0f}s total")
    print(f"{'='*72}")

    vcnn_mean = np.mean(results["V-CNN"])
    print(f"\n{'Method':<20} {'Mean MSE':<14} {'vs V-CNN':<10}")
    print("-" * 45)
    for m in method_names:
        arr = np.array(results[m])
        ratio = arr.mean() / vcnn_mean if vcnn_mean > 0 else float("inf")
        print(f"{m:<20} {arr.mean():.7f}     {ratio:.3f}x")

    # Identify best t_val
    best_t = None
    best_ratio = float("inf")
    for tv in t_vals:
        m = f"1S@t={tv}"
        ratio = np.mean(results[m]) / vcnn_mean
        if ratio < best_ratio:
            best_ratio = ratio
            best_t = tv
    print(f"\nBest single-step: t={best_t} ({best_ratio:.3f}x V-CNN)")

    # Divergence diagnostic
    print(f"\n── Divergence diagnostic (Helmholtz decomposition) ──")
    net = ddpm.network
    if hasattr(net, 'last_psi') and net.last_psi is not None:
        psi_rms = float(net.last_psi.pow(2).mean().sqrt())
        phi_rms = float(net.last_phi.pow(2).mean().sqrt())
        ratio = phi_rms / (psi_rms + 1e-12)
        print(f"  ψ RMS: {psi_rms:.6f}  |  φ RMS: {phi_rms:.6f}  |  φ/ψ ratio: {ratio:.4f}")
    else:
        print("  (no forward pass diagnostics available)")


if __name__ == "__main__":
    main()
