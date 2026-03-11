#!/usr/bin/env python3
"""Compare 3 stage-3 models: baseline, maskxt_hightmax, selfcond.

Runs matched 3-stage inference for each model and reports MSE vs V-CNN.
Self-conditioning model gets two-pass inference (first pass → self_cond).

Usage:
    PYTHONPATH=. python3 tmp_compare_3models.py
    PYTHONPATH=. python3 tmp_compare_3models.py --n-samples 10 --coverage 0.5
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

BASE_DIR = Path(__file__).resolve().parent

from ddpm.neural_networks.ddpm import GaussianDDPM
from ddpm.neural_networks.unets.unet_xl_attn import MyUNet_Attn
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

MODEL_CONFIGS = {
    "baseline": {
        "fracs": [0.30, 0.20, 0.20],
        "weights": "experiments/11_bootstrap_rollout/voronoi_detached_stage3/results/inpaint_gaussian_t250_best_ema_weights.pt",
        "self_cond": False,
    },
    "maskxt": {
        "fracs": [0.50, 0.40, 0.30],
        "weights": "experiments/11_bootstrap_rollout/voronoi_stage3_maskxt_hightmax/results/inpaint_gaussian_t250_best_ema_weights.pt",
        "self_cond": False,
    },
    "selfcond": {
        "fracs": [0.50, 0.40, 0.30],
        "weights": "experiments/11_bootstrap_rollout/voronoi_stage3_selfcond/results/inpaint_gaussian_t250_best_ema_weights.pt",
        "self_cond": True,
    },
}
NUM_STAGES = 3
VCNN_CKPT = "results/voronoi_cnn/voronoi_cnn_best.pt"

# ── Args ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--coverage", type=float, default=0.5,
                    help="Percent of ocean cells observed (0.5 = 0.5%%)")
parser.add_argument("--n-samples", type=int, default=5)
parser.add_argument("--resample-steps", type=int, default=3,
                    help="RePaint resample iterations per timestep")
parser.add_argument("--seed", type=int, default=42)
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
def load_model(cfg):
    self_cond_ch = 2 if cfg["self_cond"] else 0
    net = MyUNet_Attn(n_steps=N_STEPS, time_emb_dim=256,
                      n_stage_tokens=3, self_cond_channels=self_cond_ch)
    ddpm = GaussianDDPM(net, n_steps=N_STEPS,
                        min_beta=0.0001, max_beta=0.02, device=device)
    state = torch.load(str(BASE_DIR / cfg["weights"]), map_location="cpu",
                       weights_only=False)
    ddpm.load_state_dict(state)
    ddpm.to(device)
    ddpm.eval()
    return ddpm


# ── Reverse diffusion ───────────────────────────────────────────────
def reverse_diffuse_stage(ddpm, x_init, t_start, stage_idx,
                          known_std, known_mask, miss_mask,
                          resample_steps=3, use_self_cond=False):
    """Full DDPM reverse chain from t_start to 0 for one stage."""
    stage_tensor = torch.tensor([stage_idx], device=device, dtype=torch.long)

    alpha_bar_t = ddpm.alpha_bars[t_start]
    eps = torch.randn_like(x_init)
    x = alpha_bar_t.sqrt() * x_init + (1 - alpha_bar_t).sqrt() * eps

    prev_x0 = None  # for self-conditioning

    for t in range(t_start, -1, -1):
        n_resample = resample_steps if t > 0 else 1

        for r in range(n_resample):
            alpha_t = ddpm.alphas[t]
            alpha_bar_t = ddpm.alpha_bars[t]
            beta_t = ddpm.betas[t]
            time_tensor = torch.full((1, 1), t, device=device, dtype=torch.long)

            # Self-conditioning: pass previous x0 prediction
            sc = prev_x0 if use_self_cond else None
            x0_pred = ddpm.network(x, time_tensor, stage=stage_tensor,
                                   self_cond=sc)
            prev_x0 = x0_pred.detach()

            if t > 0:
                alpha_bar_prev = ddpm.alpha_bars[t - 1]
                coeff_x0 = (alpha_bar_prev.sqrt() * beta_t) / (1 - alpha_bar_t)
                coeff_xt = (alpha_t.sqrt() * (1 - alpha_bar_prev)) / (1 - alpha_bar_t)
                mu = coeff_x0 * x0_pred + coeff_xt * x
                beta_tilde = ((1 - alpha_bar_prev) / (1 - alpha_bar_t)) * beta_t
                sigma_t = beta_tilde.sqrt()
                z = torch.randn_like(x)
                x_denoised = mu + sigma_t * z
            else:
                x_denoised = x0_pred

            if t > 0:
                alpha_bar_prev = ddpm.alpha_bars[t - 1]
                noise_known = torch.randn_like(known_std)
                x_known = alpha_bar_prev.sqrt() * known_std + (1 - alpha_bar_prev).sqrt() * noise_known
                x = x_known * known_mask + x_denoised * miss_mask
            else:
                x = known_std * known_mask + x_denoised * miss_mask

            if r < n_resample - 1 and t > 0:
                noise_back = torch.randn_like(x)
                x = alpha_t.sqrt() * x + (1 - alpha_t).sqrt() * noise_back

    return x


def run_matched_3stage(ddpm, gt_ocean, obs_mask, ocean_mask, seed,
                       stage_t_max, resample_steps=3, use_self_cond=False):
    """Run matched 3-stage inference."""
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
        for stage_idx in range(NUM_STAGES):
            t_start = stage_t_max[stage_idx]
            current = reverse_diffuse_stage(
                ddpm, current, t_start, stage_idx,
                known_std, known_mask, miss_mask,
                resample_steps=resample_steps,
                use_self_cond=use_self_cond,
            )
            current = known_std * known_mask + current * miss_mask

    result_phys = standardizer.unstandardize(current.squeeze(0).cpu())
    return result_phys[:, :OCEAN_H, :OCEAN_W].numpy()


def run_single_step_3stage(ddpm, gt_ocean, obs_mask, ocean_mask, seed,
                           stage_t_max, use_self_cond=False):
    """3-stage single-step x0 prediction (one forward pass per stage)."""
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
        for stage_idx in range(NUM_STAGES):
            t_val = stage_t_max[stage_idx] // 2
            stage_tensor = torch.tensor([stage_idx], device=device, dtype=torch.long)
            time_tensor = torch.full((1, 1), t_val, device=device, dtype=torch.long)

            alpha_bar_t = ddpm.alpha_bars[t_val]
            eps = torch.randn_like(current)
            x_t = alpha_bar_t.sqrt() * current + (1 - alpha_bar_t).sqrt() * eps

            # Self-conditioning: two-pass
            sc = None
            if use_self_cond:
                first = ddpm.network(x_t, time_tensor, stage=stage_tensor,
                                     self_cond=None)
                sc = first.detach()

            x0_pred = ddpm.network(x_t, time_tensor, stage=stage_tensor,
                                   self_cond=sc)
            current = known_std * known_mask + x0_pred * miss_mask

    result_phys = standardizer.unstandardize(current.squeeze(0).cpu())
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

    print(f"\n{'='*72}")
    print(f"3-MODEL COMPARISON — STAGE-3 EVALUATION")
    print(f"Coverage: {coverage}%  |  Samples: {n_samples}  |  "
          f"Resample: {args.resample_steps}")
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

    # Load all models
    print("\nLoading models...")
    models = {}
    for name, cfg in MODEL_CONFIGS.items():
        wpath = BASE_DIR / cfg["weights"]
        if not wpath.exists():
            print(f"  {name}: SKIPPED (weights not found: {cfg['weights']})")
            continue
        models[name] = {"ddpm": load_model(cfg), "cfg": cfg}
        t_max = [int(round(f * (N_STEPS - 1))) for f in cfg["fracs"]]
        print(f"  {name}: loaded (t_max={t_max}, self_cond={cfg['self_cond']})")

    vcnn = load_vcnn(device)
    print("  V-CNN loaded")

    # Methods: Voronoi, V-CNN, then 1-Step + 3-Stage for each model
    method_names = ["Voronoi", "V-CNN"]
    for name in models:
        method_names.append(f"1S-{name}")
        method_names.append(f"3S-{name}")
    results = {m: [] for m in method_names}

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
        vel_obs = gt * obs_mask[None]

        t0 = time.time()

        # Voronoi baseline
        vor = voronoi_fill(vel_obs, obs_mask, ocean_mask)
        results["Voronoi"].append(ocean_mse(vor, gt, ocean_mask))

        # V-CNN
        vcnn_pred = predict_vcnn(vcnn, vel_obs, obs_mask, ocean_mask, device)
        results["V-CNN"].append(ocean_mse(vcnn_pred, gt, ocean_mask))

        # Each DDPM model: 1-step and 3-stage
        for name, info in models.items():
            ddpm = info["ddpm"]
            cfg = info["cfg"]
            t_max = [int(round(f * (N_STEPS - 1))) for f in cfg["fracs"]]
            sc = cfg["self_cond"]

            # Single-step 3-stage
            ss = run_single_step_3stage(
                ddpm, gt, obs_mask, ocean_mask, seed,
                stage_t_max=t_max, use_self_cond=sc,
            )
            results[f"1S-{name}"].append(ocean_mse(ss, gt, ocean_mask))

            # Full 3-stage reverse
            s3 = run_matched_3stage(
                ddpm, gt, obs_mask, ocean_mask, seed,
                stage_t_max=t_max, resample_steps=args.resample_steps,
                use_self_cond=sc,
            )
            results[f"3S-{name}"].append(ocean_mse(s3, gt, ocean_mask))

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

    # Win rates: each model's 3-stage vs V-CNN
    if len(val_indices) >= 2:
        print(f"\nWin rates vs V-CNN (per sample):")
        vcnn_arr = np.array(results["V-CNN"])
        for name in models:
            s3 = np.array(results[f"3S-{name}"])
            n = len(s3)
            wins = (s3 < vcnn_arr).sum()
            print(f"  3S-{name} beats V-CNN: {wins}/{n}")


if __name__ == "__main__":
    main()
