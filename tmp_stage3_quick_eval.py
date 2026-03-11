#!/usr/bin/env python3
"""Quick evaluation of the stage-3 detached rollout model.

Lean script: loads only ONE DDPM + V-CNN, runs 3 samples, ~5 min total on MPS.
Compares: GP, Voronoi, V-CNN, Vor-Diff (stage3 x0).

Usage:
    PYTHONPATH=. python3 tmp_stage3_quick_eval.py
    PYTHONPATH=. python3 tmp_stage3_quick_eval.py --n-samples 5
"""

import argparse, os, pickle, sys, time
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from scripts.voronoi_cnn_model import VoronoiCNN, build_voronoi_input
from ddpm.neural_networks.ddpm import GaussianDDPM
from ddpm.neural_networks.unets.unet_xl_attn import MyUNet_Attn
from ddpm.helper_functions.standardize_data import ZScoreStandardizer
from ddpm.helper_functions.interpolation_tool import gp_fill
from ddpm.helper_functions.masks.border_mask import BorderMaskGenerator
from ddpm.utils.inpainting_utils import repaint_gp_init_adaptive
from ddpm.utils.noise_utils import get_noise_strategy

# ── constants ────────────────────────────────────────────────────────
OCEAN_H, OCEAN_W = 44, 94
FULL_H, FULL_W = 64, 128
N_STEPS = 250

U_MEAN, U_STD = -0.06929559429949586, 0.1358005549716049
V_MEAN, V_STD = -0.0323937796117541, 0.08899177232117582
NM = np.array([U_MEAN, V_MEAN], dtype=np.float32)
NS = np.array([U_STD, V_STD], dtype=np.float32)
standardizer = ZScoreStandardizer(U_MEAN, U_STD, V_MEAN, V_STD)

GP_PARAMS = dict(lengthscale=14.1, variance=0.0103420345, noise=1e-8,
                 kernel_type="rbf_legacy", coord_system="pixels")

S6_DEFAULTS = dict(
    max_stages=6, t_start=75, t_refine=50, resample_steps=5,
    noise_floor=0.2, noise_floor_refine=0.3, var_decay=0.1, gamma=3.0, seed=42,
)

STAGE3_WEIGHTS = (
    "experiments/11_bootstrap_rollout/voronoi_detached_stage3/results/"
    "inpaint_gaussian_t250_best_ema_weights.pt"
)
VCNN_CKPT = "results/voronoi_cnn/voronoi_cnn_best.pt"

parser = argparse.ArgumentParser()
parser.add_argument("--coverage", type=float, default=0.5)
parser.add_argument("--n-samples", type=int, default=3)
args = parser.parse_args()

# ── device ───────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Device: {device}")

# ── helpers ──────────────────────────────────────────────────────────
noise_strategy = get_noise_strategy("gaussian")
border_gen = BorderMaskGenerator()


def load_val_data():
    """Load validation data from gp_precomputed.pt (avoids 800MB data.pickle)."""
    gp = torch.load(str(BASE_DIR / "data/rams_head/gp_precomputed.pt"),
                    map_location="cpu", weights_only=False)
    # gp_test: (N, 2, 64, 128) physical-space velocity (GP-filled ground truth)
    return torch.nan_to_num(gp["gp_test"].float(), nan=0.0)


def get_ocean_mask(val_tensor):
    """Ocean mask in 44x94 subdomain."""
    return (val_tensor[0, :, :OCEAN_H, :OCEAN_W].abs() > 1e-5).any(dim=0).float().numpy()


def random_mask(ocean_mask, pct, rng):
    idx = np.argwhere(ocean_mask > 0.5)
    n = max(1, round(len(idx) * pct / 100.0))
    sel = rng.choice(len(idx), size=n, replace=False)
    m = np.zeros((OCEAN_H, OCEAN_W), dtype=np.float32)
    for i in sel:
        m[idx[i][0], idx[i][1]] = 1.0
    return m


def compute_gp(vel, obs_mask, ocean_mask):
    vf = np.zeros((1, 2, FULL_H, FULL_W), dtype=np.float32)
    vf[0, :, :OCEAN_H, :OCEAN_W] = vel * ocean_mask[None]
    gm = np.ones((FULL_H, FULL_W), dtype=np.float32)
    gm[:OCEAN_H, :OCEAN_W] = 1.0 - obs_mask
    gm[:OCEAN_H, :OCEAN_W] *= ocean_mask
    gm[OCEAN_H:, :] = 0.0
    gm[:, OCEAN_W:] = 0.0
    gm_t = torch.from_numpy(gm).unsqueeze(0).unsqueeze(0).expand(1, 2, -1, -1).clone()
    mean, var = gp_fill(torch.from_numpy(vf), gm_t, return_variance=True,
                        use_double=True, **GP_PARAMS)
    return (mean[0, :, :OCEAN_H, :OCEAN_W].numpy(),
            var[0, :, :OCEAN_H, :OCEAN_W].numpy())


def compute_voronoi(vel, obs_mask, ocean_mask):
    ky, kx = np.where(obs_mask > 0.5)
    if len(ky) == 0:
        return np.zeros_like(vel), np.ones_like(vel)
    tree = cKDTree(np.stack([ky, kx], axis=1).astype(np.float64))
    gy, gx = np.mgrid[0:OCEAN_H, 0:OCEAN_W]
    dist, idx = tree.query(np.stack([gy.ravel(), gx.ravel()], axis=1).astype(np.float64), k=1)
    dist = dist.reshape(OCEAN_H, OCEAN_W)
    idx = idx.reshape(OCEAN_H, OCEAN_W)
    vor_mean = np.stack([vel[0, ky, kx][idx], vel[1, ky, kx][idx]], axis=0) * ocean_mask
    dist_var = np.stack([dist ** 2, dist ** 2], axis=0) * ocean_mask
    return vor_mean, dist_var


def load_vcnn(dev):
    ck = torch.load(str(BASE_DIR / VCNN_CKPT), map_location="cpu", weights_only=False)
    m = VoronoiCNN(**ck["model_config"]).to(dev)
    m.load_state_dict(ck["model_state"])
    m.eval()
    return m


def predict_vcnn(model, vel, obs_mask, ocean_mask, dev):
    vel_n = ((vel - NM[:, None, None]) / NS[:, None, None]) * ocean_mask[None]
    vi = build_voronoi_input(vel_n, obs_mask, ocean_mask)
    with torch.no_grad():
        p = model(torch.from_numpy(vi).unsqueeze(0).to(dev))
    mt = torch.tensor(NM).view(1, 2, 1, 1).to(dev)
    st = torch.tensor(NS).view(1, 2, 1, 1).to(dev)
    phys = (p * st + mt) * torch.from_numpy(ocean_mask).float().to(dev).view(1, 1, OCEAN_H, OCEAN_W)
    return phys.squeeze(0).cpu().numpy()


def load_stage3_model():
    """Load stage-3 model with n_stage_tokens=3."""
    net = MyUNet_Attn(n_steps=N_STEPS, time_emb_dim=256, n_stage_tokens=3)
    ddpm = GaussianDDPM(net, n_steps=N_STEPS,
                        min_beta=0.0001, max_beta=0.02, device=device)
    state = torch.load(str(BASE_DIR / STAGE3_WEIGHTS), map_location="cpu",
                       weights_only=False)
    ddpm.load_state_dict(state)
    ddpm.to(device)
    ddpm.eval()
    return ddpm


def run_s6(ddpm_model, gt_np, obs_mask, ocean_mask, prior_mean, prior_var, sample_seed):
    """Run S6 RePaint with x0 prediction. Returns (2, 44, 94) numpy."""
    gt_full = np.zeros((1, 2, FULL_H, FULL_W), dtype=np.float32)
    gt_full[0, :, :OCEAN_H, :OCEAN_W] = gt_np * ocean_mask[None]
    input_image = standardizer(
        torch.from_numpy(gt_full).squeeze(0)).unsqueeze(0).to(device)

    prior_full = np.zeros((1, 2, FULL_H, FULL_W), dtype=np.float32)
    prior_full[0, :, :OCEAN_H, :OCEAN_W] = prior_mean * ocean_mask[None]
    prior_std = standardizer(
        torch.from_numpy(prior_full).squeeze(0)).unsqueeze(0).to(device)

    land_mask = (torch.from_numpy(gt_full).abs() > 1e-5).float().to(device)
    raw_miss = np.ones((FULL_H, FULL_W), dtype=np.float32)
    raw_miss[:OCEAN_H, :OCEAN_W] -= obs_mask
    raw_miss[:OCEAN_H, :OCEAN_W] *= ocean_mask
    raw_miss[OCEAN_H:, :] = 0.0
    raw_miss[:, OCEAN_W:] = 0.0
    raw_miss_t = torch.from_numpy(raw_miss).unsqueeze(0).unsqueeze(0).to(device)
    border = border_gen.generate_mask(torch.Size([1, 2, FULL_H, FULL_W])).to(device)
    missing_mask = raw_miss_t * border * land_mask

    var_full = np.zeros((1, 2, FULL_H, FULL_W), dtype=np.float32)
    var_full[0, :, :OCEAN_H, :OCEAN_W] = prior_var * ocean_mask[None]
    var_t = torch.from_numpy(var_full).to(device)

    p = S6_DEFAULTS
    current_prior = prior_std.clone()
    current_var = var_t.clone()

    for stage in range(1, p["max_stages"] + 1):
        if stage == 1:
            t_s, nf = p["t_start"], p["noise_floor"]
            seed_s = sample_seed
        else:
            t_s, nf = p["t_refine"], p["noise_floor_refine"]
            seed_s = sample_seed + stage * 10000
            current_var = current_var * p["var_decay"]

        torch.manual_seed(seed_s)
        with torch.no_grad():
            stage_out = repaint_gp_init_adaptive(
                ddpm_model, input_image, missing_mask,
                gp_image=current_prior, gp_variance_map=current_var,
                t_start=t_s, noise_floor=nf, n_samples=1, device=device,
                noise_strategy=noise_strategy, prediction_target="x0",
                resample_steps=p["resample_steps"], project_div_free=False,
                anneal_floor=False, gamma=p["gamma"],
            )
        current_prior = stage_out.clone()

    result_phys = standardizer.unstandardize(stage_out.squeeze(0).cpu())
    return result_phys[:, :OCEAN_H, :OCEAN_W].numpy()


def ocean_mse(pred, gt, ocean_mask):
    ocean_b = ocean_mask.astype(bool)
    diff = pred[:, ocean_b] - gt[:, ocean_b]
    return float((diff ** 2).mean())


# ═══════════════════════════════════════════════════════════════════
def main():
    coverage = args.coverage
    n_samples = args.n_samples

    print(f"\n{'='*70}")
    print(f"STAGE-3 DETACHED ROLLOUT — QUICK EVAL")
    print(f"Coverage: {coverage}%  |  Samples: {n_samples}")
    print(f"{'='*70}")

    val = load_val_data()
    ocean_mask = get_ocean_mask(val)
    n_ocean = int(ocean_mask.sum())
    n_obs = max(1, round(n_ocean * coverage / 100.0))
    print(f"  Ocean cells: {n_ocean}, Observations: {n_obs}")

    rng = np.random.default_rng(seed=7777)
    val_indices = rng.choice(val.shape[0], size=min(n_samples, val.shape[0]), replace=False)
    val_indices.sort()

    print("\nLoading models...")
    vcnn_model = load_vcnn(device)
    print("  V-CNN loaded")
    stage3_ddpm = load_stage3_model()
    print("  Stage-3 DDPM loaded (n_stage_tokens=3)")

    results = {"gp": [], "vor": [], "vcnn": [], "stage3": []}

    print(f"\n{'#':>3} {'ValIdx':>7} {'GP':>10} {'Voronoi':>10} {'V-CNN':>10} {'Stage3':>10} {'Time':>7}")
    print("-" * 65)

    t0_global = time.time()
    for i, vi in enumerate(val_indices):
        gt_full = val[vi].numpy()  # (2, 64, 128)
        gt = gt_full[:, :OCEAN_H, :OCEAN_W]  # (2, 44, 94) ocean subdomain
        seed = S6_DEFAULTS["seed"] + vi
        obs_mask = random_mask(ocean_mask, coverage, np.random.default_rng(seed=seed))
        vel_obs = gt * obs_mask[None]

        t0 = time.time()

        gp_mean, gp_var = compute_gp(vel_obs, obs_mask, ocean_mask)
        gp_mse = ocean_mse(gp_mean, gt, ocean_mask)

        vor_mean, vor_var = compute_voronoi(vel_obs, obs_mask, ocean_mask)
        vor_mse = ocean_mse(vor_mean, gt, ocean_mask)

        vcnn_pred = predict_vcnn(vcnn_model, vel_obs, obs_mask, ocean_mask, device)
        vcnn_mse = ocean_mse(vcnn_pred, gt, ocean_mask)

        s3_pred = run_s6(stage3_ddpm, gt, obs_mask, ocean_mask,
                         vor_mean, vor_var, seed)
        s3_mse = ocean_mse(s3_pred, gt, ocean_mask)

        elapsed = time.time() - t0
        print(f"{i+1:>3} {vi:>7} {gp_mse:>10.5f} {vor_mse:>10.5f} "
              f"{vcnn_mse:>10.5f} {s3_mse:>10.5f} {elapsed:>6.1f}s")

        results["gp"].append(gp_mse)
        results["vor"].append(vor_mse)
        results["vcnn"].append(vcnn_mse)
        results["stage3"].append(s3_mse)

    total = time.time() - t0_global

    gp = np.array(results["gp"])
    vor = np.array(results["vor"])
    vcnn = np.array(results["vcnn"])
    s3 = np.array(results["stage3"])

    print(f"\n{'='*70}")
    print(f"SUMMARY — {coverage}% coverage, {len(val_indices)} samples, {total:.0f}s")
    print(f"{'='*70}")
    print(f"{'Method':<20} {'Mean MSE':<12} {'vs V-CNN':<12}")
    print("-" * 45)
    for name, arr in [("GP", gp), ("Voronoi", vor), ("V-CNN", vcnn), ("Stage3-DDPM", s3)]:
        ratio = arr.mean() / vcnn.mean() if vcnn.mean() > 0 else float("inf")
        print(f"{name:<20} {arr.mean():.6f}     {ratio:.3f}x")


if __name__ == "__main__":
    main()
