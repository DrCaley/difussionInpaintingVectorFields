#!/usr/bin/env python3
"""Standalone topo-aware GP-Diff evaluation at 0.5% coverage.

Generates everything from scratch: random masks, GP, V-CNN, baseline GP-Diff,
and topo GP-Diff S6 pipeline. No dependency on pre-computed bulk eval data.

Usage:
    PYTHONPATH=. python tmp_topo_eval_0p5.py
    PYTHONPATH=. python tmp_topo_eval_0p5.py --n-samples 10 --coverage 0.5
"""

import argparse, os, pickle, sys, time
from pathlib import Path
import numpy as np
import torch

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
FULL_H, FULL_W   = 64, 128
N_STEPS = 250

U_MEAN, U_STD = -0.06929559429949586, 0.1358005549716049
V_MEAN, V_STD = -0.0323937796117541, 0.08899177232117582
NM = np.array([U_MEAN, V_MEAN], dtype=np.float32)
NS = np.array([U_STD, V_STD], dtype=np.float32)
standardizer = ZScoreStandardizer(U_MEAN, U_STD, V_MEAN, V_STD)

GP_PARAMS = dict(lengthscale=14.1, variance=0.0103420345, noise=1e-8,
                 kernel_type="rbf_legacy", coord_system="pixels")

# S6 parameters
S6_DEFAULTS = dict(
    max_stages=6,
    t_start=75,
    t_refine=50,
    resample_steps=5,
    noise_floor=0.2,
    noise_floor_refine=0.3,
    var_decay=0.1,
    gamma=3.0,
    seed=42,
)

# Model paths
TOPO_WEIGHTS = (
    "experiments/10_topology_metrics/topo_aware_training/results/"
    "inpaint_gaussian_t250_best_ema_weights.pt"
)
BASELINE_CKPT = (
    "experiments/08_network_architecture/repaint_gaussian_attn_v2/results/"
    "inpaint_gaussian_t250_best_checkpoint.pt"
)
VCNN_CKPT = "results/voronoi_cnn/voronoi_cnn_best.pt"

# ── args ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--coverage", type=float, default=0.5,
                    help="Coverage %% (default 0.5)")
parser.add_argument("--n-samples", type=int, default=10,
                    help="Number of validation samples to evaluate")
args = parser.parse_args()

# ── device ───────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Device: {device}")

# ── data loading ─────────────────────────────────────────────────────
noise_strategy = get_noise_strategy("gaussian")
border_gen = BorderMaskGenerator()


def load_val_data():
    """Load validation split. Returns (N, 2, 44, 94) physical-space tensor."""
    with open(str(BASE_DIR / "data.pickle"), "rb") as f:
        _train, val, _test = pickle.load(f)
    t = torch.nan_to_num(
        torch.from_numpy(np.ascontiguousarray(val)).float().permute(3, 2, 1, 0),
        nan=0.0)
    return t  # (N, 2, 44, 94) physical space


def get_ocean_mask(val_tensor):
    """Derive ocean mask from first validation sample. val: (N, 2, 44, 94)."""
    sample = val_tensor[0]  # (2, 44, 94)
    ocean = (sample.abs() > 1e-5).any(dim=0).float()
    return ocean.numpy()


def random_mask(ocean_mask, pct, rng):
    """Generate random mask with given coverage percentage."""
    idx = np.argwhere(ocean_mask > 0.5)
    n = max(1, round(len(idx) * pct / 100.0))
    sel = rng.choice(len(idx), size=n, replace=False)
    m = np.zeros((OCEAN_H, OCEAN_W), dtype=np.float32)
    for i in sel:
        m[idx[i][0], idx[i][1]] = 1.0
    return m


# ── GP ───────────────────────────────────────────────────────────────

def compute_gp(vel, obs_mask, ocean_mask):
    """Run GP interpolation. Returns (mean, var) each (2, 44, 94)."""
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


# ── V-CNN ────────────────────────────────────────────────────────────

def load_vcnn(dev):
    ck = torch.load(str(BASE_DIR / VCNN_CKPT), map_location="cpu",
                    weights_only=False)
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
    phys = (p * st + mt) * torch.from_numpy(ocean_mask).float().to(dev).view(
        1, 1, OCEAN_H, OCEAN_W)
    return phys.squeeze(0).cpu().numpy()


# ── DDPM model loading ───────────────────────────────────────────────

def load_uncond_model(weight_path, is_checkpoint=False):
    """Load unconditional MyUNet_Attn DDPM."""
    if is_checkpoint:
        ckpt = torch.load(str(BASE_DIR / weight_path), map_location="cpu",
                          weights_only=False)
        n_steps = ckpt.get("n_steps", 250)
        min_beta = ckpt.get("min_beta", 0.0001)
        max_beta = ckpt.get("max_beta", 0.02)
        net = MyUNet_Attn(n_steps=n_steps, time_emb_dim=256)
        ddpm = GaussianDDPM(net, n_steps=n_steps,
                            min_beta=min_beta, max_beta=max_beta, device=device)
        ddpm.load_state_dict(ckpt["model_state_dict"])
    else:
        net = MyUNet_Attn(n_steps=N_STEPS, time_emb_dim=256)
        ddpm = GaussianDDPM(net, n_steps=N_STEPS,
                            min_beta=0.0001, max_beta=0.02, device=device)
        state = torch.load(str(BASE_DIR / weight_path), map_location="cpu",
                           weights_only=False)
        ddpm.load_state_dict(state)
    ddpm.to(device)
    ddpm.eval()
    return ddpm


# ── S6 RePaint ───────────────────────────────────────────────────────

def run_s6(ddpm_model, gt_np, obs_mask, ocean_mask, gp_mean, gp_var,
           sample_seed):
    """Run S6 multi-stage GP-Diff on a single sample. Returns (2,44,94) numpy."""
    gt_full = np.zeros((1, 2, FULL_H, FULL_W), dtype=np.float32)
    gt_full[0, :, :OCEAN_H, :OCEAN_W] = gt_np * ocean_mask[None]
    input_image = standardizer(
        torch.from_numpy(gt_full).squeeze(0)).unsqueeze(0).to(device)

    gp_full = np.zeros((1, 2, FULL_H, FULL_W), dtype=np.float32)
    gp_full[0, :, :OCEAN_H, :OCEAN_W] = gp_mean * ocean_mask[None]
    gp_std = standardizer(
        torch.from_numpy(gp_full).squeeze(0)).unsqueeze(0).to(device)

    land_mask = (torch.from_numpy(gt_full).abs() > 1e-5).float().to(device)
    raw_miss = np.ones((FULL_H, FULL_W), dtype=np.float32)
    raw_miss[:OCEAN_H, :OCEAN_W] -= obs_mask
    raw_miss[:OCEAN_H, :OCEAN_W] *= ocean_mask
    raw_miss[OCEAN_H:, :] = 0.0
    raw_miss[:, OCEAN_W:] = 0.0
    raw_miss_t = torch.from_numpy(raw_miss).unsqueeze(0).unsqueeze(0).to(device)
    border = border_gen.generate_mask(
        torch.Size([1, 2, FULL_H, FULL_W])).to(device)
    missing_mask = raw_miss_t * border * land_mask

    gp_var_full = np.zeros((1, 2, FULL_H, FULL_W), dtype=np.float32)
    gp_var_full[0, :, :OCEAN_H, :OCEAN_W] = gp_var * ocean_mask[None]
    gp_var_t = torch.from_numpy(gp_var_full).to(device)

    p = S6_DEFAULTS
    current_prior = gp_std.clone()
    current_var = gp_var_t.clone()

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
                gp_image=current_prior,
                gp_variance_map=current_var,
                t_start=t_s,
                noise_floor=nf,
                n_samples=1, device=device,
                noise_strategy=noise_strategy,
                prediction_target="eps",
                resample_steps=p["resample_steps"],
                project_div_free=False,
                anneal_floor=False,
                gamma=p["gamma"],
            )
        current_prior = stage_out.clone()

    result_phys = standardizer.unstandardize(stage_out.squeeze(0).cpu())
    return result_phys[:, :OCEAN_H, :OCEAN_W].numpy()


def ocean_mse(pred, gt, ocean_mask):
    """MSE over ocean cells. pred, gt: (2, H, W) numpy."""
    ocean_b = ocean_mask.astype(bool)
    diff = pred[:, ocean_b] - gt[:, ocean_b]
    return float((diff ** 2).mean())


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    coverage = args.coverage
    n_samples = args.n_samples

    print(f"\n{'='*80}")
    print(f"STANDALONE TOPO-AWARE GP-DIFF EVALUATION")
    print(f"Coverage: {coverage}%  |  Samples: {n_samples}")
    print(f"{'='*80}")

    # Load validation data
    print("\nLoading validation data...")
    val = load_val_data()
    n_val = val.shape[0]
    ocean_mask = get_ocean_mask(val)
    n_ocean = int(ocean_mask.sum())
    n_obs = max(1, round(n_ocean * coverage / 100.0))
    print(f"  Validation frames: {n_val}")
    print(f"  Ocean cells: {n_ocean}")
    print(f"  Observations per sample: {n_obs} ({coverage}%)")

    # Select random validation indices
    rng = np.random.default_rng(seed=7777)
    val_indices = rng.choice(n_val, size=min(n_samples, n_val), replace=False)
    val_indices.sort()
    n_samples = len(val_indices)

    # Load models
    print("\nLoading models...")
    print("  V-CNN...")
    vcnn_model = load_vcnn(device)
    print("  Baseline DDPM (exp02)...")
    baseline_ddpm = load_uncond_model(BASELINE_CKPT, is_checkpoint=True)
    print("  Topo-aware DDPM (exp10)...")
    topo_ddpm = load_uncond_model(TOPO_WEIGHTS, is_checkpoint=False)
    print("  Models loaded.\n")

    # Results
    results = {
        "gp_mse": [], "vcnn_mse": [],
        "gpdiff_mse": [], "topo_gpdiff_mse": [],
    }

    print(f"{'#':>4} {'ValIdx':>7} {'GP MSE':>12} {'VCNN MSE':>12} "
          f"{'GPDiff MSE':>12} {'Topo MSE':>12} {'Topo/VCNN':>10} {'Time':>7}")
    print("-" * 95)

    t0_global = time.time()

    for count, vi in enumerate(val_indices):
        sample_seed = S6_DEFAULTS["seed"] + vi

        # Extract GT — val is already physical space (2, 44, 94)
        gt = val[vi].numpy()  # (2, 44, 94) physical space

        # Generate random mask
        mask_rng = np.random.default_rng(seed=sample_seed)
        obs_mask = random_mask(ocean_mask, coverage, mask_rng)

        # Observed velocity (physical space, masked)
        vel_obs = gt * obs_mask[None]

        t0 = time.time()

        # GP
        gp_mean, gp_var = compute_gp(vel_obs, obs_mask, ocean_mask)
        gp_pred = gp_mean  # (2, 44, 94)
        gp_mse = ocean_mse(gp_pred, gt, ocean_mask)

        # V-CNN
        vcnn_pred = predict_vcnn(vcnn_model, vel_obs, obs_mask, ocean_mask, device)
        vcnn_mse = ocean_mse(vcnn_pred, gt, ocean_mask)

        # Baseline GP-Diff S6
        base_pred = run_s6(baseline_ddpm, gt, obs_mask, ocean_mask,
                           gp_mean, gp_var, sample_seed)
        base_mse = ocean_mse(base_pred, gt, ocean_mask)

        # Topo GP-Diff S6
        topo_pred = run_s6(topo_ddpm, gt, obs_mask, ocean_mask,
                           gp_mean, gp_var, sample_seed)
        topo_mse = ocean_mse(topo_pred, gt, ocean_mask)

        elapsed = time.time() - t0

        ratio_vcnn = topo_mse / vcnn_mse if vcnn_mse > 0 else float("inf")

        print(f"{count+1:>4} {vi:>7} {gp_mse:>12.8f} {vcnn_mse:>12.8f} "
              f"{base_mse:>12.8f} {topo_mse:>12.8f} {ratio_vcnn:>9.3f}x "
              f"{elapsed:>6.1f}s")

        results["gp_mse"].append(gp_mse)
        results["vcnn_mse"].append(vcnn_mse)
        results["gpdiff_mse"].append(base_mse)
        results["topo_gpdiff_mse"].append(topo_mse)

    elapsed_total = time.time() - t0_global

    # ── Save results ─────────────────────────────────────────────────
    n = len(results["topo_gpdiff_mse"])
    gp = np.array(results["gp_mse"])
    vcnn = np.array(results["vcnn_mse"])
    gpdiff = np.array(results["gpdiff_mse"])
    topo = np.array(results["topo_gpdiff_mse"])

    out_path = (BASE_DIR / "experiments/10_topology_metrics/topo_aware_training/"
                f"results/topo_gpdiff_eval_{coverage}pct.pt")
    torch.save({
        "coverage_pct": coverage,
        "n_samples": n,
        "val_indices": val_indices.tolist(),
        "results": results,
        "s6_params": S6_DEFAULTS,
        "topo_weights": TOPO_WEIGHTS,
        "baseline_ckpt": BASELINE_CKPT,
    }, str(out_path))
    print(f"\nResults saved to {out_path}")

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"SUMMARY: {coverage}% coverage, {n} samples, "
          f"{elapsed_total:.0f}s total")
    print(f"{'='*80}")

    methods = {"GP": gp, "V-CNN": vcnn, "GP-Diff (base)": gpdiff,
               "GP-Diff (topo)": topo}

    print(f"\n{'Method':<20} {'Mean MSE':>12} {'Median MSE':>12} "
          f"{'Std':>12} {'Wins':>6}")
    print("-" * 65)

    all_methods = list(methods.items())
    for name, arr in all_methods:
        wins = sum(1 for j in range(n) if all(
            arr[j] <= methods[other_name][j]
            for other_name, other_arr in all_methods if other_name != name))
        print(f"{name:<20} {arr.mean():>12.8f} {np.median(arr):>12.8f} "
              f"{arr.std():>12.8f} {wins:>6}")

    # Head-to-head
    topo_beats_vcnn = sum(1 for j in range(n) if topo[j] < vcnn[j])
    topo_beats_gpdiff = sum(1 for j in range(n) if topo[j] < gpdiff[j])
    vcnn_beats_gpdiff = sum(1 for j in range(n) if vcnn[j] < gpdiff[j])

    print(f"\nHead-to-head ({n} samples):")
    print(f"  Topo beats V-CNN:      {topo_beats_vcnn}/{n} "
          f"({100*topo_beats_vcnn/n:.1f}%)")
    print(f"  Topo beats GP-Diff:    {topo_beats_gpdiff}/{n} "
          f"({100*topo_beats_gpdiff/n:.1f}%)")
    print(f"  V-CNN beats GP-Diff:   {vcnn_beats_gpdiff}/{n} "
          f"({100*vcnn_beats_gpdiff/n:.1f}%)")

    print(f"\nMean MSE ratios:")
    print(f"  Topo / GP:       {topo.mean() / gp.mean():.4f}x")
    print(f"  Topo / V-CNN:    {topo.mean() / vcnn.mean():.4f}x")
    print(f"  Topo / GP-Diff:  {topo.mean() / gpdiff.mean():.4f}x")
    print(f"  V-CNN / GP:      {vcnn.mean() / gp.mean():.4f}x")
    print(f"  GP-Diff / GP:    {gpdiff.mean() / gp.mean():.4f}x")


if __name__ == "__main__":
    main()
