#!/usr/bin/env python3
"""Single-step Voronoi refinement with the voronoi-trained DDPM.

Instead of iterative S6 RePaint (which causes distribution mismatch),
this script:
  1. Computes Voronoi fill from sparse observations
  2. Noises it to a fixed timestep t: x_t = sqrt(ᾱ_t)*vor + sqrt(1-ᾱ_t)*ε
  3. Makes ONE x0 prediction with the voronoi-trained model
  4. Replaces observed cells with ground truth (data consistency)
  5. Reports MSE

Sweeps over multiple timesteps to find the sweet spot.

Usage:
    PYTHONPATH=. python3 tmp_vor_single_step.py
    PYTHONPATH=. python3 tmp_vor_single_step.py --n-samples 10 --coverage 0.5
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

# Model paths
VOR_TRAINED_WEIGHTS = (
    "experiments/10_topology_metrics/voronoi_topo_training/results/"
    "inpaint_gaussian_t250_best_ema_weights.pt"
)
TOPO_WEIGHTS = (
    "experiments/10_topology_metrics/topo_aware_training/results/"
    "inpaint_gaussian_t250_best_ema_weights.pt"
)
VCNN_CKPT = "results/voronoi_cnn/voronoi_cnn_best.pt"

# Timesteps to sweep
TIMESTEPS = [1, 3, 5, 10, 15, 20, 30, 50, 75, 100, 150, 200, 249]

# ── args ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--coverage", type=float, default=0.5)
parser.add_argument("--n-samples", type=int, default=10)
parser.add_argument("--n-ensemble", type=int, default=5,
                    help="Number of noise draws per (sample, timestep) pair")
parser.add_argument("--vor-weights", type=str, default=VOR_TRAINED_WEIGHTS,
                    help="Path to Voronoi-trained DDPM weights")
parser.add_argument("--output-suffix", type=str, default="",
                    help="Optional suffix appended to saved result filename")
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
def load_val_data():
    with open(str(BASE_DIR / "data.pickle"), "rb") as f:
        _train, val, _test = pickle.load(f)
    t = torch.nan_to_num(
        torch.from_numpy(np.ascontiguousarray(val)).float().permute(3, 2, 1, 0),
        nan=0.0)
    return t


def get_ocean_mask(val_tensor):
    sample = val_tensor[0]
    ocean = (sample.abs() > 1e-5).any(dim=0).float()
    return ocean.numpy()


def random_mask(ocean_mask, pct, rng):
    idx = np.argwhere(ocean_mask > 0.5)
    n = max(1, round(len(idx) * pct / 100.0))
    sel = rng.choice(len(idx), size=n, replace=False)
    m = np.zeros((OCEAN_H, OCEAN_W), dtype=np.float32)
    for i in sel:
        m[idx[i][0], idx[i][1]] = 1.0
    return m


def compute_voronoi(vel, obs_mask, ocean_mask):
    ky, kx = np.where(obs_mask > 0.5)
    if len(ky) == 0:
        return np.zeros_like(vel)
    obs_coords = np.stack([ky, kx], axis=1).astype(np.float64)
    tree = cKDTree(obs_coords)
    gy, gx = np.mgrid[0:OCEAN_H, 0:OCEAN_W]
    grid_coords = np.stack([gy.ravel(), gx.ravel()], axis=1).astype(np.float64)
    _, idx = tree.query(grid_coords, k=1)
    idx = idx.reshape(OCEAN_H, OCEAN_W)
    obs_u = vel[0, ky, kx]
    obs_v = vel[1, ky, kx]
    vor_u = obs_u[idx] * ocean_mask
    vor_v = obs_v[idx] * ocean_mask
    return np.stack([vor_u, vor_v], axis=0)


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


def load_uncond_model(weight_path):
    net = MyUNet_Attn(n_steps=N_STEPS, time_emb_dim=256)
    ddpm = GaussianDDPM(net, n_steps=N_STEPS,
                        min_beta=0.0001, max_beta=0.02, device=device)
    state = torch.load(str(BASE_DIR / weight_path), map_location="cpu",
                       weights_only=False)
    ddpm.load_state_dict(state)
    ddpm.to(device)
    ddpm.eval()
    return ddpm


def ocean_mse(pred, gt, ocean_mask):
    ocean_b = ocean_mask.astype(bool)
    diff = pred[:, ocean_b] - gt[:, ocean_b]
    return float((diff ** 2).mean())


def single_step_x0(ddpm_model, voronoi_std_full, t_int, n_ensemble, rng_seed):
    """Single-step x0 prediction from noised Voronoi.

    voronoi_std_full: (1, 2, 64, 128) standardized Voronoi, on device.
    t_int: integer timestep (0-indexed).
    n_ensemble: number of noise draws to average.

    Returns: (1, 2, 64, 128) x0 prediction (standardized), on device.
    """
    a_bar = ddpm_model.alpha_bars[t_int]
    sqrt_a = a_bar.sqrt()
    sqrt_one_minus_a = (1 - a_bar).sqrt()

    t_tensor = torch.tensor([t_int], dtype=torch.long, device=device)
    preds = []

    for k in range(n_ensemble):
        torch.manual_seed(rng_seed + k * 1000)
        eps = torch.randn_like(voronoi_std_full)
        x_t = sqrt_a * voronoi_std_full + sqrt_one_minus_a * eps
        with torch.no_grad():
            x0_pred = ddpm_model.backward(x_t, t_tensor)
        preds.append(x0_pred)

    return torch.stack(preds).mean(dim=0)


def single_step_eps(ddpm_model, voronoi_std_full, t_int, n_ensemble, rng_seed):
    """Single-step eps-prediction → x0 from noised Voronoi.

    For eps-prediction models: x0 = (x_t - sqrt(1-ᾱ)*eps_pred) / sqrt(ᾱ)
    """
    a_bar = ddpm_model.alpha_bars[t_int]
    sqrt_a = a_bar.sqrt()
    sqrt_one_minus_a = (1 - a_bar).sqrt()

    t_tensor = torch.tensor([t_int], dtype=torch.long, device=device)
    preds = []

    for k in range(n_ensemble):
        torch.manual_seed(rng_seed + k * 1000)
        eps = torch.randn_like(voronoi_std_full)
        x_t = sqrt_a * voronoi_std_full + sqrt_one_minus_a * eps
        with torch.no_grad():
            eps_pred = ddpm_model.backward(x_t, t_tensor)
        x0_pred = (x_t - sqrt_one_minus_a * eps_pred) / sqrt_a
        preds.append(x0_pred)

    return torch.stack(preds).mean(dim=0)


# ═════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════

def main():
    coverage = args.coverage
    n_samples = args.n_samples
    n_ensemble = args.n_ensemble

    print(f"\n{'='*90}")
    print(f"SINGLE-STEP VORONOI REFINEMENT — Timestep Sweep")
    print(f"Coverage: {coverage}%  |  Samples: {n_samples}  |  Ensemble: {n_ensemble}")
    print(f"Timesteps: {TIMESTEPS}")
    print(f"{'='*90}")

    # Load data
    print("\nLoading data...")
    val = load_val_data()
    n_val = val.shape[0]
    ocean_mask = get_ocean_mask(val)
    n_ocean = int(ocean_mask.sum())
    n_obs = max(1, round(n_ocean * coverage / 100.0))
    print(f"  Val frames: {n_val},  Ocean cells: {n_ocean},  Obs: {n_obs}")

    rng = np.random.default_rng(seed=7777)
    val_indices = rng.choice(n_val, size=min(n_samples, n_val), replace=False)
    val_indices.sort()
    n_samples = len(val_indices)

    # Load models
    print("\nLoading models...")
    vcnn_model = load_vcnn(device)
    print("  V-CNN loaded")
    vor_ddpm = load_uncond_model(args.vor_weights)
    print("  Vor-trained DDPM loaded (x0-prediction)")
    topo_ddpm = load_uncond_model(TOPO_WEIGHTS)
    print("  Old topo DDPM loaded (eps-prediction)")

    # Pre-allocate results: {method: [sample_mse, ...]}
    # Methods: voronoi, vcnn, single_step_vor_t{T}, single_step_topo_t{T}
    results = {
        "voronoi": [],
        "vcnn": [],
    }
    for t in TIMESTEPS:
        results[f"ss_vor_t{t}"] = []       # single-step with vor-trained model
        results[f"ss_topo_t{t}"] = []       # single-step with old topo model

    ocean_mask_t = torch.from_numpy(ocean_mask).float().to(device)

    print(f"\nProcessing {n_samples} samples...\n")
    t0_all = time.time()

    for count, vi in enumerate(val_indices):
        sample_seed = 42 + vi
        gt = val[vi].numpy()  # (2, 44, 94) physical

        # Generate mask
        mask_rng = np.random.default_rng(seed=sample_seed)
        obs_mask = random_mask(ocean_mask, coverage, mask_rng)
        vel_obs = gt * obs_mask[None]

        # Voronoi fill
        vor_mean = compute_voronoi(vel_obs, obs_mask, ocean_mask)
        vor_mse = ocean_mse(vor_mean, gt, ocean_mask)
        results["voronoi"].append(vor_mse)

        # V-CNN
        vcnn_pred = predict_vcnn(vcnn_model, vel_obs, obs_mask, ocean_mask, device)
        vcnn_mse = ocean_mse(vcnn_pred, gt, ocean_mask)
        results["vcnn"].append(vcnn_mse)

        # Prepare Voronoi for DDPM: standardize, pad to (1, 2, 64, 128)
        vor_full = np.zeros((2, FULL_H, FULL_W), dtype=np.float32)
        vor_full[:, :OCEAN_H, :OCEAN_W] = vor_mean * ocean_mask[None]
        vor_std = standardizer(torch.from_numpy(vor_full)).unsqueeze(0).to(device)

        # Prepare observed data in standardized space for data consistency
        gt_full = np.zeros((2, FULL_H, FULL_W), dtype=np.float32)
        gt_full[:, :OCEAN_H, :OCEAN_W] = gt * ocean_mask[None]
        gt_std = standardizer(torch.from_numpy(gt_full)).unsqueeze(0).to(device)

        # Observed mask in full space
        obs_full = np.zeros((FULL_H, FULL_W), dtype=np.float32)
        obs_full[:OCEAN_H, :OCEAN_W] = obs_mask
        obs_mask_t = torch.from_numpy(obs_full).unsqueeze(0).unsqueeze(0).to(device)

        # Single-step at each timestep
        for t_val in TIMESTEPS:
            # Vor-trained model (x0 prediction)
            x0_vor = single_step_x0(vor_ddpm, vor_std, t_val, n_ensemble,
                                     rng_seed=sample_seed)
            # Data consistency: replace observed cells
            x0_vor = x0_vor * (1 - obs_mask_t) + gt_std * obs_mask_t
            # Unstandardize
            pred_phys = standardizer.unstandardize(x0_vor.squeeze(0).cpu())
            pred_np = pred_phys[:, :OCEAN_H, :OCEAN_W].numpy() * ocean_mask[None]
            mse_vor = ocean_mse(pred_np, gt, ocean_mask)
            results[f"ss_vor_t{t_val}"].append(mse_vor)

            # Old topo model (eps prediction) — for comparison
            x0_topo = single_step_eps(topo_ddpm, vor_std, t_val, n_ensemble,
                                       rng_seed=sample_seed)
            x0_topo = x0_topo * (1 - obs_mask_t) + gt_std * obs_mask_t
            pred_phys_t = standardizer.unstandardize(x0_topo.squeeze(0).cpu())
            pred_np_t = pred_phys_t[:, :OCEAN_H, :OCEAN_W].numpy() * ocean_mask[None]
            mse_topo = ocean_mse(pred_np_t, gt, ocean_mask)
            results[f"ss_topo_t{t_val}"].append(mse_topo)

        elapsed = time.time() - t0_all
        print(f"  Sample {count+1}/{n_samples} (val idx {vi}) — "
              f"Voronoi: {vor_mse:.5f}, V-CNN: {vcnn_mse:.5f}, "
              f"best SS(vor-tr): {min(results[f'ss_vor_t{t}'][-1] for t in TIMESTEPS):.5f}, "
              f"elapsed: {elapsed:.1f}s")

    elapsed_total = time.time() - t0_all

    # ── Save raw results ─────────────────────────────────────────────
    suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    out_path = (BASE_DIR / "experiments/10_topology_metrics/voronoi_topo_training/"
                f"results/single_step_sweep_{coverage}pct{suffix}.pt")
    torch.save({
        "coverage_pct": coverage,
        "n_samples": n_samples,
        "n_ensemble": n_ensemble,
        "val_indices": val_indices.tolist(),
        "timesteps": TIMESTEPS,
        "vor_trained_weights": args.vor_weights,
        "results": results,
    }, str(out_path))
    print(f"\nResults saved to {out_path}")

    # ── Summary table ────────────────────────────────────────────────
    vor_arr = np.array(results["voronoi"])
    vcnn_arr = np.array(results["vcnn"])
    vcnn_mean = vcnn_arr.mean()

    print(f"\n{'='*90}")
    print(f"SINGLE-STEP SWEEP RESULTS — {coverage}% coverage, "
          f"{n_samples} samples, {n_ensemble} ensemble, {elapsed_total:.0f}s")
    print(f"{'='*90}")

    print(f"\n{'Method':<30}\t{'Mean MSE':>10}\t{'Median':>10}\t{'vs V-CNN':>12}")
    print("-" * 80)

    # Baselines
    for name, arr in [("Voronoi (raw)", vor_arr), ("V-CNN", vcnn_arr)]:
        m, md = arr.mean(), np.median(arr)
        vs = "1.0x" if name == "V-CNN" else f"{m/vcnn_mean:.1f}x"
        print(f"{name:<30}\t{m:>10.5f}\t{md:>10.5f}\t{vs:>12}")

    print()

    # Single-step with vor-trained model
    best_t_vor, best_mse_vor = None, 1e10
    print("  Voronoi-trained model (x0-prediction):")
    for t_val in TIMESTEPS:
        arr = np.array(results[f"ss_vor_t{t_val}"])
        m = arr.mean()
        md = np.median(arr)
        vs = f"{m/vcnn_mean:.2f}x"
        marker = ""
        if m < best_mse_vor:
            best_mse_vor = m
            best_t_vor = t_val
        print(f"    SS vor-tr t={t_val:<4}\t\t{m:>10.5f}\t{md:>10.5f}\t{vs:>12}")

    print()

    # Single-step with old topo model
    best_t_topo, best_mse_topo = None, 1e10
    print("  Old topo model (eps-prediction, on Voronoi input):")
    for t_val in TIMESTEPS:
        arr = np.array(results[f"ss_topo_t{t_val}"])
        m = arr.mean()
        md = np.median(arr)
        vs = f"{m/vcnn_mean:.2f}x"
        if m < best_mse_topo:
            best_mse_topo = m
            best_t_topo = t_val
        print(f"    SS topo t={t_val:<4}\t\t{m:>10.5f}\t{md:>10.5f}\t{vs:>12}")

    print(f"\n{'='*90}")
    print(f"BEST TIMESTEPS:")
    print(f"  Vor-trained model: t={best_t_vor}, mean MSE = {best_mse_vor:.5f} "
          f"({best_mse_vor/vcnn_mean:.2f}x V-CNN)")
    print(f"  Old topo model:    t={best_t_topo}, mean MSE = {best_mse_topo:.5f} "
          f"({best_mse_topo/vcnn_mean:.2f}x V-CNN)")
    print(f"  Voronoi raw:       mean MSE = {vor_arr.mean():.5f} "
          f"({vor_arr.mean()/vcnn_mean:.2f}x V-CNN)")
    print(f"  V-CNN:             mean MSE = {vcnn_mean:.5f}")

    # Does it at least beat raw Voronoi?
    if best_mse_vor < vor_arr.mean():
        pct_imp = (1 - best_mse_vor / vor_arr.mean()) * 100
        print(f"\n  → Vor-trained single-step beats raw Voronoi by {pct_imp:.1f}%")
    else:
        print(f"\n  → Vor-trained single-step does NOT beat raw Voronoi")

    if best_mse_vor < best_mse_topo:
        print(f"  → Vor-trained model wins over old topo (both single-step)")
    else:
        print(f"  → Old topo model wins over vor-trained (both single-step)")

    print(f"{'='*90}\n")


if __name__ == "__main__":
    main()
