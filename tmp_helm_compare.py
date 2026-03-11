#!/usr/bin/env python3
"""Compare all 3 Helmholtz models + V-CNN + diagnostic for cancellation check.

Models:
  1. helmholtz_baseline   (MSE loss only — known cancellation degeneracy)
  2. helmholtz_supervised (decomp supervision + orthogonality penalty)
  3. helmholtz_split_noise (Helmholtz-split noise strategy)

For each model we run single-step x0 prediction at several t_vals,
report MSE vs V-CNN, and print Helmholtz decomposition diagnostics
(energy fractions, φ/ψ ratio, per-component MSE, solenoidal leakage).

Usage:
    PYTHONPATH=. python3 tmp_helm_compare.py
    PYTHONPATH=. python3 tmp_helm_compare.py --n-samples 10
"""

import argparse, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

BASE_DIR = Path(__file__).resolve().parent

from ddpm.neural_networks.ddpm import GaussianDDPM
from ddpm.neural_networks.unets.unet_helmholtz import MyUNet_Helmholtz
from ddpm.helper_functions.standardize_data import ZScoreStandardizer
from ddpm.utils.helmholtz_split import helmholtz_decompose
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

MODELS = {
    "baseline": "experiments/12_helmholtz_dual_head/helmholtz_baseline/results/"
                "inpaint_gaussian_t250_best_weights.pt",
    "supervised": "experiments/12_helmholtz_dual_head/helmholtz_supervised/results/"
                  "inpaint_gaussian_t250_best_weights.pt",
    "split_noise": "experiments/12_helmholtz_dual_head/helmholtz_split_noise/results/"
                   "inpaint_gaussian_t250_best_weights.pt",
}
VCNN_CKPT = "results/voronoi_cnn/voronoi_cnn_best.pt"

parser = argparse.ArgumentParser()
parser.add_argument("--coverage", type=float, default=0.5)
parser.add_argument("--n-samples", type=int, default=5)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--t-vals", type=int, nargs="+", default=[25, 50, 75])
args = parser.parse_args()

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Device: {device}")


# ── Helpers ──────────────────────────────────────────────────────────
def load_val_data():
    import pickle
    with open(str(BASE_DIR / "data.pickle"), "rb") as f:
        _, _, test_raw = pickle.load(f)
    test = np.transpose(test_raw, (3, 2, 1, 0)).astype(np.float32)
    return torch.from_numpy(np.nan_to_num(test, nan=0.0))


def get_ocean_mask(val_tensor):
    return (val_tensor[0].abs().sum(dim=0) > 1e-7).float().numpy()


def make_border_mask(dev):
    m = torch.zeros(1, 1, FULL_H, FULL_W, device=dev)
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


def ocean_mse(pred, gt, ocean_mask):
    b = ocean_mask.astype(bool)
    return float(((pred[:, b] - gt[:, b]) ** 2).mean())


def energy_fraction(part, total):
    p = float(part.pow(2).sum())
    t = float(total.pow(2).sum())
    return p / t if t > 0 else 0.0


# ── V-CNN ────────────────────────────────────────────────────────────
def load_vcnn():
    ck = torch.load(str(BASE_DIR / VCNN_CKPT), map_location="cpu", weights_only=False)
    m = VoronoiCNN(**ck["model_config"]).to(device)
    m.load_state_dict(ck["model_state"])
    m.eval()
    return m


def predict_vcnn(model, vel_obs, obs_mask, ocean_mask):
    vel_n = ((vel_obs - NM[:, None, None]) / NS[:, None, None]) * ocean_mask[None]
    vi = build_voronoi_input(vel_n, obs_mask, ocean_mask)
    with torch.no_grad():
        p = model(torch.from_numpy(vi).unsqueeze(0).to(device))
    mt = torch.tensor(NM).view(1, 2, 1, 1).to(device)
    st = torch.tensor(NS).view(1, 2, 1, 1).to(device)
    om = torch.from_numpy(ocean_mask).float().to(device).view(1, 1, OCEAN_H, OCEAN_W)
    return ((p * st + mt) * om).squeeze(0).cpu().numpy()


# ── Helmholtz model ─────────────────────────────────────────────────
def load_helmholtz(weights_path):
    net = MyUNet_Helmholtz(n_steps=N_STEPS, time_emb_dim=256)
    ddpm = GaussianDDPM(net, n_steps=N_STEPS,
                        min_beta=0.0001, max_beta=0.02, device=device)
    state = torch.load(str(BASE_DIR / weights_path), map_location="cpu",
                       weights_only=False)
    ddpm.load_state_dict(state)
    ddpm.to(device).eval()
    return ddpm


def single_step(ddpm, gt_ocean, obs_mask, ocean_mask, seed, t_val):
    gt_full = np.zeros((1, 2, FULL_H, FULL_W), dtype=np.float32)
    gt_full[0, :, :OCEAN_H, :OCEAN_W] = gt_ocean * ocean_mask[None]
    known_std = standardizer(
        torch.from_numpy(gt_full).squeeze(0)).unsqueeze(0).to(device)

    border = make_border_mask(device)
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
        tt = torch.full((1, 1), t_val, device=device, dtype=torch.long)
        ab = ddpm.alpha_bars[t_val]
        eps = torch.randn_like(current)
        x_t = ab.sqrt() * current + (1 - ab).sqrt() * eps
        x0_pred = ddpm.network(x_t, tt)
        result = known_std * known_mask + x0_pred * miss_mask

    result_phys = standardizer.unstandardize(result.squeeze(0).cpu())
    return result_phys[:, :OCEAN_H, :OCEAN_W].numpy()


def get_diagnostic(ddpm, gt_std):
    """Extract decomposition diagnostic from last forward pass."""
    net = ddpm.network
    if not hasattr(net, 'last_v_sol') or net.last_v_sol is None:
        # Fallback: recompute from psi/phi
        psi = net.last_psi
        phi = net.last_phi
        psi_padded = F.pad(psi, (0, 1, 0, 1), mode="constant", value=0.0)
        v_sol = MyUNet_Helmholtz._curl_from_streamfunction(psi_padded)
        phi_4d = phi.unsqueeze(1)
        v_irr = net._grad_potential(phi_4d)
    else:
        v_sol = net.last_v_sol
        v_irr = net.last_v_irr

    psi_rms = float(net.last_psi.pow(2).mean().sqrt())
    phi_rms = float(net.last_phi.pow(2).mean().sqrt())

    ocean_sl = (slice(None), slice(None), slice(OCEAN_H), slice(OCEAN_W))
    x0_pred = v_sol + v_irr
    sol_frac = energy_fraction(v_sol[ocean_sl], x0_pred[ocean_sl])
    irr_frac = energy_fraction(v_irr[ocean_sl], x0_pred[ocean_sl])

    # GT decomposition for per-component MSE
    gt_sol, gt_irr = helmholtz_decompose(gt_std)
    om = torch.zeros(1, 1, FULL_H, FULL_W, device=v_sol.device)
    om[:, :, :OCEAN_H, :OCEAN_W] = 1.0
    diff_s = (v_sol - gt_sol) * om
    mse_sol = float(diff_s[:, :, :OCEAN_H, :OCEAN_W].pow(2).mean())
    diff_i = (v_irr - gt_irr) * om
    mse_irr = float(diff_i[:, :, :OCEAN_H, :OCEAN_W].pow(2).mean())
    total = (x0_pred - gt_std) * om
    mse_total = float(total[:, :, :OCEAN_H, :OCEAN_W].pow(2).mean())

    # Solenoidal leakage in v_irr
    v_irr_sol, _ = helmholtz_decompose(v_irr)
    leak = energy_fraction(v_irr_sol[ocean_sl], v_irr[ocean_sl])

    # GT energy split
    gt_sol_frac = energy_fraction(gt_sol[ocean_sl], gt_std[ocean_sl])
    gt_irr_frac = energy_fraction(gt_irr[ocean_sl], gt_std[ocean_sl])

    return {
        "psi_rms": psi_rms, "phi_rms": phi_rms,
        "phi_psi_ratio": phi_rms / (psi_rms + 1e-12),
        "sol_frac": sol_frac, "irr_frac": irr_frac,
        "gt_sol_frac": gt_sol_frac, "gt_irr_frac": gt_irr_frac,
        "mse_sol": mse_sol, "mse_irr": mse_irr, "mse_total": mse_total,
        "leak": leak,
        "v_sol_rms": float(v_sol[ocean_sl].pow(2).mean().sqrt()),
        "v_irr_rms": float(v_irr[ocean_sl].pow(2).mean().sqrt()),
    }


# ══════════════════════════════════════════════════════════════════════
def main():
    print(f"\n{'='*72}")
    print("HELMHOLTZ 3-MODEL COMPARISON + DECOMPOSITION DIAGNOSTIC")
    print(f"Coverage: {args.coverage}%  |  Samples: {args.n_samples}  |  "
          f"t_vals: {args.t_vals}")
    print(f"{'='*72}")

    val = load_val_data()
    ocean_mask = get_ocean_mask(val)
    n_ocean = int(ocean_mask.sum())
    print(f"Ocean cells: {n_ocean}, Obs: {max(1, round(n_ocean * args.coverage / 100))}")

    rng = np.random.default_rng(seed=7777)
    val_indices = rng.choice(val.shape[0], size=min(args.n_samples, val.shape[0]),
                             replace=False)
    val_indices.sort()

    # Load V-CNN
    vcnn = load_vcnn()
    print("V-CNN loaded")

    # Load Helmholtz models (skip missing)
    ddpm_models = {}
    for name, path in MODELS.items():
        full = BASE_DIR / path
        if full.exists():
            ddpm_models[name] = load_helmholtz(path)
            print(f"  {name}: loaded ({path})")
        else:
            print(f"  {name}: MISSING ({path})")

    # ── MSE evaluation ───────────────────────────────────────────────
    # For each model and t_val, collect MSE
    mse_results = {}  # key = (model, t_val) → list of MSE
    vcnn_mses = []
    vor_mses = []

    for name in ddpm_models:
        for tv in args.t_vals:
            mse_results[(name, tv)] = []

    print(f"\n── Per-sample MSE ──")
    header = f"{'#':>3} {'Idx':>5} {'Voronoi':>10} {'V-CNN':>10}"
    for name in ddpm_models:
        for tv in args.t_vals:
            header += f" {name[:6]}@{tv:>3}".rjust(12)
    print(header)
    print("-" * len(header))

    for i, vi in enumerate(val_indices):
        gt = val[vi, :, :OCEAN_H, :OCEAN_W].numpy()
        seed = args.seed + vi
        obs_mask = random_obs_mask(ocean_mask, args.coverage,
                                   np.random.default_rng(seed=seed))

        # Voronoi
        vel_obs = gt * obs_mask[None]
        vor = voronoi_fill(vel_obs, obs_mask, ocean_mask)
        vor_mse = ocean_mse(vor, gt, ocean_mask)
        vor_mses.append(vor_mse)

        # V-CNN
        vcnn_pred = predict_vcnn(vcnn, vel_obs, obs_mask, ocean_mask)
        vcnn_mse = ocean_mse(vcnn_pred, gt, ocean_mask)
        vcnn_mses.append(vcnn_mse)

        row = f"{i+1:>3} {vi:>5} {vor_mse:>10.6f} {vcnn_mse:>10.6f}"

        for name, ddpm in ddpm_models.items():
            for tv in args.t_vals:
                pred = single_step(ddpm, gt, obs_mask, ocean_mask, seed, tv)
                mse = ocean_mse(pred, gt, ocean_mask)
                mse_results[(name, tv)].append(mse)
                row += f" {mse:>11.6f}"

        print(row)

    # ── MSE Summary ──────────────────────────────────────────────────
    vcnn_mean = np.mean(vcnn_mses)
    print(f"\n{'='*72}")
    print(f"MSE SUMMARY — {args.coverage}% coverage, {len(val_indices)} samples")
    print(f"{'='*72}")
    print(f"\n{'Method':<25} {'Mean MSE':>12} {'vs V-CNN':>10}")
    print("-" * 50)
    print(f"{'Voronoi':<25} {np.mean(vor_mses):>12.7f} {np.mean(vor_mses)/vcnn_mean:>9.3f}x")
    print(f"{'V-CNN':<25} {vcnn_mean:>12.7f} {1.000:>9.3f}x")
    for name in ddpm_models:
        best_tv, best_ratio = None, float("inf")
        for tv in args.t_vals:
            m = np.mean(mse_results[(name, tv)])
            r = m / vcnn_mean
            label = f"{name} @t={tv}"
            print(f"{label:<25} {m:>12.7f} {r:>9.3f}x")
            if r < best_ratio:
                best_ratio = r
                best_tv = tv
        print(f"  → best: t={best_tv} ({best_ratio:.3f}x V-CNN)")

    # ── Decomposition diagnostic ─────────────────────────────────────
    print(f"\n{'='*72}")
    print("HELMHOLTZ DECOMPOSITION DIAGNOSTIC (last sample)")
    print(f"{'='*72}")

    # Re-run last sample on each model to get diagnostics
    vi = val_indices[-1]
    gt = val[vi, :, :OCEAN_H, :OCEAN_W].numpy()
    seed = args.seed + vi
    obs_mask = random_obs_mask(ocean_mask, args.coverage,
                               np.random.default_rng(seed=seed))

    gt_full = np.zeros((1, 2, FULL_H, FULL_W), dtype=np.float32)
    gt_full[0, :, :OCEAN_H, :OCEAN_W] = gt * ocean_mask[None]
    gt_std = standardizer(
        torch.from_numpy(gt_full).squeeze(0)).unsqueeze(0).to(device)

    for name, ddpm in ddpm_models.items():
        # Run a forward pass to populate diagnostics
        _ = single_step(ddpm, gt, obs_mask, ocean_mask, seed, args.t_vals[0])
        diag = get_diagnostic(ddpm, gt_std)

        print(f"\n  ── {name} ──")
        print(f"    ψ RMS: {diag['psi_rms']:.4f}  |  φ RMS: {diag['phi_rms']:.4f}  "
              f"|  φ/ψ: {diag['phi_psi_ratio']:.3f}")
        print(f"    v_sol RMS: {diag['v_sol_rms']:.5f}  |  v_irr RMS: {diag['v_irr_rms']:.5f}")
        print(f"    Pred energy: sol={diag['sol_frac']*100:.1f}%  "
              f"irr={diag['irr_frac']*100:.1f}%")
        print(f"    GT energy:   sol={diag['gt_sol_frac']*100:.1f}%  "
              f"irr={diag['gt_irr_frac']*100:.1f}%")
        print(f"    Per-head MSE: v_sol={diag['mse_sol']:.6f}  "
              f"v_irr={diag['mse_irr']:.6f}  total={diag['mse_total']:.6f}")
        print(f"    Sol leakage in v_irr: {diag['leak']*100:.1f}%")

    # Key comparison
    if "baseline" in ddpm_models and "supervised" in ddpm_models:
        print(f"\n── KEY COMPARISON: Cancellation degeneracy fixed? ──")
        _ = single_step(ddpm_models["baseline"], gt, obs_mask, ocean_mask, seed, args.t_vals[0])
        d_base = get_diagnostic(ddpm_models["baseline"], gt_std)
        _ = single_step(ddpm_models["supervised"], gt, obs_mask, ocean_mask, seed, args.t_vals[0])
        d_sup = get_diagnostic(ddpm_models["supervised"], gt_std)
        print(f"  {'Metric':<25} {'Baseline':>12} {'Supervised':>12} {'Improvement':>12}")
        print(f"  {'-'*61}")
        print(f"  {'φ/ψ ratio':<25} {d_base['phi_psi_ratio']:>12.3f} {d_sup['phi_psi_ratio']:>12.3f}")
        print(f"  {'v_sol MSE':<25} {d_base['mse_sol']:>12.6f} {d_sup['mse_sol']:>12.6f} "
              f"{d_base['mse_sol']/max(d_sup['mse_sol'],1e-9):>11.1f}x")
        print(f"  {'v_irr MSE':<25} {d_base['mse_irr']:>12.6f} {d_sup['mse_irr']:>12.6f} "
              f"{d_base['mse_irr']/max(d_sup['mse_irr'],1e-9):>11.1f}x")
        print(f"  {'Total MSE':<25} {d_base['mse_total']:>12.6f} {d_sup['mse_total']:>12.6f} "
              f"{d_base['mse_total']/max(d_sup['mse_total'],1e-9):>11.1f}x")
        print(f"  {'Sol leak in v_irr':<25} {d_base['leak']*100:>11.1f}% {d_sup['leak']*100:>11.1f}%")


if __name__ == "__main__":
    main()
