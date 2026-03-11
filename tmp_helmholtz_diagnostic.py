#!/usr/bin/env python3
"""Helmholtz decomposition diagnostic — investigate φ/ψ balance.

Questions answered:
1. What fraction of predicted velocity energy comes from curl(ψ) vs grad(φ)?
2. How does that compare to the GT solenoidal/irrotational split?
3. Per-component accuracy: is v_sol matching GT_sol? Is v_irr matching GT_irr?
4. Is the model's v_irr "leaking" solenoidal content?

Usage:
    PYTHONPATH=. python3 tmp_helmholtz_diagnostic.py
"""

import argparse
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

# ── Constants ────────────────────────────────────────────────────────
OCEAN_H, OCEAN_W = 44, 94
FULL_H, FULL_W = 64, 128
N_STEPS = 250

U_MEAN, U_STD = -0.06929559429949586, 0.1358005549716049
V_MEAN, V_STD = -0.0323937796117541, 0.08899177232117582
standardizer = ZScoreStandardizer(U_MEAN, U_STD, V_MEAN, V_STD)

HELMHOLTZ_WEIGHTS = (
    "experiments/12_helmholtz_dual_head/helmholtz_baseline/results/"
    "inpaint_gaussian_t250_best_weights.pt"
)

parser = argparse.ArgumentParser()
parser.add_argument("--n-samples", type=int, default=5)
parser.add_argument("--coverage", type=float, default=0.5)
parser.add_argument("--t-val", type=int, default=25)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# ── Helpers ──────────────────────────────────────────────────────────
def load_val_data():
    import pickle
    with open(str(BASE_DIR / "data.pickle"), "rb") as f:
        _, _, test_raw = pickle.load(f)
    test = np.transpose(test_raw, (3, 2, 1, 0)).astype(np.float32)
    test = np.nan_to_num(test, nan=0.0)
    return torch.from_numpy(test)


def get_ocean_mask(val_tensor):
    return (val_tensor[0].abs().sum(dim=0) > 1e-7).float().numpy()


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


def energy(field):
    """RMS energy of a (2, H, W) or (B, 2, H, W) field."""
    return float(field.pow(2).mean().sqrt())


def energy_fraction(part, total):
    """Fraction of energy: ||part||² / ||total||²."""
    p = float(part.pow(2).sum())
    t = float(total.pow(2).sum())
    return p / t if t > 0 else 0.0


def make_border_mask(shape, dev):
    m = torch.zeros(1, 1, shape[2], shape[3], device=dev)
    m[:, :, :OCEAN_H, :OCEAN_W] = 1.0
    return m


# ── Recompute v_sol and v_irr from stored potentials ─────────────────
def decompose_prediction(net):
    """After a forward pass, recompute v_sol and v_irr from last_psi/last_phi."""
    psi = net.last_psi  # (B, H, W)
    phi = net.last_phi  # (B, H, W)

    # Solenoidal: curl(ψ) — same as in forward()
    psi_padded = F.pad(psi, (0, 1, 0, 1), mode="constant", value=0.0)
    v_sol = MyUNet_Helmholtz._curl_from_streamfunction(psi_padded)

    # Irrotational: grad(φ) — same as in forward()
    phi_4d = phi.unsqueeze(1)  # (B, 1, H, W)
    v_irr = net._grad_potential(phi_4d)

    return v_sol, v_irr


# ══════════════════════════════════════════════════════════════════════
def main():
    print(f"Device: {device}")
    print(f"\n{'='*72}")
    print(f"HELMHOLTZ DECOMPOSITION DIAGNOSTIC")
    print(f"Coverage: {args.coverage}%  |  Samples: {args.n_samples}  |  t_val: {args.t_val}")
    print(f"{'='*72}")

    val = load_val_data()
    ocean_mask = get_ocean_mask(val)

    # Load model
    net = MyUNet_Helmholtz(n_steps=N_STEPS, time_emb_dim=256,
                           n_stage_tokens=0, self_cond_channels=0)
    ddpm = GaussianDDPM(net, n_steps=N_STEPS,
                        min_beta=0.0001, max_beta=0.02, device=device)
    state = torch.load(str(BASE_DIR / HELMHOLTZ_WEIGHTS), map_location="cpu",
                       weights_only=False)
    ddpm.load_state_dict(state)
    ddpm.to(device)
    ddpm.eval()
    print(f"Loaded: {HELMHOLTZ_WEIGHTS}")

    rng = np.random.default_rng(seed=7777)
    val_indices = rng.choice(val.shape[0], size=min(args.n_samples, val.shape[0]),
                             replace=False)
    val_indices.sort()

    # Accumulators
    gt_sol_frac_list = []
    gt_irr_frac_list = []
    pred_sol_frac_list = []
    pred_irr_frac_list = []
    sol_on_sol_mse = []     # MSE of model's v_sol vs GT's solenoidal
    irr_on_irr_mse = []     # MSE of model's v_irr vs GT's irrotational
    sol_leak_frac = []       # solenoidal content in model's v_irr
    irr_leak_frac = []       # irrotational content in model's v_sol
    total_mse_list = []
    psi_rms_list = []
    phi_rms_list = []
    v_sol_rms_list = []
    v_irr_rms_list = []

    print(f"\n{'#':>3} {'Idx':>5}  {'GT sol%':>7} {'GT irr%':>7}  "
          f"{'Pred sol%':>9} {'Pred irr%':>9}  "
          f"{'ψ RMS':>8} {'φ RMS':>8}  "
          f"{'v_sol RMS':>9} {'v_irr RMS':>9}  "
          f"{'sol→sol':>8} {'irr→irr':>8}  "
          f"{'irr leak':>8}")
    print("-" * 120)

    for i, vi in enumerate(val_indices):
        gt = val[vi, :, :OCEAN_H, :OCEAN_W].numpy()
        seed = args.seed + vi
        obs_mask = random_obs_mask(ocean_mask, args.coverage,
                                   np.random.default_rng(seed=seed))

        # ── Ground truth Helmholtz decomposition ──
        gt_full = np.zeros((1, 2, FULL_H, FULL_W), dtype=np.float32)
        gt_full[0, :, :OCEAN_H, :OCEAN_W] = gt * ocean_mask[None]
        gt_std = standardizer(
            torch.from_numpy(gt_full).squeeze(0)).unsqueeze(0).to(device)
        gt_sol, gt_irr = helmholtz_decompose(gt_std)
        gt_sol_e = energy_fraction(gt_sol[:, :, :OCEAN_H, :OCEAN_W],
                                   gt_std[:, :, :OCEAN_H, :OCEAN_W])
        gt_irr_e = energy_fraction(gt_irr[:, :, :OCEAN_H, :OCEAN_W],
                                   gt_std[:, :, :OCEAN_H, :OCEAN_W])

        # ── Model prediction ──
        known_std = gt_std.clone()
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

        vel_obs = gt * obs_mask[None]
        vor_fill = voronoi_fill(vel_obs, obs_mask, ocean_mask)
        vor_full = np.zeros((1, 2, FULL_H, FULL_W), dtype=np.float32)
        vor_full[0, :, :OCEAN_H, :OCEAN_W] = vor_fill
        vor_std = standardizer(
            torch.from_numpy(vor_full).squeeze(0)).unsqueeze(0).to(device)
        current = known_std * known_mask + vor_std * miss_mask

        torch.manual_seed(seed)
        with torch.no_grad():
            time_tensor = torch.full((1, 1), args.t_val, device=device, dtype=torch.long)
            alpha_bar_t = ddpm.alpha_bars[args.t_val]
            eps = torch.randn_like(current)
            x_t = alpha_bar_t.sqrt() * current + (1 - alpha_bar_t).sqrt() * eps
            x0_pred = ddpm.network(x_t, time_tensor)

        # ── Extract model's v_sol and v_irr ──
        v_sol, v_irr = decompose_prediction(ddpm.network)
        x0_combined = known_std * known_mask + x0_pred * miss_mask

        # Predicted energy fractions (over ocean region)
        ocean_sl = (slice(None), slice(None), slice(OCEAN_H), slice(OCEAN_W))
        pred_sol_e = energy_fraction(v_sol[ocean_sl], x0_pred[ocean_sl])
        pred_irr_e = energy_fraction(v_irr[ocean_sl], x0_pred[ocean_sl])

        # ── Cross-decomposition: does model's v_irr contain solenoidal content? ──
        # FFT-decompose the model's v_irr output
        v_irr_sol, v_irr_irr = helmholtz_decompose(v_irr)
        leak = energy_fraction(v_irr_sol[ocean_sl], v_irr[ocean_sl])

        # ── Per-component MSE (standardized space, ocean only) ──
        om_t = torch.from_numpy(ocean_mask).float().to(device).unsqueeze(0).unsqueeze(0)
        # model v_sol vs GT solenoidal (over ocean)
        diff_sol = (v_sol - gt_sol)[:, :, :OCEAN_H, :OCEAN_W] * om_t
        mse_sol = float(diff_sol.pow(2).sum() / om_t.sum() / 2)
        # model v_irr vs GT irrotational (over ocean)
        diff_irr = (v_irr - gt_irr)[:, :, :OCEAN_H, :OCEAN_W] * om_t
        mse_irr = float(diff_irr.pow(2).sum() / om_t.sum() / 2)

        # Total MSE
        diff_total = (x0_combined - gt_std)[:, :, :OCEAN_H, :OCEAN_W] * om_t
        mse_total = float(diff_total.pow(2).sum() / om_t.sum() / 2)

        # Raw potential RMS
        psi_rms = float(ddpm.network.last_psi.pow(2).mean().sqrt())
        phi_rms = float(ddpm.network.last_phi.pow(2).mean().sqrt())
        vs_rms = energy(v_sol[ocean_sl])
        vi_rms = energy(v_irr[ocean_sl])

        # Accumulate
        gt_sol_frac_list.append(gt_sol_e)
        gt_irr_frac_list.append(gt_irr_e)
        pred_sol_frac_list.append(pred_sol_e)
        pred_irr_frac_list.append(pred_irr_e)
        sol_on_sol_mse.append(mse_sol)
        irr_on_irr_mse.append(mse_irr)
        sol_leak_frac.append(leak)
        total_mse_list.append(mse_total)
        psi_rms_list.append(psi_rms)
        phi_rms_list.append(phi_rms)
        v_sol_rms_list.append(vs_rms)
        v_irr_rms_list.append(vi_rms)

        print(f"{i+1:>3} {vi:>5}  {gt_sol_e*100:>6.1f}% {gt_irr_e*100:>6.1f}%  "
              f"{pred_sol_e*100:>8.1f}% {pred_irr_e*100:>8.1f}%  "
              f"{psi_rms:>8.3f} {phi_rms:>8.3f}  "
              f"{vs_rms:>9.4f} {vi_rms:>9.4f}  "
              f"{mse_sol:>8.5f} {mse_irr:>8.5f}  "
              f"{leak*100:>7.1f}%")

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("SUMMARY")
    print(f"{'='*72}")

    print(f"\n  Ground truth energy split (standardized space):")
    print(f"    Solenoidal:    {np.mean(gt_sol_frac_list)*100:.1f}%")
    print(f"    Irrotational:  {np.mean(gt_irr_frac_list)*100:.1f}%")

    print(f"\n  Model prediction energy split:")
    print(f"    v_sol (curl ψ):   {np.mean(pred_sol_frac_list)*100:.1f}%")
    print(f"    v_irr (grad φ):   {np.mean(pred_irr_frac_list)*100:.1f}%")

    print(f"\n  Raw potential magnitudes:")
    print(f"    ψ RMS: {np.mean(psi_rms_list):.4f}")
    print(f"    φ RMS: {np.mean(phi_rms_list):.4f}")
    print(f"    φ/ψ ratio: {np.mean(phi_rms_list)/np.mean(psi_rms_list):.3f}")

    print(f"\n  Velocity magnitudes (ocean only):")
    print(f"    v_sol RMS: {np.mean(v_sol_rms_list):.5f}")
    print(f"    v_irr RMS: {np.mean(v_irr_rms_list):.5f}")
    print(f"    v_irr/v_sol ratio: {np.mean(v_irr_rms_list)/np.mean(v_sol_rms_list):.3f}")

    print(f"\n  Per-component MSE (standardized, ocean):")
    print(f"    v_sol vs GT_sol: {np.mean(sol_on_sol_mse):.6f}")
    print(f"    v_irr vs GT_irr: {np.mean(irr_on_irr_mse):.6f}")
    print(f"    Total MSE:       {np.mean(total_mse_list):.6f}")

    print(f"\n  Solenoidal leakage into v_irr: {np.mean(sol_leak_frac)*100:.1f}%")
    print(f"    (ideal = 0%: grad(φ) should be purely irrotational)")
    print(f"    (nonzero means central-diff gradient ≠ exact spectral projection)")

    # Interpretation
    print(f"\n{'='*72}")
    print("INTERPRETATION")
    print(f"{'='*72}")
    gt_sol_pct = np.mean(gt_sol_frac_list) * 100
    pred_sol_pct = np.mean(pred_sol_frac_list) * 100
    if pred_sol_pct < gt_sol_pct * 0.7:
        print(f"  ⚠  Model under-utilizes ψ head: {pred_sol_pct:.0f}% sol vs {gt_sol_pct:.0f}% GT")
        print(f"     The curl(ψ) pathway isn't capturing enough of the solenoidal signal.")
    elif pred_sol_pct > gt_sol_pct * 1.3:
        print(f"  ⚠  Model over-utilizes ψ head: {pred_sol_pct:.0f}% sol vs {gt_sol_pct:.0f}% GT")
    else:
        print(f"  ✓  Energy split roughly matches GT ({pred_sol_pct:.0f}% sol vs {gt_sol_pct:.0f}% GT)")

    leak_pct = np.mean(sol_leak_frac) * 100
    if leak_pct > 10:
        print(f"  ⚠  Significant solenoidal leakage ({leak_pct:.0f}%) in v_irr")
        print(f"     Central-diff gradient doesn't exactly zero out curl — some")
        print(f"     solenoidal content routes through φ, defeating the decomposition.")
    else:
        print(f"  ✓  Low solenoidal leakage ({leak_pct:.1f}%) in v_irr — decomposition is clean")


if __name__ == "__main__":
    main()
