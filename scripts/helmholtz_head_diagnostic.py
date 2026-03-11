#!/usr/bin/env python3
"""Helmholtz head diagnostics: cancellation, orthogonality, energy partition.

Runs N samples through the split-decoder model at the best single-step t,
captures v_sol, v_irr, and computes detailed decomposition health metrics.

Usage:
    PYTHONPATH=. python scripts/helmholtz_head_diagnostic.py
    PYTHONPATH=. python scripts/helmholtz_head_diagnostic.py --n-samples 20
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

BASE_DIR = Path(__file__).resolve().parent.parent

from ddpm.neural_networks.ddpm import GaussianDDPM
from ddpm.neural_networks.unets.unet_helmholtz_split import MyUNet_Helmholtz_Split
from ddpm.helper_functions.standardize_data import ZScoreStandardizer, UnifiedZScoreStandardizer
from ddpm.utils.noise_utils import HelmholtzMatchedNoise

# ── Constants ────────────────────────────────────────────────────────
OCEAN_H, OCEAN_W = 44, 94
FULL_H, FULL_W = 64, 128
N_STEPS = 250

U_MEAN, U_STD = -0.06929559429949586, 0.1358005549716049
V_MEAN, V_STD = -0.0323937796117541, 0.08899177232117582
SHARED_MEAN, SHARED_STD = -0.05084468695562498, 0.11479844598042026

DEFAULT_WEIGHTS = (
    "experiments/12_helmholtz_dual_head/helmholtz_split_decoder/results/"
    "inpaint_gaussian_t250_best_weights.pt"
)

parser = argparse.ArgumentParser()
parser.add_argument("--n-samples", type=int, default=10)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS)
parser.add_argument("--t-val", type=int, default=25)
parser.add_argument("--coverage", type=float, default=0.5)
args = parser.parse_args()

# ── Auto-detect standardizer and noise type ───────────────────────
use_matched_noise = False
cfg_path = Path(args.weights).parent / "resolved_config.yaml"
if cfg_path.exists():
    import yaml
    with open(cfg_path) as f:
        _cfg = yaml.safe_load(f)
    if _cfg.get("noise_function") == "helmholtz_matched":
        use_matched_noise = True
        print("Auto-detected helmholtz_matched noise → using matched inference")
    if _cfg.get("noise_function") in ("helmholtz_matched", "div_free",
        "spectral_div_free", "forward_diff_div_free", "fwd_diff_eq_divfree"):
        standardizer = UnifiedZScoreStandardizer(SHARED_MEAN, SHARED_STD)
        print("Using unified standardizer")
    else:
        standardizer = ZScoreStandardizer(U_MEAN, U_STD, V_MEAN, V_STD)
else:
    standardizer = ZScoreStandardizer(U_MEAN, U_STD, V_MEAN, V_STD)

_matched_gen = HelmholtzMatchedNoise() if use_matched_noise else None

def noise_fn(shape, dev):
    if _matched_gen is not None:
        return _matched_gen.generate(shape, device=dev)
    return torch.randn(shape, device=dev)

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# ── Utilities ────────────────────────────────────────────────────────
def load_val_data():
    import pickle
    with open(str(BASE_DIR / "data.pickle"), "rb") as f:
        _, _, test_raw = pickle.load(f)
    test = np.transpose(test_raw, (3, 2, 1, 0)).astype(np.float32)
    return torch.from_numpy(np.nan_to_num(test, nan=0.0))


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


def make_border_mask(dev):
    m = torch.zeros(1, 1, FULL_H, FULL_W, device=dev)
    m[:, :, :OCEAN_H, :OCEAN_W] = 1.0
    return m


def finite_div(u, v):
    du_dx = np.zeros_like(u)
    dv_dy = np.zeros_like(v)
    du_dx[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / 2.0
    dv_dy[1:-1, :] = (v[2:, :] - v[:-2, :]) / 2.0
    return du_dx + dv_dy


def finite_curl(u, v):
    """Vorticity = dv/dx - du/dy (scalar)."""
    dv_dx = np.zeros_like(v)
    du_dy = np.zeros_like(u)
    dv_dx[:, 1:-1] = (v[:, 2:] - v[:, :-2]) / 2.0
    du_dy[1:-1, :] = (u[2:, :] - u[:-2, :]) / 2.0
    return dv_dx - du_dy


# ── Main ─────────────────────────────────────────────────────────────
def main():
    val = load_val_data()
    ocean_mask = get_ocean_mask(val)
    ocean_b = ocean_mask.astype(bool)

    rng = np.random.default_rng(seed=7777)
    val_indices = rng.choice(val.shape[0], size=min(args.n_samples, val.shape[0]),
                             replace=False)
    val_indices.sort()

    # Load model
    net = MyUNet_Helmholtz_Split(n_steps=N_STEPS, time_emb_dim=256,
                                 n_stage_tokens=0, self_cond_channels=0)
    ddpm = GaussianDDPM(net, n_steps=N_STEPS,
                        min_beta=0.0001, max_beta=0.02, device=device)
    state = torch.load(str(BASE_DIR / args.weights), map_location="cpu",
                       weights_only=False)
    ddpm.load_state_dict(state)
    ddpm.to(device)
    ddpm.eval()

    border = make_border_mask(device)

    # Accumulators
    stats = {k: [] for k in [
        "psi_rms", "phi_rms", "v_sol_rms", "v_irr_rms", "total_rms",
        "cancel_ratio", "dot_cos", "sol_div_rms", "irr_curl_rms",
        "sol_energy_frac", "irr_energy_frac", "cross_energy",
        "gt_sol_frac",  # ground truth solenoidal fraction for comparison
    ]}

    # For one example plot
    example_data = None

    print(f"\n{'='*72}")
    print("HELMHOLTZ HEAD DIAGNOSTIC")
    print(f"Weights: {args.weights}")
    print(f"t={args.t_val}, coverage={args.coverage}%, {len(val_indices)} samples")
    print(f"{'='*72}\n")

    header = (f"{'#':>3} {'psi':>8} {'phi':>8} {'v_sol':>8} {'v_irr':>8} "
              f"{'total':>8} {'cancel':>8} {'cos':>8} {'sol_div':>8} "
              f"{'irr_crl':>8} {'sol%':>6} {'irr%':>6}")
    print(header)
    print("-" * len(header))

    for i, vi in enumerate(val_indices):
        gt = val[vi, :, :OCEAN_H, :OCEAN_W].numpy()
        seed = args.seed + vi
        obs_mask = random_obs_mask(ocean_mask, args.coverage,
                                   np.random.default_rng(seed=seed))

        # Build inputs
        gt_full = np.zeros((1, 2, FULL_H, FULL_W), dtype=np.float32)
        gt_full[0, :, :OCEAN_H, :OCEAN_W] = gt * ocean_mask[None]
        known_std = standardizer(
            torch.from_numpy(gt_full).squeeze(0)).unsqueeze(0).to(device)

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

        # Forward pass
        torch.manual_seed(seed)
        with torch.no_grad():
            t_tensor = torch.full((1, 1), args.t_val, device=device, dtype=torch.long)
            alpha_bar_t = ddpm.alpha_bars[args.t_val]
            eps = noise_fn(current.shape, device)
            x_t = alpha_bar_t.sqrt() * current + (1 - alpha_bar_t).sqrt() * eps
            x0_pred = ddpm.network(x_t, t_tensor)

        # Extract head outputs (standardized space, full grid)
        psi = net.last_psi.cpu().numpy()[0]        # (H, W)
        phi = net.last_phi.cpu().numpy()[0]        # (H, W)
        v_sol = net.last_v_sol.cpu().numpy()[0]    # (2, H, W)
        v_irr = net.last_v_irr.cpu().numpy()[0]    # (2, H, W)
        total = x0_pred.cpu().numpy()[0]           # (2, H, W)

        # Crop to ocean
        v_sol_o = v_sol[:, :OCEAN_H, :OCEAN_W]
        v_irr_o = v_irr[:, :OCEAN_H, :OCEAN_W]
        total_o = total[:, :OCEAN_H, :OCEAN_W]

        # ── RMS magnitudes (ocean only) ──
        sol_rms = np.sqrt((v_sol_o[:, ocean_b] ** 2).mean())
        irr_rms = np.sqrt((v_irr_o[:, ocean_b] ** 2).mean())
        tot_rms = np.sqrt((total_o[:, ocean_b] ** 2).mean())
        psi_rms = np.sqrt((psi[:OCEAN_H, :OCEAN_W][ocean_b] ** 2).mean())
        phi_rms = np.sqrt((phi[:OCEAN_H, :OCEAN_W][ocean_b] ** 2).mean())

        # ── Cancellation ratio ──
        cancel = (sol_rms + irr_rms) / (tot_rms + 1e-12)

        # ── Cosine similarity (are they anti-aligned?) ──
        sol_flat = v_sol_o[:, ocean_b].flatten()
        irr_flat = v_irr_o[:, ocean_b].flatten()
        dot = np.dot(sol_flat, irr_flat)
        cos_sim = dot / (np.linalg.norm(sol_flat) * np.linalg.norm(irr_flat) + 1e-12)

        # ── Physical consistency: div of v_sol, curl of v_irr ──
        sol_div = finite_div(v_sol_o[0], v_sol_o[1])
        irr_curl = finite_curl(v_irr_o[0], v_irr_o[1])
        sol_div_rms = np.sqrt((sol_div[ocean_b] ** 2).mean())
        irr_curl_rms = np.sqrt((irr_curl[ocean_b] ** 2).mean())

        # ── Energy partition ──
        sol_energy = (v_sol_o[:, ocean_b] ** 2).sum()
        irr_energy = (v_irr_o[:, ocean_b] ** 2).sum()
        total_energy = sol_energy + irr_energy
        sol_frac = sol_energy / (total_energy + 1e-12)
        irr_frac = irr_energy / (total_energy + 1e-12)

        # Cross energy: how much of v_sol projects onto v_irr
        cross = abs(dot) / (total_energy + 1e-12)

        # ── Ground truth solenoidal fraction (for comparison) ──
        gt_std = standardizer(torch.from_numpy(
            gt * ocean_mask[None]).unsqueeze(0).squeeze(0)).numpy()
        gt_u, gt_v = gt_std[0, :OCEAN_H, :OCEAN_W], gt_std[1, :OCEAN_H, :OCEAN_W]
        gt_curl = finite_curl(gt_u, gt_v)
        gt_div = finite_div(gt_u, gt_v)
        # Rough solenoidal fraction from curl²/(curl²+div²)
        curl_e = (gt_curl[ocean_b] ** 2).sum()
        div_e = (gt_div[ocean_b] ** 2).sum()
        gt_sol_frac = curl_e / (curl_e + div_e + 1e-12)

        # Store
        stats["psi_rms"].append(psi_rms)
        stats["phi_rms"].append(phi_rms)
        stats["v_sol_rms"].append(sol_rms)
        stats["v_irr_rms"].append(irr_rms)
        stats["total_rms"].append(tot_rms)
        stats["cancel_ratio"].append(cancel)
        stats["dot_cos"].append(cos_sim)
        stats["sol_div_rms"].append(sol_div_rms)
        stats["irr_curl_rms"].append(irr_curl_rms)
        stats["sol_energy_frac"].append(sol_frac)
        stats["irr_energy_frac"].append(irr_frac)
        stats["cross_energy"].append(cross)
        stats["gt_sol_frac"].append(gt_sol_frac)

        if example_data is None:
            example_data = {
                "gt": gt, "v_sol": v_sol_o, "v_irr": v_irr_o,
                "total": total_o, "psi": psi[:OCEAN_H, :OCEAN_W],
                "phi": phi[:OCEAN_H, :OCEAN_W],
                "ocean_mask": ocean_mask, "obs_mask": obs_mask,
            }

        print(f"{i+1:>3} {psi_rms:>8.4f} {phi_rms:>8.4f} {sol_rms:>8.4f} "
              f"{irr_rms:>8.4f} {tot_rms:>8.4f} {cancel:>8.2f} {cos_sim:>8.4f} "
              f"{sol_div_rms:>8.5f} {irr_curl_rms:>8.5f} {sol_frac:>5.1%} {irr_frac:>5.1%}")

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("SUMMARY (mean ± std over samples)")
    print(f"{'='*72}")
    for k in stats:
        arr = np.array(stats[k])
        print(f"  {k:<20}: {arr.mean():.5f} ± {arr.std():.5f}")

    print(f"\n── INTERPRETATION ──")
    cr = np.mean(stats["cancel_ratio"])
    cs = np.mean(stats["dot_cos"])
    sf = np.mean(stats["sol_energy_frac"])
    gtsf = np.mean(stats["gt_sol_frac"])
    sdiv = np.mean(stats["sol_div_rms"])
    icrl = np.mean(stats["irr_curl_rms"])
    svr = np.mean(stats["v_sol_rms"])
    ivr = np.mean(stats["v_irr_rms"])

    print(f"  Cancellation ratio: {cr:.2f}x  ", end="")
    if cr < 1.5:
        print("✓ Excellent — minimal cancellation")
    elif cr < 2.5:
        print("~ Moderate — some cancellation")
    else:
        print("⚠ High — heads are fighting each other")

    print(f"  Cosine similarity:  {cs:.4f}  ", end="")
    if abs(cs) < 0.3:
        print("✓ Near-orthogonal (healthy)")
    elif cs < -0.3:
        print("⚠ Anti-aligned (cancelling)")
    else:
        print("~ Co-aligned (additive)")

    print(f"  Energy split:  sol={sf:.1%}  irr={1-sf:.1%}  ", end="")
    print(f"(GT ratio: sol≈{gtsf:.1%})")

    print(f"  div(v_sol) RMS: {sdiv:.5f}  (should be ~0 if truly solenoidal)")
    print(f"  curl(v_irr) RMS: {icrl:.5f}  (should be ~0 if truly irrotational)")
    print(f"  v_sol/v_irr ratio: {svr/ivr:.2f}  (GT ≈ {gtsf/(1-gtsf+1e-12):.2f})")

    # ── Visualization ────────────────────────────────────────────────
    if example_data is not None:
        d = example_data
        om = d["ocean_mask"].astype(bool)

        fig, axes = plt.subplots(3, 4, figsize=(20, 12))
        fig.suptitle(f"Helmholtz Split Decoder — Head Diagnostics\n"
                     f"cancel={cr:.2f}x, cos={cs:.3f}, sol:irr energy = "
                     f"{sf:.0%}:{1-sf:.0%}", fontsize=14)

        def plot_field(ax, field, title, cmap="RdBu_r"):
            masked = np.where(om, field, np.nan)
            vmax = max(abs(np.nanmin(masked)), abs(np.nanmax(masked)))
            im = ax.imshow(masked, origin="lower", cmap=cmap,
                          vmin=-vmax, vmax=vmax, aspect="auto")
            ax.set_title(title, fontsize=10)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        def plot_scalar(ax, field, title, cmap="viridis"):
            masked = np.where(om, field, np.nan)
            im = ax.imshow(masked, origin="lower", cmap=cmap, aspect="auto")
            ax.set_title(title, fontsize=10)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Row 1: u-component
        gt_std = standardizer(torch.from_numpy(
            d["gt"] * d["ocean_mask"][None]).unsqueeze(0).squeeze(0)).numpy()
        plot_field(axes[0, 0], gt_std[0, :OCEAN_H, :OCEAN_W], "GT (std) u")
        plot_field(axes[0, 1], d["v_sol"][0], "v_sol u")
        plot_field(axes[0, 2], d["v_irr"][0], "v_irr u")
        plot_field(axes[0, 3], d["total"][0], "total u (sol+irr)")

        # Row 2: v-component
        plot_field(axes[1, 0], gt_std[1, :OCEAN_H, :OCEAN_W], "GT (std) v")
        plot_field(axes[1, 1], d["v_sol"][1], "v_sol v")
        plot_field(axes[1, 2], d["v_irr"][1], "v_irr v")
        plot_field(axes[1, 3], d["total"][1], "total v (sol+irr)")

        # Row 3: potentials + diagnostic fields
        plot_scalar(axes[2, 0], d["psi"], "ψ (streamfunction)", cmap="RdBu_r")
        plot_scalar(axes[2, 1], d["phi"], "φ (velocity potential)", cmap="RdBu_r")

        # Divergence of v_sol (should be 0)
        sol_div = finite_div(d["v_sol"][0], d["v_sol"][1])
        plot_field(axes[2, 2], sol_div, f"div(v_sol) — RMS={sdiv:.4f}")

        # Curl of v_irr (should be 0)
        irr_curl = finite_curl(d["v_irr"][0], d["v_irr"][1])
        plot_field(axes[2, 3], irr_curl, f"curl(v_irr) — RMS={icrl:.4f}")

        plt.tight_layout()
        out_path = Path(args.weights).parent / "helmholtz_head_diagnostic.png"
        plt.savefig(str(BASE_DIR / out_path), dpi=150, bbox_inches="tight")
        print(f"\nPlot saved to {out_path}")
        plt.close()

    # ── Save data ────────────────────────────────────────────────────
    out_path = Path(args.weights).parent / "helmholtz_head_diagnostic.pt"
    torch.save({"stats": stats, "args": vars(args)}, str(BASE_DIR / out_path))
    print(f"Data saved to {out_path}")


if __name__ == "__main__":
    main()
