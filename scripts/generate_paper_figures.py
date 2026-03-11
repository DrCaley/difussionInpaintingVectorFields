#!/usr/bin/env python3
"""Generate publication-quality comparison figures for the paper.

Produces:
  1. Multi-panel velocity field quiver plots (GT / GP / GP-CNN / Composite / VCNN)
     with Gamma1 eddy contours overlaid — for selected eddy & non-eddy samples.
  2. Gamma1 heatmap comparison panels.
  3. Observation mask + sensor location visualization.

Each figure: one row per sample, one column per method.

Usage:
    PYTHONPATH=. python scripts/generate_paper_figures.py [--reveal-pct 5.0]
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba, Normalize
from matplotlib import cm

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from scripts.voronoi_cnn_model import VoronoiCNN, build_voronoi_input
from ddpm.neural_networks.ddpm import GaussianDDPM
from ddpm.neural_networks.unets.unet_film_attn import MyUNet_FiLM_Attn
from ddpm.helper_functions.standardize_data import ZScoreStandardizer
from ddpm.helper_functions.interpolation_tool import gp_fill
from ddpm.utils.eddy_detection import detect_eddies_gamma

# ──────────────────────────────────────────────────────────────────────
# Constants (same as eddy_compare_composite_vcnn.py)
# ──────────────────────────────────────────────────────────────────────
OCEAN_H, OCEAN_W = 44, 94
FULL_H, FULL_W = 64, 128
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

EDDY_PARAMS = dict(
    radius=8, gamma_threshold=0.65, min_area=25, shore_buffer=2,
    smooth_sigma=2.0, min_mean_speed_ratio=0.3, min_vorticity=0.03,
)

DDPM_WEIGHT_PATH = "experiments/06_gp_forward/gp_conditioned/results/inpaint_gaussian_t250_best_ema_weights.pt"
GP_CNN_CKPT = "results/gp_cnn_diverse/gp_cnn_diverse_best.pt"
VCNN_CKPT = "results/voronoi_cnn/voronoi_cnn_best.pt"
DDPM_EVAL_PT = "results/eddy_balanced_eval/bulk_eval_eddy_balanced_100.pt"
OUT_DIR = Path("paper/figures")

# Plot style
LAND_COLOR = (0.22, 0.55, 0.24, 0.7)
QUIVER_STEP = 2
QUIVER_COLOR = "#222222"
QUIVER_ALPHA = 0.75
QUIVER_WIDTH = 0.003
QUIVER_HEADWIDTH = 2.5
QUIVER_HEADLEN = 3.0
EDDY_CONTOUR_COLOR = "magenta"
EDDY_CENTER_COLOR = "magenta"
FP_CONTOUR_COLOR = "red"
SPEED_CMAP = "viridis"
GAMMA_CMAP = "RdBu_r"


# ──────────────────────────────────────────────────────────────────────
# Data loading (same as comparison script)
# ──────────────────────────────────────────────────────────────────────
def load_pickle_data(path="data.pickle"):
    with open(path, "rb") as f:
        train_np, val_np, _test_np = pickle.load(f)
    def to_tensor(arr):
        t = torch.from_numpy(np.ascontiguousarray(arr)).float()
        t = t.permute(3, 2, 1, 0)
        t = torch.nan_to_num(t, nan=0.0)
        return t
    return to_tensor(train_np), to_tensor(val_np)


def generate_random_mask(ocean_mask, reveal_pct, rng):
    ocean_indices = np.argwhere(ocean_mask > 0.5)
    n_ocean = len(ocean_indices)
    n_reveal = max(1, round(n_ocean * reveal_pct / 100.0))
    chosen = rng.choice(n_ocean, size=n_reveal, replace=False)
    obs_mask = np.zeros((OCEAN_H, OCEAN_W), dtype=np.float32)
    for idx in chosen:
        r, c = ocean_indices[idx]
        obs_mask[r, c] = 1.0
    return obs_mask


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


def load_ddpm(device):
    network = MyUNet_FiLM_Attn(n_steps=N_STEPS, time_emb_dim=256, in_channels=5)
    ddpm = GaussianDDPM(network, n_steps=N_STEPS, min_beta=0.0001, max_beta=0.02, device=device)
    ddpm.load_state_dict(torch.load(DDPM_WEIGHT_PATH, map_location="cpu", weights_only=False))
    ddpm = ddpm.to(device); ddpm.eval()
    return ddpm


def load_gp_cnn(device):
    ckpt = torch.load(GP_CNN_CKPT, map_location="cpu", weights_only=False)
    cfg = ckpt["model_config"]
    model = VoronoiCNN(in_channels=cfg["in_channels"], out_channels=cfg["out_channels"],
                       base_ch=cfg.get("base_ch", 32), depth=cfg.get("depth", 3)).to(device)
    model.load_state_dict(ckpt["model_state"]); model.eval()
    return model, ckpt


def load_vcnn(device):
    ckpt = torch.load(VCNN_CKPT, map_location="cpu", weights_only=False)
    cfg = ckpt["model_config"]
    model = VoronoiCNN(**cfg).to(device)
    model.load_state_dict(ckpt["model_state"]); model.eval()
    return model, ckpt


def gp_cnn_predict(model, gp_mean_raw, gp_var_raw, obs_mask, ocean_mask,
                   norm_mean, norm_std, device):
    gp_mean_norm = (gp_mean_raw - norm_mean[:, None, None]) / norm_std[:, None, None]
    gp_mean_norm *= ocean_mask[None, :, :]
    gp_mean_t = torch.from_numpy(gp_mean_norm.astype(np.float32))
    gp_std_raw = np.sqrt(np.clip(gp_var_raw, 0, None))
    gp_std_norm = gp_std_raw / norm_std[:, None, None]
    gp_std_norm *= ocean_mask[None, :, :]
    gp_std_t = torch.from_numpy(gp_std_norm.astype(np.float32))
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


def vcnn_predict(model, vel_phys, obs_mask, ocean_mask, norm_mean, norm_std, device):
    vel_n = (vel_phys - norm_mean[:, None, None]) / norm_std[:, None, None]
    vel_n *= ocean_mask[None, :, :]
    voronoi_in = build_voronoi_input(vel_n, obs_mask, ocean_mask)
    voronoi_t = torch.from_numpy(voronoi_in).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_n = model(voronoi_t)
    mean_t = torch.tensor(norm_mean, dtype=torch.float32).view(1, 2, 1, 1).to(device)
    std_t = torch.tensor(norm_std, dtype=torch.float32).view(1, 2, 1, 1).to(device)
    pred_phys = pred_n * std_t + mean_t
    ocean_t = torch.from_numpy(ocean_mask).float().to(device).unsqueeze(0).unsqueeze(0)
    pred_phys = pred_phys * ocean_t
    return pred_phys.squeeze(0).cpu()


def ddpm_ensemble(ddpm, cond_std_field, missing_mask_1ch, t_val, n_ens, device):
    preds = []
    for k in range(n_ens):
        torch.manual_seed(42 + k * 1000)
        alpha_bar = ddpm.alpha_bars[t_val].to(device)
        noise = torch.randn_like(cond_std_field)
        noisy = alpha_bar.sqrt() * cond_std_field + (1 - alpha_bar).sqrt() * noise
        x_cond = torch.cat([noisy, missing_mask_1ch, cond_std_field], dim=1)
        time_tensor = torch.full((1, 1), t_val, device=device, dtype=torch.long)
        with torch.no_grad():
            pred = ddpm.network(x_cond, time_tensor)
        preds.append(pred)
    stack = torch.stack(preds, dim=0)
    return stack.mean(dim=0), stack.std(dim=0)


def compute_gp_var_weight(gp_var, ocean_mask):
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


def build_ddpm_missing_mask(obs_mask, ocean_mask):
    from ddpm.helper_functions.masks.border_mask import BorderMaskGenerator
    border_gen = BorderMaskGenerator()
    raw_mask = np.ones((FULL_H, FULL_W), dtype=np.float32)
    raw_mask[:OCEAN_H, :OCEAN_W] -= obs_mask
    raw_mask[:OCEAN_H, :OCEAN_W] *= ocean_mask
    raw_mask[OCEAN_H:, :] = 0.0
    raw_mask[:, OCEAN_W:] = 0.0
    raw_mask_t = torch.from_numpy(raw_mask).unsqueeze(0).unsqueeze(0)
    border = border_gen.generate_mask(torch.Size([1, 2, FULL_H, FULL_W])).cpu()
    return raw_mask_t * border


def run_gamma1(vel, ocean_mask_np=None):
    """Run Gamma1 eddy detection.  Returns (eddies, gamma1_field, vorticity)."""
    vel = torch.nan_to_num(vel, nan=0.0)
    om = None
    if ocean_mask_np is not None:
        om = torch.from_numpy(ocean_mask_np) if isinstance(ocean_mask_np, np.ndarray) else ocean_mask_np
    return detect_eddies_gamma(vel, ocean_mask=om, **EDDY_PARAMS)


# ──────────────────────────────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────────────────────────────

def add_land_overlay(ax, ocean_mask_np, H, W):
    """Overlay land in green."""
    land_rgba = np.zeros((H, W, 4))
    land_rgba[ocean_mask_np == 0, :3] = LAND_COLOR[:3]
    land_rgba[ocean_mask_np == 0, 3] = LAND_COLOR[3]
    ax.imshow(land_rgba, origin="upper", extent=[0, W, H, 0], zorder=1)


def add_quiver(ax, vel_np, ocean_mask_np, H, W, shared_scale=None):
    """Add velocity quiver arrows."""
    step = QUIVER_STEP
    ys = np.arange(0.5, H, step)
    xs = np.arange(0.5, W, step)
    X, Y = np.meshgrid(xs, ys)
    u = vel_np[0, ::step, ::step].copy()
    v = vel_np[1, ::step, ::step].copy()
    land_ds = ocean_mask_np[::step, ::step]
    u[land_ds == 0] = np.nan
    v[land_ds == 0] = np.nan

    scale = shared_scale if shared_scale else 1.0
    ax.quiver(X, Y, u, v, angles="xy", scale_units="xy", scale=scale,
              width=QUIVER_WIDTH, headwidth=QUIVER_HEADWIDTH,
              headlength=QUIVER_HEADLEN, color=QUIVER_COLOR,
              alpha=QUIVER_ALPHA, zorder=3)


def add_speed_bg(ax, vel_np, ocean_mask_np, H, W, vmax=None):
    """Speed magnitude heatmap background."""
    speed = np.sqrt(vel_np[0]**2 + vel_np[1]**2)
    speed_masked = np.ma.array(speed, mask=(ocean_mask_np == 0))
    if vmax is None:
        ocean_vals = speed[ocean_mask_np > 0]
        vmax = np.percentile(ocean_vals, 97) if len(ocean_vals) > 0 else 0.3
    ax.imshow(speed_masked, origin="upper", cmap=SPEED_CMAP, alpha=0.6,
              extent=[0, W, H, 0], vmin=0, vmax=vmax, zorder=0)


def add_eddy_contours(ax, eddies, color=EDDY_CONTOUR_COLOR, linestyle="-"):
    """Draw contours around detected eddies + mark centers."""
    for e in eddies:
        mask = e.mask
        if mask is not None:
            if hasattr(mask, 'numpy'):
                mask = mask.numpy()
            if mask.ndim == 2:
                ax.contour(mask.astype(float), levels=[0.5], colors=[color],
                           linewidths=1.8, linestyles=[linestyle],
                           origin="upper", extent=[0, mask.shape[1], mask.shape[0], 0],
                           zorder=5)
        ax.plot(e.center_x, e.center_y, "o", ms=8, mew=2, fillstyle="none",
                color=color, zorder=6)


def add_sensor_dots(ax, obs_mask, ocean_mask_np=None):
    """Mark observed pixels as small dots."""
    ys, xs = np.where(obs_mask > 0.5)
    ax.scatter(xs + 0.5, ys + 0.5, s=4, c="gold", edgecolors="k",
               linewidths=0.3, zorder=7, label="observations")


def format_ax(ax, H, W, title=""):
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold", pad=4)


def compute_shared_quiver_scale(gt_vel_np, ocean_mask_np):
    speed = np.sqrt(gt_vel_np[0]**2 + gt_vel_np[1]**2)
    vals = speed[ocean_mask_np > 0]
    median_mag = np.median(vals[vals > 1e-8]) if len(vals[vals > 1e-8]) > 0 else 0.1
    return median_mag / 0.35  # ARROW_GAIN


# ──────────────────────────────────────────────────────────────────────
# Main figure generators
# ──────────────────────────────────────────────────────────────────────

def make_velocity_comparison(samples_data, out_path, ocean_mask_np):
    """
    Multi-panel figure: rows = samples, cols = [Observations, GP, GP-CNN, Composite, VCNN].
    Each panel has speed background + quiver + eddy detection contours.
    """
    n_rows = len(samples_data)
    n_cols = 6  # GT, Obs, GP, GP-CNN, Composite, VCNN
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.0, n_rows * 2.4),
                             dpi=200)
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Ground Truth", "Observations", "GP", "GP-CNN",
                  "Composite\n(GP-CNN+DDPM)", "Voronoi-CNN"]

    for ri, sd in enumerate(samples_data):
        gt = sd["gt"]
        obs_mask = sd["obs_mask"]
        methods = [
            ("gt", gt, sd["gt_eddies"]),
            ("obs", gt, []),  # will overlay mask
            ("gp", sd["gp"], sd["gp_eddies"]),
            ("gpcnn", sd["gpcnn"], sd["gpcnn_eddies"]),
            ("comp", sd["comp"], sd["comp_eddies"]),
            ("vcnn", sd["vcnn"], sd["vcnn_eddies"]),
        ]

        # Shared speed scale and quiver scale from GT
        gt_speed = np.sqrt(gt[0]**2 + gt[1]**2)
        ocean_vals = gt_speed[ocean_mask_np > 0]
        vmax = np.percentile(ocean_vals, 97) if len(ocean_vals) > 0 else 0.3
        qscale = compute_shared_quiver_scale(gt, ocean_mask_np)

        for ci, (key, vel, eddies) in enumerate(methods):
            ax = axes[ri, ci]

            if key == "obs":
                # Show GT speed background dimmed, with sensor dots
                add_speed_bg(ax, vel, ocean_mask_np, OCEAN_H, OCEAN_W, vmax=vmax)
                add_land_overlay(ax, ocean_mask_np, OCEAN_H, OCEAN_W)

                # Dim overlay for missing region
                missing_rgba = np.zeros((OCEAN_H, OCEAN_W, 4))
                missing_ocean = (ocean_mask_np > 0) & (obs_mask < 0.5)
                missing_rgba[missing_ocean, :3] = 0.0
                missing_rgba[missing_ocean, 3] = 0.55
                ax.imshow(missing_rgba, origin="upper",
                          extent=[0, OCEAN_W, OCEAN_H, 0], zorder=2)

                add_sensor_dots(ax, obs_mask)
                n_obs = int(obs_mask.sum())
                pct = n_obs / int(ocean_mask_np.sum()) * 100
                format_ax(ax, OCEAN_H, OCEAN_W,
                          f"{col_titles[ci]}\n({n_obs} pts, {pct:.1f}%)" if ri == 0
                          else f"{n_obs} pts, {pct:.1f}%")
            else:
                add_speed_bg(ax, vel, ocean_mask_np, OCEAN_H, OCEAN_W, vmax=vmax)
                add_land_overlay(ax, ocean_mask_np, OCEAN_H, OCEAN_W)
                add_quiver(ax, vel, ocean_mask_np, OCEAN_H, OCEAN_W,
                           shared_scale=qscale)

                # Draw GT eddy contours as dashed reference
                if key != "gt" and sd["gt_eddies"]:
                    add_eddy_contours(ax, sd["gt_eddies"],
                                      color="cyan", linestyle="--")

                # Draw detected eddies
                if eddies:
                    add_eddy_contours(ax, eddies)

                if key == "gt":
                    n_e = len(eddies)
                    title = f"{col_titles[ci]}" if ri == 0 else ""
                    if n_e > 0:
                        title += f"\n({n_e} eddy)" if ri == 0 else f"({n_e} eddy)"
                    format_ax(ax, OCEAN_H, OCEAN_W, title)
                else:
                    # MSE badge
                    missing_ocean = (ocean_mask_np > 0) & (obs_mask < 0.5)
                    gt_t = torch.from_numpy(gt)
                    pred_t = torch.from_numpy(vel)
                    mo_t = torch.from_numpy(missing_ocean)
                    mse = (gt_t - pred_t)[:, mo_t].pow(2).mean().item()
                    n_det = len(eddies)

                    method_title = col_titles[ci] if ri == 0 else ""
                    format_ax(ax, OCEAN_H, OCEAN_W, method_title)

                    # MSE text
                    ax.text(1, OCEAN_H - 1,
                            f"MSE={mse:.2e}\nEddies={n_det}",
                            fontsize=6.5, va="bottom", ha="left",
                            color="white", fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.2",
                                      fc="black", alpha=0.6),
                            zorder=10)

            # Row label
            if ci == 0:
                tag = "EDDY" if sd.get("is_eddy") else "No eddy"
                ax.set_ylabel(f"val {sd['val_idx']}\n{tag}",
                              fontsize=8, fontweight="bold", rotation=0,
                              labelpad=40, va="center")

    fig.tight_layout(w_pad=0.3, h_pad=0.5)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def make_gamma1_comparison(samples_data, out_path, ocean_mask_np):
    """
    Gamma1 heatmap panels: rows = samples, cols = [GT, GP, GP-CNN, Composite, VCNN].
    """
    n_rows = len(samples_data)
    n_cols = 5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.8, n_rows * 2.2),
                             dpi=200)
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Ground Truth", "GP", "GP-CNN", "Composite", "Voronoi-CNN"]

    for ri, sd in enumerate(samples_data):
        gamma_fields = [
            sd["gt_gamma1"],
            sd["gp_gamma1"],
            sd["gpcnn_gamma1"],
            sd["comp_gamma1"],
            sd["vcnn_gamma1"],
        ]
        eddy_lists = [
            sd["gt_eddies"],
            sd["gp_eddies"],
            sd["gpcnn_eddies"],
            sd["comp_eddies"],
            sd["vcnn_eddies"],
        ]

        for ci, (g1, eddies) in enumerate(zip(gamma_fields, eddy_lists)):
            ax = axes[ri, ci]

            if g1 is not None:
                g1_masked = np.ma.array(g1, mask=(ocean_mask_np == 0))
                im = ax.imshow(g1_masked, origin="upper", cmap=GAMMA_CMAP,
                               vmin=-1, vmax=1,
                               extent=[0, OCEAN_W, OCEAN_H, 0], zorder=0)
            add_land_overlay(ax, ocean_mask_np, OCEAN_H, OCEAN_W)

            # Eddy contours
            if eddies:
                add_eddy_contours(ax, eddies)

            # Threshold line
            if g1 is not None:
                ax.contour(np.abs(g1), levels=[EDDY_PARAMS["gamma_threshold"]],
                           colors=["yellow"], linewidths=0.8, linestyles=["--"],
                           origin="upper",
                           extent=[0, OCEAN_W, OCEAN_H, 0], zorder=4, alpha=0.6)

            title = col_titles[ci] if ri == 0 else ""
            n_det = len(eddies)
            if n_det > 0:
                title += f"\n({n_det} detected)" if ri == 0 else f"({n_det} det.)"
            format_ax(ax, OCEAN_H, OCEAN_W, title)

            if ci == 0:
                tag = "EDDY" if sd.get("is_eddy") else "No eddy"
                ax.set_ylabel(f"val {sd['val_idx']}\n{tag}",
                              fontsize=8, fontweight="bold", rotation=0,
                              labelpad=40, va="center")

    # Colorbar
    cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    norm = Normalize(vmin=-1, vmax=1)
    sm = cm.ScalarMappable(cmap=GAMMA_CMAP, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cax, label=r"$\Gamma_1$")

    fig.tight_layout(rect=[0, 0, 0.91, 1], w_pad=0.3, h_pad=0.5)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def make_gp_variance_figure(sd, out_path, ocean_mask_np):
    """Show GP posterior std and the resulting blending weight."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), dpi=200)

    # Panel 1: GP posterior std (u-component)
    gp_std = np.sqrt(np.clip(sd["gp_var"][0], 0, None))
    gp_std_masked = np.ma.array(gp_std, mask=(ocean_mask_np == 0))
    im0 = axes[0].imshow(gp_std_masked, origin="upper", cmap="hot",
                         extent=[0, OCEAN_W, OCEAN_H, 0])
    add_land_overlay(axes[0], ocean_mask_np, OCEAN_H, OCEAN_W)
    add_sensor_dots(axes[0], sd["obs_mask"])
    format_ax(axes[0], OCEAN_H, OCEAN_W, r"GP posterior $\sigma_u$")
    fig.colorbar(im0, ax=axes[0], shrink=0.8, label="std (m/s)")

    # Panel 2: Blending weight w ∈ [0,1]
    w = sd["weight"][0, 0, :OCEAN_H, :OCEAN_W]
    if hasattr(w, 'numpy'):
        w = w.numpy()
    w_masked = np.ma.array(w, mask=(ocean_mask_np == 0))
    im1 = axes[1].imshow(w_masked, origin="upper", cmap="plasma",
                         vmin=0, vmax=1,
                         extent=[0, OCEAN_W, OCEAN_H, 0])
    add_land_overlay(axes[1], ocean_mask_np, OCEAN_H, OCEAN_W)
    add_sensor_dots(axes[1], sd["obs_mask"])
    format_ax(axes[1], OCEAN_H, OCEAN_W, "Blending weight $w$\n(0=CNN, 1=DDPM)")
    fig.colorbar(im1, ax=axes[1], shrink=0.8)

    # Panel 3: Composite vs CNN difference
    diff = np.sqrt((sd["comp"][0] - sd["gpcnn"][0])**2 + (sd["comp"][1] - sd["gpcnn"][1])**2)
    diff_masked = np.ma.array(diff, mask=(ocean_mask_np == 0))
    im2 = axes[2].imshow(diff_masked, origin="upper", cmap="inferno",
                         extent=[0, OCEAN_W, OCEAN_H, 0])
    add_land_overlay(axes[2], ocean_mask_np, OCEAN_H, OCEAN_W)
    format_ax(axes[2], OCEAN_H, OCEAN_W, "|Composite − GP-CNN|")
    fig.colorbar(im2, ax=axes[2], shrink=0.8, label="speed diff (m/s)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reveal-pct", type=float, default=5.0)
    parser.add_argument("--n-ensemble", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--timestep", type=int, default=None)
    args = parser.parse_args()

    if args.timestep is not None:
        t_val = args.timestep
    elif args.reveal_pct <= 1.5:
        t_val = 200
    else:
        t_val = 75

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
    print("Loading models...")
    ddpm = load_ddpm(device)
    gpcnn, gpcnn_ckpt = load_gp_cnn(device)
    gpcnn_norm_mean = gpcnn_ckpt["norm_mean"].numpy()
    gpcnn_norm_std = gpcnn_ckpt["norm_std"].numpy()
    ocean_mask = gpcnn_ckpt["ocean_mask"]

    vcnn, vcnn_ckpt = load_vcnn(device)
    vcnn_norm_mean = vcnn_ckpt["norm_mean"].numpy()
    vcnn_norm_std = vcnn_ckpt["norm_std"].numpy()

    _, val_vel = load_pickle_data()

    # Load eddy-balanced indices
    ddpm_data = torch.load(DDPM_EVAL_PT, map_location="cpu", weights_only=False)
    ddpm_samples = ddpm_data["samples"]
    eddy_set = set(ddpm_data.get("eddy_indices", []))

    # ── Pick representative samples ──
    # From the 5% results, good picks:
    #   vi=566:  GT=2,  VCNN=2, rest=0  (VCNN eddy advantage)
    #   vi=460:  GT=1,  all=1            (all methods agree)
    #   vi=709:  GT=1,  Comp=1, VCNN=1, GP/CNN=0 (gradient of detection)
    #   vi=1032: GT=1,  CNN/Comp/VCNN=1, GP=0 (learning helps)
    # + 1 non-eddy sample (take any clean one)

    # Find the val_index → sample_index mapping
    vi_to_si = {s["val_idx"]: i for i, s in enumerate(ddpm_samples)}

    PICKS = [
        ("eddy_all_detect", 460),   # all methods detect
        ("eddy_vcnn_only", 566),    # only VCNN detects both
        ("eddy_comp_vcnn", 709),    # composite+VCNN detect, GP misses
        ("eddy_learning", 1032),    # learning methods detect, GP fails
    ]

    # Add a non-eddy sample
    for s in ddpm_samples:
        if s["val_idx"] not in eddy_set and s["val_idx"] > 100:
            PICKS.append(("non_eddy", s["val_idx"]))
            break

    rng = np.random.default_rng(args.seed)

    # Process each sample — we need to reproduce the exact same mask
    # by replaying the RNG in order (same as eddy_compare)
    all_val_indices = [s["val_idx"] for s in ddpm_samples]

    # Pre-generate all masks to keep RNG in sync
    all_masks = {}
    rng_masks = np.random.default_rng(args.seed)
    for vi in all_val_indices:
        all_masks[vi] = generate_random_mask(ocean_mask, args.reveal_pct, rng_masks)

    print(f"\nGenerating figures for {len(PICKS)} samples at {args.reveal_pct}% coverage, t={t_val}")
    print(f"{'=' * 70}")

    samples_data = []
    for tag, vi in PICKS:
        print(f"  Processing val_idx={vi} ({tag})...")

        gt_vel = val_vel[vi].numpy()
        obs_mask = all_masks[vi]
        is_eddy = vi in eddy_set

        # GP
        gp_mean, gp_var = compute_gp(gt_vel, obs_mask, ocean_mask)

        # GP-CNN
        gpcnn_phys, gpcnn_ddpm_std = gp_cnn_predict(
            gpcnn, gp_mean, gp_var, obs_mask, ocean_mask,
            gpcnn_norm_mean, gpcnn_norm_std, device
        )
        gpcnn_cpu = gpcnn_phys.cpu()
        gpcnn_small = gpcnn_cpu[0, :, :OCEAN_H, :OCEAN_W].numpy()

        # DDPM composite
        missing_mask_1ch = build_ddpm_missing_mask(obs_mask, ocean_mask)
        ens_mean_std, _ = ddpm_ensemble(
            ddpm, gpcnn_ddpm_std.to(device), missing_mask_1ch.to(device),
            t_val, args.n_ensemble, device
        )
        ens_mean_phys = ddpm_standardizer.unstandardize(
            ens_mean_std.cpu().squeeze(0)
        ).unsqueeze(0)
        w = compute_gp_var_weight(gp_var, ocean_mask)
        comp_phys = (1.0 - w) * gpcnn_cpu + w * ens_mean_phys
        comp_small = comp_phys[0, :, :OCEAN_H, :OCEAN_W].numpy()

        # VCNN
        vcnn_pred = vcnn_predict(
            vcnn, gt_vel, obs_mask, ocean_mask,
            vcnn_norm_mean, vcnn_norm_std, device
        )
        vcnn_small = vcnn_pred.numpy()

        # Convert ocean_mask to the right type for eddy detection
        ocean_bool = torch.from_numpy(ocean_mask).bool()

        # Eddy detection on each
        gt_eddies, gt_g1, gt_vort = run_gamma1(torch.from_numpy(gt_vel), ocean_mask)
        gp_eddies, gp_g1, _ = run_gamma1(torch.from_numpy(gp_mean), ocean_mask)
        cnn_eddies, cnn_g1, _ = run_gamma1(torch.from_numpy(gpcnn_small), ocean_mask)
        comp_eddies, comp_g1, _ = run_gamma1(torch.from_numpy(comp_small), ocean_mask)
        vcnn_eddies, vcnn_g1, _ = run_gamma1(torch.from_numpy(vcnn_small), ocean_mask)

        # Ensure gamma1 fields are numpy
        def _to_np(x):
            if x is None: return None
            return x.numpy() if hasattr(x, 'numpy') else np.asarray(x)

        gt_g1, gp_g1, cnn_g1, comp_g1, vcnn_g1 = [
            _to_np(g) for g in [gt_g1, gp_g1, cnn_g1, comp_g1, vcnn_g1]
        ]

        sd = {
            "val_idx": vi, "tag": tag, "is_eddy": is_eddy,
            "obs_mask": obs_mask,
            "gt": gt_vel, "gp": gp_mean, "gpcnn": gpcnn_small,
            "comp": comp_small, "vcnn": vcnn_small,
            "gp_var": gp_var, "weight": w.numpy(),
            "gt_eddies": gt_eddies, "gp_eddies": gp_eddies,
            "gpcnn_eddies": cnn_eddies, "comp_eddies": comp_eddies,
            "vcnn_eddies": vcnn_eddies,
            "gt_gamma1": gt_g1, "gp_gamma1": gp_g1, "gpcnn_gamma1": cnn_g1,
            "comp_gamma1": comp_g1, "vcnn_gamma1": vcnn_g1,
        }
        samples_data.append(sd)
        print(f"    GT={len(gt_eddies)} GP={len(gp_eddies)} CNN={len(cnn_eddies)} "
              f"Comp={len(comp_eddies)} VCNN={len(vcnn_eddies)}")

    # ── Generate figures ──
    print(f"\n{'=' * 70}")
    print("Generating figures...")

    # Fig 1: Velocity field comparison (eddy samples only)
    eddy_samples = [sd for sd in samples_data if sd["is_eddy"]]
    make_velocity_comparison(
        eddy_samples,
        OUT_DIR / "fig_velocity_comparison_eddy.png",
        ocean_mask,
    )

    # Fig 2: Velocity comparison (all samples)
    make_velocity_comparison(
        samples_data,
        OUT_DIR / "fig_velocity_comparison_all.png",
        ocean_mask,
    )

    # Fig 3: Gamma1 heatmap comparison (eddy samples)
    make_gamma1_comparison(
        eddy_samples,
        OUT_DIR / "fig_gamma1_comparison.png",
        ocean_mask,
    )

    # Fig 4: GP variance / blending weight (pick first eddy sample)
    make_gp_variance_figure(
        samples_data[0],
        OUT_DIR / "fig_gp_variance_weight.png",
        ocean_mask,
    )

    # Fig 5: Individual best-case panels (where all methods detect)
    for sd in samples_data:
        if sd["tag"] == "eddy_all_detect":
            make_velocity_comparison(
                [sd],
                OUT_DIR / f"fig_single_eddy_vi{sd['val_idx']}.png",
                ocean_mask,
            )
            make_gamma1_comparison(
                [sd],
                OUT_DIR / f"fig_single_gamma1_vi{sd['val_idx']}.png",
                ocean_mask,
            )

    # Fig 6: VCNN-only detection case
    for sd in samples_data:
        if sd["tag"] == "eddy_vcnn_only":
            make_velocity_comparison(
                [sd],
                OUT_DIR / f"fig_vcnn_advantage_vi{sd['val_idx']}.png",
                ocean_mask,
            )

    print(f"\nAll figures saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
