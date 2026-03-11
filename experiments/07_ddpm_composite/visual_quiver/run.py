#!/usr/bin/env python3
"""Visual quiver-plot comparison for the DDPM composite pipeline.

Generates one quiver-plot PNG per method (GT, GP, VCNN, Composite) using
the project's standard plot_vector_field style. All outputs go to the
results/ subfolder within this experiment directory.

Usage:
    PYTHONPATH=. python experiments/07_ddpm_composite/visual_quiver/run.py
    PYTHONPATH=. python experiments/07_ddpm_composite/visual_quiver/run.py --reveal-pct 5.0 --t-val 75
    PYTHONPATH=. python experiments/07_ddpm_composite/visual_quiver/run.py --reveal-pct 1.0 --t-val 200 --val-idx 460
"""

import argparse, os, pickle, sys
from pathlib import Path
import numpy as np
import torch

# Ensure project root is on sys.path
BASE_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE_DIR))

from plots.visualization_tools.plot_vector_field_tool import plot_vector_field
from scripts.voronoi_cnn_model import VoronoiCNN, build_voronoi_input
from ddpm.neural_networks.ddpm import GaussianDDPM
from ddpm.neural_networks.unets.unet_film_attn import MyUNet_FiLM_Attn
from ddpm.helper_functions.standardize_data import ZScoreStandardizer
from ddpm.helper_functions.interpolation_tool import gp_fill

# ── constants ────────────────────────────────────────────────────────
OCEAN_H, OCEAN_W = 44, 94
FULL_H, FULL_W   = 64, 128
N_STEPS = 250

U_MEAN, U_STD = -0.06929559429949586, 0.1358005549716049
V_MEAN, V_STD = -0.0323937796117541, 0.08899177232117582
ddpm_standardizer = ZScoreStandardizer(U_MEAN, U_STD, V_MEAN, V_STD)

GP_PARAMS = dict(lengthscale=14.1, variance=0.0103420345, noise=1e-8,
                 kernel_type="rbf_legacy", coord_system="pixels")

# Checkpoints (paths relative to project root)
DDPM_WEIGHT = "experiments/06_gp_forward/gp_conditioned/results/inpaint_gaussian_t250_best_ema_weights.pt"
GP_CNN_CKPT = "results/gp_cnn_diverse/gp_cnn_diverse_best.pt"
VCNN_CKPT   = "results/voronoi_cnn/voronoi_cnn_best.pt"

# Output: results/ directory next to this script
EXPERIMENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXPERIMENT_DIR / "results"

# ── helpers ──────────────────────────────────────────────────────────

def load_data():
    data_path = BASE_DIR / "data.pickle"
    with open(data_path, "rb") as f:
        _train, val, _test = pickle.load(f)
    def to_t(arr):
        return torch.nan_to_num(
            torch.from_numpy(np.ascontiguousarray(arr)).float().permute(3,2,1,0), nan=0.0)
    return to_t(val)


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
    gm[OCEAN_H:, :] = 0.0; gm[:, OCEAN_W:] = 0.0
    gm_t = torch.from_numpy(gm).unsqueeze(0).unsqueeze(0).expand(1,2,-1,-1).clone()
    mean, var = gp_fill(torch.from_numpy(vf), gm_t, return_variance=True, use_double=True,
                        **GP_PARAMS)
    return (mean[0, :, :OCEAN_H, :OCEAN_W].numpy(),
            var[0, :, :OCEAN_H, :OCEAN_W].numpy())


def load_ddpm(dev):
    net = MyUNet_FiLM_Attn(n_steps=N_STEPS, time_emb_dim=256, in_channels=5)
    ddpm = GaussianDDPM(net, n_steps=N_STEPS, min_beta=1e-4, max_beta=0.02, device=dev)
    ddpm.load_state_dict(torch.load(str(BASE_DIR / DDPM_WEIGHT),
                                     map_location="cpu", weights_only=False))
    ddpm.to(dev); ddpm.eval()
    return ddpm


def load_gp_cnn(dev):
    ck = torch.load(str(BASE_DIR / GP_CNN_CKPT), map_location="cpu", weights_only=False)
    cfg = ck["model_config"]
    m = VoronoiCNN(in_channels=cfg["in_channels"], out_channels=cfg["out_channels"],
                   base_ch=cfg.get("base_ch",32), depth=cfg.get("depth",3)).to(dev)
    m.load_state_dict(ck["model_state"]); m.eval()
    return m, ck


def load_vcnn(dev):
    ck = torch.load(str(BASE_DIR / VCNN_CKPT), map_location="cpu", weights_only=False)
    m = VoronoiCNN(**ck["model_config"]).to(dev)
    m.load_state_dict(ck["model_state"]); m.eval()
    return m, ck


def predict_gp_cnn(model, gp_mean, gp_var, obs_mask, ocean_mask, dev):
    nm = np.array([U_MEAN, V_MEAN], dtype=np.float32)
    ns = np.array([U_STD, V_STD], dtype=np.float32)
    gp_n = ((gp_mean - nm[:,None,None]) / ns[:,None,None]) * ocean_mask[None]
    std_n = (np.sqrt(np.clip(gp_var,0,None)) / ns[:,None,None]) * ocean_mask[None]
    inp = torch.cat([torch.from_numpy(gp_n.astype(np.float32)),
                     torch.from_numpy(std_n.astype(np.float32)),
                     torch.from_numpy(obs_mask).unsqueeze(0),
                     torch.from_numpy(ocean_mask).unsqueeze(0)], dim=0).unsqueeze(0).to(dev)
    with torch.no_grad():
        p = model(inp)
    mt = torch.tensor(nm).view(1,2,1,1).to(dev)
    st = torch.tensor(ns).view(1,2,1,1).to(dev)
    phys = (p * st + mt) * torch.from_numpy(ocean_mask).float().to(dev).view(1,1,OCEAN_H,OCEAN_W)
    full = torch.zeros(1,2,FULL_H,FULL_W, device=dev)
    full[:,:,:OCEAN_H,:OCEAN_W] = phys
    return full, ddpm_standardizer(full.squeeze(0)).unsqueeze(0)


def predict_vcnn(model, vel, obs_mask, ocean_mask, dev):
    nm = np.array([U_MEAN, V_MEAN], dtype=np.float32)
    ns = np.array([U_STD, V_STD], dtype=np.float32)
    vel_n = ((vel - nm[:,None,None]) / ns[:,None,None]) * ocean_mask[None]
    vi = build_voronoi_input(vel_n, obs_mask, ocean_mask)
    with torch.no_grad():
        p = model(torch.from_numpy(vi).unsqueeze(0).to(dev))
    mt = torch.tensor(nm).view(1,2,1,1).to(dev)
    st = torch.tensor(ns).view(1,2,1,1).to(dev)
    phys = (p * st + mt) * torch.from_numpy(ocean_mask).float().to(dev).view(1,1,OCEAN_H,OCEAN_W)
    return phys.squeeze(0).cpu().numpy()


def ddpm_ensemble(ddpm, cond_std, miss_mask, t_val, n_ens, dev):
    preds = []
    for k in range(n_ens):
        torch.manual_seed(42 + k*1000)
        ab = ddpm.alpha_bars[t_val].to(dev)
        noise = torch.randn_like(cond_std)
        noisy = ab.sqrt() * cond_std + (1-ab).sqrt() * noise
        xc = torch.cat([noisy, miss_mask, cond_std], dim=1)
        tt = torch.full((1,1), t_val, device=dev, dtype=torch.long)
        with torch.no_grad():
            preds.append(ddpm.network(xc, tt))
    s = torch.stack(preds)
    return s.mean(0), s.std(0)


def composite(gpcnn_full, ens_mean, gp_var, ocean_mask, dev):
    ocean_b = ocean_mask.astype(bool)
    gp_std = np.sqrt(np.clip(gp_var, 0, None))
    w = np.zeros_like(gp_std)
    for c in range(2):
        v = gp_std[c][ocean_b]
        lo, hi = v.min(), v.max()
        if hi > lo:
            w[c] = (gp_std[c] - lo) / (hi - lo)
        w[c] *= ocean_mask
    wf = torch.zeros(1,2,FULL_H,FULL_W, device=dev)
    wf[0,:,:OCEAN_H,:OCEAN_W] = torch.from_numpy(w.astype(np.float32)).to(dev)
    ens_phys = ddpm_standardizer.unstandardize(ens_mean.squeeze(0)).unsqueeze(0)
    comp = (1 - wf) * gpcnn_full + wf * ens_phys
    return comp


def build_miss_mask(obs_mask, ocean_mask):
    from ddpm.helper_functions.masks.border_mask import BorderMaskGenerator
    bm = BorderMaskGenerator()
    raw = np.ones((FULL_H,FULL_W), dtype=np.float32)
    raw[:OCEAN_H,:OCEAN_W] -= obs_mask
    raw[:OCEAN_H,:OCEAN_W] *= ocean_mask
    raw[OCEAN_H:,:] = 0.0; raw[:,OCEAN_W:] = 0.0
    rt = torch.from_numpy(raw).unsqueeze(0).unsqueeze(0)
    border = bm.generate_mask(torch.Size([1,2,FULL_H,FULL_W])).cpu()
    return rt * border


# ── plotting ─────────────────────────────────────────────────────────

def plot_single(vel_np, ocean_mask_np, title, out_path,
                obs_mask=None, arrow_len=0.9, step=2):
    """One quiver image using the project's standard plot_vector_field style."""
    vx = torch.from_numpy(vel_np[0].astype(np.float32))
    vy = torch.from_numpy(vel_np[1].astype(np.float32))
    land = torch.from_numpy((ocean_mask_np < 0.5).astype(np.float32))

    miss = None
    if obs_mask is not None:
        miss_np = ((ocean_mask_np > 0.5) & (obs_mask < 0.5)).astype(np.float32)
        miss = torch.from_numpy(miss_np)

    plot_vector_field(
        vx, vy,
        step=step,
        scale=1.0,
        title=title,
        file=str(out_path),
        land_mask=land,
        land_color="forestgreen",
        crop_top_right_zero_pad=True,
        auto_rescale_for_display=True,
        target_median_arrow_len=arrow_len,
        missing_mask=miss,
        missing_color="red",
        missing_alpha=0.25,
    )
    print(f"  saved {out_path}")


# ── main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate quiver-plot comparison of inpainting methods")
    parser.add_argument("--reveal-pct", type=float, default=1.0,
                        help="Observation coverage percentage (default: 1.0)")
    parser.add_argument("--val-idx", type=int, default=460,
                        help="Validation sample index")
    parser.add_argument("--t-val", type=int, default=200,
                        help="DDPM timestep for single-step inference")
    parser.add_argument("--n-ensemble", type=int, default=5,
                        help="DDPM ensemble size")
    parser.add_argument("--arrow-len", type=float, default=0.9,
                        help="Target median arrow length for rescaling")
    parser.add_argument("--step", type=int, default=2,
                        help="Quiver subsampling step")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for mask generation")
    args = parser.parse_args()

    # Output goes into results/ with a coverage-tagged subdirectory
    out_dir = RESULTS_DIR / f"{args.reveal_pct:.1f}pct_vi{args.val_idx}_t{args.t_val}"
    out_dir.mkdir(parents=True, exist_ok=True)

    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {dev}")
    print(f"Output: {out_dir}")

    # ── load data ──────────────────────────────────────────────────
    val = load_data()
    gt_full = val[args.val_idx]                       # (2, 44, 94)
    gt_crop = gt_full.numpy()

    ocean_mask = (torch.abs(gt_full[0]) +
                  torch.abs(gt_full[1]) > 1e-8).float().numpy()

    rng = np.random.default_rng(args.seed)
    obs_mask = random_mask(ocean_mask, args.reveal_pct, rng)
    vel_phys = gt_crop.copy()

    print(f"Sample val[{args.val_idx}], {args.reveal_pct}% coverage "
          f"({int(obs_mask.sum())} pts), t={args.t_val}")

    # ── GT ─────────────────────────────────────────────────────────
    print("Plotting GT …")
    plot_single(gt_crop, ocean_mask, "Ground Truth", out_dir/"gt.png",
                arrow_len=args.arrow_len, step=args.step)

    # ── GP ─────────────────────────────────────────────────────────
    print("Computing GP …")
    gp_mean, gp_var = compute_gp(vel_phys, obs_mask, ocean_mask)
    plot_single(gp_mean, ocean_mask, "Gaussian Process", out_dir/"gp.png",
                obs_mask=obs_mask, arrow_len=args.arrow_len, step=args.step)

    # ── VCNN ───────────────────────────────────────────────────────
    print("Computing VCNN …")
    vcnn_model, _ = load_vcnn(dev)
    vcnn_pred = predict_vcnn(vcnn_model, vel_phys, obs_mask, ocean_mask, dev)
    plot_single(vcnn_pred, ocean_mask, "Voronoi-CNN", out_dir/"vcnn.png",
                arrow_len=args.arrow_len, step=args.step)

    # ── Composite ──────────────────────────────────────────────────
    print("Computing GP-CNN + DDPM ensemble → Composite …")
    gpcnn_model, _ = load_gp_cnn(dev)
    gpcnn_full, gpcnn_std = predict_gp_cnn(gpcnn_model, gp_mean, gp_var,
                                           obs_mask, ocean_mask, dev)

    ddpm = load_ddpm(dev)
    miss_mask = build_miss_mask(obs_mask, ocean_mask).to(dev)
    ens_mean, _ = ddpm_ensemble(ddpm, gpcnn_std.to(dev), miss_mask.to(dev),
                                args.t_val, args.n_ensemble, dev)
    comp = composite(gpcnn_full, ens_mean, gp_var, ocean_mask, dev)
    comp_np = comp[0, :, :OCEAN_H, :OCEAN_W].cpu().numpy()
    plot_single(comp_np, ocean_mask, "GP-CNN / DDPM Composite",
                out_dir/"composite.png",
                arrow_len=args.arrow_len, step=args.step)

    # ── Save tensors for downstream analysis ───────────────────────
    torch.save({
        "gt": gt_crop,
        "gp_mean": gp_mean,
        "gp_var": gp_var,
        "vcnn": vcnn_pred,
        "composite": comp_np,
        "obs_mask": obs_mask,
        "ocean_mask": ocean_mask,
        "args": vars(args),
    }, out_dir / "tensors.pt")
    print(f"  saved {out_dir / 'tensors.pt'}")

    print(f"\nDone — 4 figures + tensors in {out_dir}")


if __name__ == "__main__":
    main()
