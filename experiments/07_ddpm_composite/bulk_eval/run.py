#!/usr/bin/env python3
"""100-sample bulk evaluation: GP-Diff (S6 RePaint), DDPM Composite, GP, V-CNN.

Runs all 4 methods on N validation samples with random Gaussian masks at
a given coverage percentage. For each sample:
  - Saves a .pt file with all predictions and GT
  - Generates a quiver plot for each method

After all samples: prints avg and median MSE per method and saves a
summary .pt file.

GP-Diff = S6 multi-stage adaptive GP-Refined RePaint using the
unconditional eps-prediction model (MyUNet_Attn). 6 cascaded stages:
  Stage 1: GP → noise to t=75 → adaptive RePaint denoise
  Stages 2-6: prev output → noise to t=50 → adaptive RePaint (var decayed)

Usage:
    PYTHONPATH=. python experiments/07_ddpm_composite/bulk_eval/run.py
    PYTHONPATH=. python experiments/07_ddpm_composite/bulk_eval/run.py --reveal-pct 0.1 --n-samples 100
    PYTHONPATH=. python experiments/07_ddpm_composite/bulk_eval/run.py --reveal-pct 1.0 --n-samples 10 --skip-plots
"""

import argparse, os, pickle, sys, time
from pathlib import Path
import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE_DIR))

from plots.visualization_tools.plot_vector_field_tool import plot_vector_field
from scripts.voronoi_cnn_model import VoronoiCNN, build_voronoi_input
from ddpm.neural_networks.ddpm import GaussianDDPM
from ddpm.neural_networks.unets.unet_film_attn import MyUNet_FiLM_Attn
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
ddpm_standardizer = ZScoreStandardizer(U_MEAN, U_STD, V_MEAN, V_STD)

GP_PARAMS = dict(lengthscale=14.1, variance=0.0103420345, noise=1e-8,
                 kernel_type="rbf_legacy", coord_system="pixels")

DDPM_WEIGHT = "experiments/06_gp_forward/gp_conditioned/results/inpaint_gaussian_t250_best_ema_weights.pt"
UNCOND_CKPT = "experiments/02_inpaint_algorithm/repaint_gaussian_attn/results/inpaint_gaussian_t250_best_checkpoint.pt"
GP_CNN_CKPT = "results/gp_cnn_diverse/gp_cnn_diverse_best.pt"
VCNN_CKPT   = "results/voronoi_cnn/voronoi_cnn_best.pt"

EXPERIMENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXPERIMENT_DIR / "results"

NM = np.array([U_MEAN, V_MEAN], dtype=np.float32)
NS = np.array([U_STD, V_STD], dtype=np.float32)

# ── data ─────────────────────────────────────────────────────────────

def load_data():
    with open(str(BASE_DIR / "data.pickle"), "rb") as f:
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


# ── GP ───────────────────────────────────────────────────────────────

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


# ── model loading ────────────────────────────────────────────────────

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
    return m


def load_vcnn(dev):
    ck = torch.load(str(BASE_DIR / VCNN_CKPT), map_location="cpu", weights_only=False)
    m = VoronoiCNN(**ck["model_config"]).to(dev)
    m.load_state_dict(ck["model_state"]); m.eval()
    return m


def load_uncond_ddpm(dev):
    """Load the unconditional eps-prediction model (MyUNet_Attn) for S6 RePaint."""
    ckpt = torch.load(str(BASE_DIR / UNCOND_CKPT), map_location="cpu", weights_only=False)
    n_steps  = ckpt.get("n_steps", 250)
    min_beta = ckpt.get("min_beta", 0.0001)
    max_beta = ckpt.get("max_beta", 0.02)
    net = MyUNet_Attn(n_steps=n_steps, time_emb_dim=256)
    ddpm_unc = GaussianDDPM(net, n_steps=n_steps,
                             min_beta=min_beta, max_beta=max_beta, device=dev)
    ddpm_unc.load_state_dict(ckpt["model_state_dict"])
    ddpm_unc.to(dev); ddpm_unc.eval()
    return ddpm_unc


# ── inference helpers ────────────────────────────────────────────────

def predict_gp_cnn(model, gp_mean, gp_var, obs_mask, ocean_mask, dev):
    gp_n = ((gp_mean - NM[:,None,None]) / NS[:,None,None]) * ocean_mask[None]
    std_n = (np.sqrt(np.clip(gp_var,0,None)) / NS[:,None,None]) * ocean_mask[None]
    inp = torch.cat([torch.from_numpy(gp_n.astype(np.float32)),
                     torch.from_numpy(std_n.astype(np.float32)),
                     torch.from_numpy(obs_mask).unsqueeze(0),
                     torch.from_numpy(ocean_mask).unsqueeze(0)], dim=0).unsqueeze(0).to(dev)
    with torch.no_grad():
        p = model(inp)
    mt = torch.tensor(NM).view(1,2,1,1).to(dev)
    st = torch.tensor(NS).view(1,2,1,1).to(dev)
    phys = (p * st + mt) * torch.from_numpy(ocean_mask).float().to(dev).view(1,1,OCEAN_H,OCEAN_W)
    full = torch.zeros(1,2,FULL_H,FULL_W, device=dev)
    full[:,:,:OCEAN_H,:OCEAN_W] = phys
    return full, ddpm_standardizer(full.squeeze(0)).unsqueeze(0)


def predict_vcnn(model, vel, obs_mask, ocean_mask, dev):
    vel_n = ((vel - NM[:,None,None]) / NS[:,None,None]) * ocean_mask[None]
    vi = build_voronoi_input(vel_n, obs_mask, ocean_mask)
    with torch.no_grad():
        p = model(torch.from_numpy(vi).unsqueeze(0).to(dev))
    mt = torch.tensor(NM).view(1,2,1,1).to(dev)
    st = torch.tensor(NS).view(1,2,1,1).to(dev)
    phys = (p * st + mt) * torch.from_numpy(ocean_mask).float().to(dev).view(1,1,OCEAN_H,OCEAN_W)
    return phys.squeeze(0).cpu().numpy()


def build_miss_mask(obs_mask, ocean_mask, border_gen):
    raw = np.ones((FULL_H,FULL_W), dtype=np.float32)
    raw[:OCEAN_H,:OCEAN_W] -= obs_mask
    raw[:OCEAN_H,:OCEAN_W] *= ocean_mask
    raw[OCEAN_H:,:] = 0.0; raw[:,OCEAN_W:] = 0.0
    rt = torch.from_numpy(raw).unsqueeze(0).unsqueeze(0)
    border = border_gen.generate_mask(torch.Size([1,2,FULL_H,FULL_W])).cpu()
    return rt * border


def ddpm_single_step(ddpm, cond_std, miss_mask, t_val, seed, dev):
    """Single-step x₀ prediction."""
    torch.manual_seed(seed)
    ab = ddpm.alpha_bars[t_val].to(dev)
    noise = torch.randn_like(cond_std)
    noisy = ab.sqrt() * cond_std + (1-ab).sqrt() * noise
    xc = torch.cat([noisy, miss_mask, cond_std], dim=1)
    tt = torch.full((1,1), t_val, device=dev, dtype=torch.long)
    with torch.no_grad():
        return ddpm.network(xc, tt)


def s6_repaint(uncond_ddpm, gt_np, obs_mask, ocean_mask, gp_mean, gp_var,
               border_gen, dev, max_stages=6, t_start=75, t_refine=50,
               noise_floor=0.2, noise_floor_refine=0.3,
               var_decay=0.1, gamma=3.0, resample_steps=5, seed=42):
    """S6 multi-stage adaptive GP-Refined RePaint (GP-Diff).

    Uses the unconditional eps-prediction model (MyUNet_Attn).
    6 cascaded stages: GP → noise → adaptive RePaint denoise → repeat.
    """
    noise_strategy = get_noise_strategy("gaussian")

    # Build full-size standardized GT
    gt_full = np.zeros((1, 2, FULL_H, FULL_W), dtype=np.float32)
    gt_full[0, :, :OCEAN_H, :OCEAN_W] = gt_np * ocean_mask[None]
    input_image = ddpm_standardizer(
        torch.from_numpy(gt_full).squeeze(0)).unsqueeze(0).to(dev)

    # Build full-size standardized GP
    gp_full = np.zeros((1, 2, FULL_H, FULL_W), dtype=np.float32)
    gp_full[0, :, :OCEAN_H, :OCEAN_W] = gp_mean * ocean_mask[None]
    gp_std = ddpm_standardizer(
        torch.from_numpy(gp_full).squeeze(0)).unsqueeze(0).to(dev)

    # Build missing mask (1=missing) — match the S6 script convention
    land_mask = (torch.from_numpy(gt_full).abs() > 1e-5).float().to(dev)
    raw_miss = np.ones((FULL_H, FULL_W), dtype=np.float32)
    raw_miss[:OCEAN_H, :OCEAN_W] -= obs_mask
    raw_miss[:OCEAN_H, :OCEAN_W] *= ocean_mask
    raw_miss[OCEAN_H:, :] = 0.0
    raw_miss[:, OCEAN_W:] = 0.0
    raw_miss_t = torch.from_numpy(raw_miss).unsqueeze(0).unsqueeze(0).to(dev)
    border = border_gen.generate_mask(torch.Size([1, 2, FULL_H, FULL_W])).to(dev)
    missing_mask = raw_miss_t * border * land_mask

    # GP variance in physical space (1, 2, 64, 128)
    gp_var_full = np.zeros((1, 2, FULL_H, FULL_W), dtype=np.float32)
    gp_var_full[0, :, :OCEAN_H, :OCEAN_W] = gp_var * ocean_mask[None]
    gp_var_t = torch.from_numpy(gp_var_full).to(dev)

    # Multi-stage refinement
    current_prior = gp_std.clone()
    current_var = gp_var_t.clone()

    for stage in range(1, max_stages + 1):
        if stage == 1:
            t_s, nf = t_start, noise_floor
            seed_s = seed
        else:
            t_s, nf = t_refine, noise_floor_refine
            seed_s = seed + stage * 10000
            current_var = current_var * var_decay

        torch.manual_seed(seed_s)
        with torch.no_grad():
            stage_out = repaint_gp_init_adaptive(
                uncond_ddpm, input_image, missing_mask,
                gp_image=current_prior,
                gp_variance_map=current_var,
                t_start=t_s,
                noise_floor=nf,
                n_samples=1, device=dev,
                noise_strategy=noise_strategy,
                prediction_target="eps",
                resample_steps=resample_steps,
                project_div_free=False,
                anneal_floor=False,
                gamma=gamma,
            )
        current_prior = stage_out.clone()

    # Convert back to physical space
    result_phys = ddpm_standardizer.unstandardize(stage_out.squeeze(0).cpu())
    return result_phys[:, :OCEAN_H, :OCEAN_W].numpy()


def ensemble_single_step(ddpm, cond_std, miss_mask, t_val, n_ens, dev):
    preds = []
    for k in range(n_ens):
        preds.append(ddpm_single_step(ddpm, cond_std, miss_mask, t_val,
                                       seed=42+k*1000, dev=dev))
    s = torch.stack(preds)
    return s.mean(0), s.std(0)


def variance_composite(gpcnn_full, ens_mean_std, gp_var, ocean_mask):
    """Variance-weighted blend: CNN near observations, DDPM far away. All CPU."""
    ocean_b = ocean_mask.astype(bool)
    gp_std = np.sqrt(np.clip(gp_var, 0, None))
    w = np.zeros_like(gp_std)
    for c in range(2):
        v = gp_std[c][ocean_b]
        lo, hi = v.min(), v.max()
        if hi > lo:
            w[c] = (gp_std[c] - lo) / (hi - lo)
        w[c] *= ocean_mask
    wf = torch.zeros(1,2,FULL_H,FULL_W)
    wf[0,:,:OCEAN_H,:OCEAN_W] = torch.from_numpy(w.astype(np.float32))
    ens_phys = ddpm_standardizer.unstandardize(ens_mean_std.squeeze(0)).unsqueeze(0)
    return (1 - wf) * gpcnn_full + wf * ens_phys


# ── plotting ─────────────────────────────────────────────────────────

def plot_quiver(vel_np, ocean_mask_np, title, out_path,
                obs_mask=None, arrow_len=0.9, step=2):
    vx = torch.from_numpy(vel_np[0].astype(np.float32))
    vy = torch.from_numpy(vel_np[1].astype(np.float32))
    land = torch.from_numpy((ocean_mask_np < 0.5).astype(np.float32))
    miss = None
    if obs_mask is not None:
        miss_np = ((ocean_mask_np > 0.5) & (obs_mask < 0.5)).astype(np.float32)
        miss = torch.from_numpy(miss_np)
    plot_vector_field(
        vx, vy, step=step, scale=1.0, title=title, file=str(out_path),
        land_mask=land, land_color="forestgreen",
        crop_top_right_zero_pad=True,
        auto_rescale_for_display=True,
        target_median_arrow_len=arrow_len,
        missing_mask=miss, missing_color="red", missing_alpha=0.25,
    )


# ── MSE computation ─────────────────────────────────────────────────

def ocean_mse(pred, gt, ocean_mask):
    """MSE over all ocean cells (2, H, W)."""
    ocean_b = ocean_mask.astype(bool)
    diff = pred[:, ocean_b] - gt[:, ocean_b]
    return float((diff ** 2).mean())


# ── main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Bulk evaluation: GP-Diff (S6 RePaint), DDPM Composite, GP, V-CNN")
    parser.add_argument("--reveal-pct", type=float, default=0.1,
                        help="Observation coverage %% (default: 0.1)")
    parser.add_argument("--n-samples", type=int, default=100,
                        help="Number of validation samples")
    parser.add_argument("--t-composite", type=int, default=200,
                        help="DDPM timestep for single-step composite")
    parser.add_argument("--n-ensemble", type=int, default=5,
                        help="DDPM ensemble size for composite")
    # S6 RePaint parameters
    parser.add_argument("--max-stages", type=int, default=6,
                        help="Number of S6 RePaint stages")
    parser.add_argument("--t-start-s6", type=int, default=75,
                        help="S6 stage 1 noise timestep")
    parser.add_argument("--t-refine-s6", type=int, default=50,
                        help="S6 stages 2+ noise timestep")
    parser.add_argument("--resample-steps", type=int, default=5,
                        help="RePaint resampling iterations per timestep")
    parser.add_argument("--noise-floor", type=float, default=0.2,
                        help="S6 stage 1 noise floor")
    parser.add_argument("--noise-floor-refine", type=float, default=0.3,
                        help="S6 stages 2+ noise floor")
    parser.add_argument("--var-decay", type=float, default=0.1,
                        help="Variance decay factor per S6 stage")
    parser.add_argument("--gamma", type=float, default=3.0,
                        help="Gamma for variance-adaptive noise weighting")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for mask generation")
    parser.add_argument("--skip-plots", action="store_true",
                        help="Skip generating quiver plots (just compute MSE)")
    parser.add_argument("--arrow-len", type=float, default=0.9)
    parser.add_argument("--step", type=int, default=2,
                        help="Quiver subsampling step")
    parser.add_argument("--resume", action="store_true",
                        help="Skip samples whose tensors.pt already exists")
    args = parser.parse_args()

    tag = f"{args.reveal_pct:.1f}pct_n{args.n_samples}_tc{args.t_composite}_s6"
    out_dir = RESULTS_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {dev}")
    print(f"Output: {out_dir}")
    print(f"Coverage: {args.reveal_pct}%, Composite t={args.t_composite}, Ensemble={args.n_ensemble}")
    print(f"GP-Diff S6: {args.max_stages} stages, t_start={args.t_start_s6}, "
          f"t_refine={args.t_refine_s6}, resample={args.resample_steps}, "
          f"gamma={args.gamma}")

    # ── load data & models ─────────────────────────────────────────
    print("Loading data …")
    val = load_data()
    n_val = val.shape[0]

    print("Loading models …")
    ddpm = load_ddpm(dev)
    uncond_ddpm = load_uncond_ddpm(dev)
    gpcnn_model = load_gp_cnn(dev)
    vcnn_model = load_vcnn(dev)
    border_gen = BorderMaskGenerator()

    # Pick n_samples random validation indices
    rng = np.random.default_rng(args.seed)
    val_indices = rng.choice(n_val, size=args.n_samples, replace=False)
    val_indices.sort()

    # Storage
    all_mse = {"gp": [], "vcnn": [], "composite": [], "gpdiff": []}
    all_records = []

    print(f"\nRunning {args.n_samples} samples …")
    print(f"{'='*100}")

    t0_global = time.time()

    for run_i, vi in enumerate(val_indices):
        t0 = time.time()
        vi = int(vi)

        sample_dir = out_dir / f"sample_{run_i:03d}_vi{vi}"

        # ── resume: skip if already computed ───────────────────────
        if args.resume and (sample_dir / "tensors.pt").exists():
            cached = torch.load(sample_dir / "tensors.pt", map_location="cpu",
                                weights_only=False)
            mse = cached["mse"]
            all_mse["gp"].append(mse["gp_mse"])
            all_mse["vcnn"].append(mse["vcnn_mse"])
            all_mse["composite"].append(mse["composite_mse"])
            all_mse["gpdiff"].append(mse["gpdiff_mse"])
            all_records.append(mse)
            print(f"  [{run_i+1:3d}/{args.n_samples}] vi={vi:4d} CACHED "
                  f"GP={mse['gp_mse']:.5f}  VCNN={mse['vcnn_mse']:.5f}  "
                  f"Comp={mse['composite_mse']:.5f}  GPDiff={mse['gpdiff_mse']:.5f}")
            continue

        gt = val[vi].numpy()  # (2, 44, 94)
        ocean_mask = ((np.abs(gt[0]) + np.abs(gt[1])) > 1e-8).astype(np.float32)

        # Fresh mask per sample (seeded deterministically)
        sample_rng = np.random.default_rng(args.seed + vi)
        obs_mask = random_mask(ocean_mask, args.reveal_pct, sample_rng)
        n_obs = int(obs_mask.sum())

        # ── GP ─────────────────────────────────────────────────────
        gp_mean, gp_var = compute_gp(gt, obs_mask, ocean_mask)
        gp_mse = ocean_mse(gp_mean, gt, ocean_mask)

        # ── V-CNN ──────────────────────────────────────────────────
        vcnn_pred = predict_vcnn(vcnn_model, gt, obs_mask, ocean_mask, dev)
        vcnn_mse = ocean_mse(vcnn_pred, gt, ocean_mask)

        # ── GP-CNN → DDPM Composite (single-step) ─────────────────
        gpcnn_full, gpcnn_std = predict_gp_cnn(gpcnn_model, gp_mean, gp_var,
                                                obs_mask, ocean_mask, dev)
        miss_mask = build_miss_mask(obs_mask, ocean_mask, border_gen).to(dev)

        ens_mean_std, _ = ensemble_single_step(
            ddpm, gpcnn_std.to(dev), miss_mask, args.t_composite,
            args.n_ensemble, dev)
        comp = variance_composite(gpcnn_full.cpu(), ens_mean_std.cpu(), gp_var, ocean_mask)
        comp_np = comp[0, :, :OCEAN_H, :OCEAN_W].numpy()
        comp_mse = ocean_mse(comp_np, gt, ocean_mask)

        # ── GP-Diff: S6 multi-stage adaptive RePaint ───────────────
        gpdiff_np = s6_repaint(
            uncond_ddpm, gt, obs_mask, ocean_mask, gp_mean, gp_var,
            border_gen, dev,
            max_stages=args.max_stages,
            t_start=args.t_start_s6,
            t_refine=args.t_refine_s6,
            noise_floor=args.noise_floor,
            noise_floor_refine=args.noise_floor_refine,
            var_decay=args.var_decay,
            gamma=args.gamma,
            resample_steps=args.resample_steps,
            seed=args.seed + vi,
        )
        gpdiff_mse = ocean_mse(gpdiff_np, gt, ocean_mask)

        # ── record ─────────────────────────────────────────────────
        all_mse["gp"].append(gp_mse)
        all_mse["vcnn"].append(vcnn_mse)
        all_mse["composite"].append(comp_mse)
        all_mse["gpdiff"].append(gpdiff_mse)

        record = {
            "val_idx": vi,
            "n_obs": n_obs,
            "gp_mse": gp_mse,
            "vcnn_mse": vcnn_mse,
            "composite_mse": comp_mse,
            "gpdiff_mse": gpdiff_mse,
        }
        all_records.append(record)

        # ── save per-sample .pt ────────────────────────────────────
        sample_dir.mkdir(exist_ok=True)

        torch.save({
            "val_idx": vi,
            "n_obs": n_obs,
            "gt": gt,
            "obs_mask": obs_mask,
            "ocean_mask": ocean_mask,
            "gp_mean": gp_mean,
            "gp_var": gp_var,
            "vcnn": vcnn_pred,
            "composite": comp_np,
            "gpdiff": gpdiff_np,
            "mse": record,
        }, sample_dir / "tensors.pt")

        # ── quiver plots ──────────────────────────────────────────
        if not args.skip_plots:
            plot_quiver(gt, ocean_mask, f"GT (vi={vi})",
                        sample_dir / "gt.png",
                        arrow_len=args.arrow_len, step=args.step)
            plot_quiver(gp_mean, ocean_mask, f"GP (MSE={gp_mse:.6f})",
                        sample_dir / "gp.png", obs_mask=obs_mask,
                        arrow_len=args.arrow_len, step=args.step)
            plot_quiver(vcnn_pred, ocean_mask, f"V-CNN (MSE={vcnn_mse:.6f})",
                        sample_dir / "vcnn.png",
                        arrow_len=args.arrow_len, step=args.step)
            plot_quiver(comp_np, ocean_mask, f"Composite (MSE={comp_mse:.6f})",
                        sample_dir / "composite.png",
                        arrow_len=args.arrow_len, step=args.step)
            plot_quiver(gpdiff_np, ocean_mask, f"GP-Diff (MSE={gpdiff_mse:.6f})",
                        sample_dir / "gpdiff.png",
                        arrow_len=args.arrow_len, step=args.step)

        elapsed = time.time() - t0
        best = min(gp_mse, vcnn_mse, comp_mse, gpdiff_mse)
        winner = ["GP", "VCNN", "Composite", "GP-Diff"][
            [gp_mse, vcnn_mse, comp_mse, gpdiff_mse].index(best)]
        print(f"  [{run_i+1:3d}/{args.n_samples}] vi={vi:4d} ({n_obs}obs) "
              f"GP={gp_mse:.5f}  VCNN={vcnn_mse:.5f}  "
              f"Comp={comp_mse:.5f}  GPDiff={gpdiff_mse:.5f}  "
              f"→ {winner}  ({elapsed:.1f}s)")

    elapsed_total = time.time() - t0_global

    # ── summary ────────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print(f"BULK EVALUATION SUMMARY — {args.reveal_pct}% coverage, "
          f"{args.n_samples} samples, {elapsed_total:.0f}s total")
    print(f"{'='*100}")

    for method in ["gp", "vcnn", "composite", "gpdiff"]:
        vals = np.array(all_mse[method])
        avg = vals.mean()
        med = np.median(vals)
        std = vals.std()
        label = "GP-Diff(S6)" if method == "gpdiff" else method
        print(f"  {label:>12}:  avg MSE = {avg:.6f}  median MSE = {med:.6f}  std = {std:.6f}")

    # Win counts
    wins = {m: 0 for m in all_mse}
    for i in range(len(all_records)):
        mses = {m: all_mse[m][i] for m in all_mse}
        winner = min(mses, key=mses.get)
        wins[winner] += 1
    print(f"\n  Win counts (lowest MSE):")
    for m in wins:
        print(f"    {m:>12}: {wins[m]}/{args.n_samples}")

    # Relative to GP
    gp_avg = np.mean(all_mse["gp"])
    print(f"\n  Relative to GP (avg MSE ratio):")
    for method in ["gp", "vcnn", "composite", "gpdiff"]:
        ratio = np.mean(all_mse[method]) / gp_avg
        label = "GP-Diff(S6)" if method == "gpdiff" else method
        print(f"    {label:>12}: {ratio:.3f}x")

    # ── save summary ───────────────────────────────────────────────
    summary = {
        "args": vars(args),
        "records": all_records,
        "mse_arrays": {m: np.array(v) for m, v in all_mse.items()},
        "summary": {
            m: {"avg": float(np.mean(v)), "median": float(np.median(v)),
                "std": float(np.std(v))}
            for m, v in all_mse.items()
        },
        "wins": wins,
        "elapsed_s": elapsed_total,
    }
    summary_path = out_dir / "summary.pt"
    torch.save(summary, summary_path)
    print(f"\n  Saved summary → {summary_path}")
    print(f"  Per-sample .pt and plots → {out_dir}/sample_NNN_viXXX/")


if __name__ == "__main__":
    main()
