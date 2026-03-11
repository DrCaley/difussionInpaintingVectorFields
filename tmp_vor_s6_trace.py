#!/usr/bin/env python3
"""Trace iterative S6 denoising for the bootstrap Voronoi-trained DDPM.

Saves:
  1. A .pt artifact with GT, mask, Voronoi prior, and per-stage outputs
  2. Per-stage PNG panels rendered from the saved tensors

Default sample/mask settings match the earlier 0.5% benchmark setup.
"""

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from data_prep.data_initializer import DDInitializer
from ddpm.helper_functions.masks.border_mask import BorderMaskGenerator
from ddpm.helper_functions.standardize_data import ZScoreStandardizer
from ddpm.neural_networks.ddpm import GaussianDDPM
from ddpm.neural_networks.unets.unet_xl_attn import MyUNet_Attn
from ddpm.utils.inpainting_utils import repaint_gp_init_adaptive
from ddpm.utils.noise_utils import get_noise_strategy
from plots.visualization_tools.standard_plots import plot_inpaint_panels

OCEAN_H, OCEAN_W = 44, 94
FULL_H, FULL_W = 64, 128
N_STEPS = 250

U_MEAN, U_STD = -0.06929559429949586, 0.1358005549716049
V_MEAN, V_STD = -0.0323937796117541, 0.08899177232117582
standardizer = ZScoreStandardizer(U_MEAN, U_STD, V_MEAN, V_STD)

DEFAULT_WEIGHTS = (
    "experiments/11_bootstrap_rollout/voronoi_multistep_bootstrap/results/"
    "inpaint_gaussian_t250_best_ema_weights.remote.pt"
)
DEFAULT_OUTDIR = (
    "experiments/11_bootstrap_rollout/voronoi_multistep_bootstrap/results/s6_trace"
)
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

parser = argparse.ArgumentParser()
parser.add_argument("--coverage", type=float, default=0.5)
parser.add_argument("--val-idx", type=int, default=482,
                    help="Validation index to trace")
parser.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS,
                    help="Bootstrap Voronoi-trained DDPM weights")
parser.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR)
args = parser.parse_args()

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

noise_strategy = get_noise_strategy("gaussian")
border_gen = BorderMaskGenerator()


def load_val_data():
    with open(str(BASE_DIR / "data.pickle"), "rb") as f:
        _train, val, _test = pickle.load(f)
    return torch.nan_to_num(
        torch.from_numpy(np.ascontiguousarray(val)).float().permute(3, 2, 1, 0),
        nan=0.0,
    )


def get_ocean_mask(val_tensor):
    sample = val_tensor[0]
    return (sample.abs() > 1e-5).any(dim=0).float().numpy()


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
        return np.zeros_like(vel), np.ones_like(vel)

    obs_coords = np.stack([ky, kx], axis=1).astype(np.float64)
    tree = cKDTree(obs_coords)
    gy, gx = np.mgrid[0:OCEAN_H, 0:OCEAN_W]
    grid_coords = np.stack([gy.ravel(), gx.ravel()], axis=1).astype(np.float64)
    dist, idx = tree.query(grid_coords, k=1)
    dist = dist.reshape(OCEAN_H, OCEAN_W)
    idx = idx.reshape(OCEAN_H, OCEAN_W)

    obs_u = vel[0, ky, kx]
    obs_v = vel[1, ky, kx]
    vor_u = obs_u[idx] * ocean_mask
    vor_v = obs_v[idx] * ocean_mask
    vor_mean = np.stack([vor_u, vor_v], axis=0)

    dist_sq = (dist ** 2) * ocean_mask
    dist_var = np.stack([dist_sq, dist_sq], axis=0)
    return vor_mean, dist_var


def load_uncond_model(weight_path):
    net = MyUNet_Attn(n_steps=N_STEPS, time_emb_dim=256)
    ddpm = GaussianDDPM(
        net,
        n_steps=N_STEPS,
        min_beta=0.0001,
        max_beta=0.02,
        device=device,
    )
    state = torch.load(str(BASE_DIR / weight_path), map_location="cpu", weights_only=False)
    ddpm.load_state_dict(state)
    ddpm.to(device)
    ddpm.eval()
    return ddpm


def ocean_mse(pred, gt, ocean_mask):
    ocean_b = ocean_mask.astype(bool)
    diff = pred[:, ocean_b] - gt[:, ocean_b]
    return float((diff ** 2).mean())


def build_inputs(gt_np, obs_mask, ocean_mask, prior_mean, prior_var):
    gt_full = np.zeros((1, 2, FULL_H, FULL_W), dtype=np.float32)
    gt_full[0, :, :OCEAN_H, :OCEAN_W] = gt_np * ocean_mask[None]
    input_image = standardizer(torch.from_numpy(gt_full).squeeze(0)).unsqueeze(0).to(device)

    prior_full = np.zeros((1, 2, FULL_H, FULL_W), dtype=np.float32)
    prior_full[0, :, :OCEAN_H, :OCEAN_W] = prior_mean * ocean_mask[None]
    prior_std = standardizer(torch.from_numpy(prior_full).squeeze(0)).unsqueeze(0).to(device)

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

    return input_image, prior_std, missing_mask, var_t


def trace_s6(ddpm_model, gt_np, obs_mask, ocean_mask, prior_mean, prior_var, sample_seed):
    input_image, current_prior, missing_mask, current_var = build_inputs(
        gt_np, obs_mask, ocean_mask, prior_mean, prior_var
    )

    stages = []
    for stage in range(1, S6_DEFAULTS["max_stages"] + 1):
        if stage == 1:
            t_s = S6_DEFAULTS["t_start"]
            nf = S6_DEFAULTS["noise_floor"]
            seed_s = sample_seed
        else:
            t_s = S6_DEFAULTS["t_refine"]
            nf = S6_DEFAULTS["noise_floor_refine"]
            seed_s = sample_seed + stage * 10000
            current_var = current_var * S6_DEFAULTS["var_decay"]

        torch.manual_seed(seed_s)
        with torch.no_grad():
            stage_out = repaint_gp_init_adaptive(
                ddpm_model,
                input_image,
                missing_mask,
                gp_image=current_prior,
                gp_variance_map=current_var,
                t_start=t_s,
                noise_floor=nf,
                n_samples=1,
                device=device,
                noise_strategy=noise_strategy,
                prediction_target="x0",
                resample_steps=S6_DEFAULTS["resample_steps"],
                project_div_free=False,
                anneal_floor=False,
                gamma=S6_DEFAULTS["gamma"],
            )

        pred_phys_full = standardizer.unstandardize(stage_out.squeeze(0).cpu())
        pred_phys_full[:, :OCEAN_H, :OCEAN_W] *= torch.from_numpy(ocean_mask).float().unsqueeze(0)
        pred_np = pred_phys_full[:, :OCEAN_H, :OCEAN_W].numpy()
        mse = ocean_mse(pred_np, gt_np, ocean_mask)
        stages.append(
            {
                "stage": stage,
                "t_start": int(t_s),
                "noise_floor": float(nf),
                "seed": int(seed_s),
                "mse": float(mse),
                "pred_std": stage_out.detach().cpu(),
                "pred_phys": pred_phys_full.unsqueeze(0).clone(),
            }
        )
        current_prior = stage_out.clone()

    prior_phys = standardizer.unstandardize(current_prior.squeeze(0).cpu())
    prior_phys[:, :OCEAN_H, :OCEAN_W] *= torch.from_numpy(ocean_mask).float().unsqueeze(0)
    return stages, missing_mask.detach().cpu(), prior_phys.unsqueeze(0)


def render_from_saved(saved_path, out_dir):
    payload = torch.load(saved_path, map_location="cpu", weights_only=False)
    gt = payload["ground_truth"]
    missing_mask = payload["missing_mask"]
    methods = {"Voronoi prior": payload["voronoi_prior"]}
    mse = {"Voronoi prior": payload["voronoi_mse"]}
    extra_titles = {}

    for stage in payload["stages"]:
        name = f"S{stage['stage']}"
        methods[name] = stage["pred_phys"]
        mse[name] = stage["mse"]
        extra_titles[name] = f"(t={stage['t_start']}, floor={stage['noise_floor']:.1f})"

    plot_inpaint_panels(
        gt=gt,
        missing_mask=missing_mask,
        methods=methods,
        mse=mse,
        out_dir=out_dir,
        prefix=f"val{payload['val_idx']}_s6_trace",
        mask_label=f"random {payload['coverage_pct']}% coverage",
        extra_titles=extra_titles,
        mark_eddies=False,
    )


def main():
    os.makedirs(BASE_DIR / args.outdir, exist_ok=True)

    val = load_val_data()
    ocean_mask = get_ocean_mask(val)
    gt = val[args.val_idx].numpy()
    sample_seed = S6_DEFAULTS["seed"] + args.val_idx

    mask_rng = np.random.default_rng(seed=sample_seed)
    obs_mask = random_mask(ocean_mask, args.coverage, mask_rng)
    vel_obs = gt * obs_mask[None]
    vor_mean, vor_var = compute_voronoi(vel_obs, obs_mask, ocean_mask)
    vor_mse = ocean_mse(vor_mean, gt, ocean_mask)

    ddpm_model = load_uncond_model(args.weights)

    t0 = time.time()
    stages, missing_mask, _ = trace_s6(
        ddpm_model,
        gt,
        obs_mask,
        ocean_mask,
        vor_mean,
        vor_var,
        sample_seed,
    )
    elapsed = time.time() - t0

    gt_tensor = torch.zeros(1, 2, FULL_H, FULL_W)
    gt_tensor[:, :, :OCEAN_H, :OCEAN_W] = torch.from_numpy(gt * ocean_mask[None]).unsqueeze(0)

    vor_tensor = torch.zeros(1, 2, FULL_H, FULL_W)
    vor_tensor[:, :, :OCEAN_H, :OCEAN_W] = torch.from_numpy(vor_mean * ocean_mask[None]).unsqueeze(0)

    obs_tensor = torch.zeros(1, 1, FULL_H, FULL_W)
    obs_tensor[:, :, :OCEAN_H, :OCEAN_W] = torch.from_numpy(obs_mask * ocean_mask).unsqueeze(0).unsqueeze(0)

    cov_tag = str(args.coverage).replace(".", "p")
    out_base = BASE_DIR / args.outdir / f"val{args.val_idx}_cov{cov_tag}_s6_trace"
    pt_path = Path(f"{out_base}.pt")
    torch.save(
        {
            "val_idx": int(args.val_idx),
            "coverage_pct": float(args.coverage),
            "weights": args.weights,
            "elapsed_sec": float(elapsed),
            "s6_params": S6_DEFAULTS,
            "ground_truth": gt_tensor,
            "missing_mask": missing_mask,
            "observation_mask": obs_tensor,
            "voronoi_prior": vor_tensor,
            "voronoi_mse": float(vor_mse),
            "stages": stages,
        },
        pt_path,
    )

    render_from_saved(pt_path, str(out_base.parent))

    print("\nS6 trace saved:")
    print(f"  PT:  {pt_path}")
    print(f"  PNG dir: {out_base.parent}")
    print(f"  Sample val_idx: {args.val_idx}")
    print(f"  Coverage: {args.coverage}%")
    print(f"  Voronoi prior MSE: {vor_mse:.6f}")
    print(f"  Runtime: {elapsed:.1f}s")
    print("\nStage MSEs:")
    for stage in stages:
        print(
            f"  S{stage['stage']}: mse={stage['mse']:.6f} "
            f"(t={stage['t_start']}, floor={stage['noise_floor']:.1f})"
        )


if __name__ == "__main__":
    main()
