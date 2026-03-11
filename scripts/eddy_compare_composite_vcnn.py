#!/usr/bin/env python3
"""
Compare eddy detection: DDPM Composite vs Voronoi-CNN vs GP-CNN vs GP.

For each sample in the eddy-balanced evaluation set:
  1. Generate a random observation mask (same seed as random_mask_eval.py)
  2. Run GP regression → GP posterior mean + variance
  3. Run GP-CNN (diverse-trained) → deterministic velocity field
  4. Run DDPM ensemble → ensemble mean → variance-weighted composite
  5. Run Voronoi-CNN → deterministic velocity field
  6. Run Gamma1 eddy detection on each reconstruction + ground truth
  7. Match detected eddies to ground truth and tally TP/FP/FN

Usage:
    PYTHONPATH=. python scripts/eddy_compare_composite_vcnn.py [--n-samples 100] [--reveal-pct 1.0]
"""

import argparse
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from scripts.voronoi_cnn_model import VoronoiCNN, build_voronoi_input
from ddpm.neural_networks.ddpm import GaussianDDPM
from ddpm.neural_networks.unets.unet_film_attn import MyUNet_FiLM_Attn
from ddpm.helper_functions.standardize_data import ZScoreStandardizer
from ddpm.helper_functions.interpolation_tool import gp_fill
from ddpm.utils.eddy_detection import detect_eddies_gamma

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
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
    radius=8,
    gamma_threshold=0.65,
    min_area=25,
    shore_buffer=2,
    smooth_sigma=2.0,
    min_mean_speed_ratio=0.3,
    min_vorticity=0.03,
)

DDPM_WEIGHT_PATH = "experiments/06_gp_forward/gp_conditioned/results/inpaint_gaussian_t250_best_ema_weights.pt"
GP_CNN_CKPT = "results/gp_cnn_diverse/gp_cnn_diverse_best.pt"
VCNN_CKPT = "results/voronoi_cnn/voronoi_cnn_best.pt"
DDPM_EVAL_PT = "results/eddy_balanced_eval/bulk_eval_eddy_balanced_100.pt"
OUT_DIR = Path("results/eddy_compare")

BEST_T = 75  # best composite timestep from our sweeps (5%/10%). Use 200 for 1%


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_pickle_data(path="data.pickle"):
    with open(path, "rb") as f:
        train_np, val_np, _test_np = pickle.load(f)
    def to_tensor(arr):
        t = torch.from_numpy(np.ascontiguousarray(arr)).float()
        t = t.permute(3, 2, 1, 0)
        t = torch.nan_to_num(t, nan=0.0)
        return t
    return to_tensor(train_np), to_tensor(val_np)


# ---------------------------------------------------------------------------
# Random mask generation (same as random_mask_eval.py)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# GP computation
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_ddpm(device):
    network = MyUNet_FiLM_Attn(n_steps=N_STEPS, time_emb_dim=256, in_channels=5)
    ddpm = GaussianDDPM(network, n_steps=N_STEPS, min_beta=0.0001, max_beta=0.02, device=device)
    ddpm.load_state_dict(torch.load(DDPM_WEIGHT_PATH, map_location="cpu", weights_only=False))
    ddpm = ddpm.to(device)
    ddpm.eval()
    return ddpm


def load_gp_cnn(device):
    ckpt = torch.load(GP_CNN_CKPT, map_location="cpu", weights_only=False)
    cfg = ckpt["model_config"]
    model = VoronoiCNN(
        in_channels=cfg["in_channels"],
        out_channels=cfg["out_channels"],
        base_ch=cfg.get("base_ch", 32),
        depth=cfg.get("depth", 3),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def load_vcnn(device):
    ckpt = torch.load(VCNN_CKPT, map_location="cpu", weights_only=False)
    cfg = ckpt["model_config"]
    model = VoronoiCNN(**cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


# ---------------------------------------------------------------------------
# GP-CNN inference
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Voronoi-CNN inference
# ---------------------------------------------------------------------------

def vcnn_predict(model, vel_phys, obs_mask, ocean_mask, norm_mean, norm_std, device):
    """Run Voronoi-CNN on a single sample with arbitrary mask."""
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

    return pred_phys.squeeze(0).cpu()  # (2, H, W)


# ---------------------------------------------------------------------------
# DDPM ensemble + composite
# ---------------------------------------------------------------------------

def ddpm_single_step(ddpm, cond_std_field, missing_mask_1ch, t_val, seed, device):
    torch.manual_seed(seed)
    alpha_bar = ddpm.alpha_bars[t_val].to(device)
    noise = torch.randn_like(cond_std_field)
    noisy = alpha_bar.sqrt() * cond_std_field + (1 - alpha_bar).sqrt() * noise
    x_cond = torch.cat([noisy, missing_mask_1ch, cond_std_field], dim=1)
    time_tensor = torch.full((1, 1), t_val, device=device, dtype=torch.long)
    with torch.no_grad():
        return ddpm.network(x_cond, time_tensor)


def ddpm_ensemble(ddpm, cond_std_field, missing_mask_1ch, t_val, n_ens, device):
    preds = []
    for k in range(n_ens):
        pred = ddpm_single_step(ddpm, cond_std_field, missing_mask_1ch, t_val,
                                seed=42 + k * 1000, device=device)
        preds.append(pred)
    stack = torch.stack(preds, dim=0)
    return stack.mean(dim=0), stack.std(dim=0)


def unstd_ddpm(t):
    return ddpm_standardizer.unstandardize(t.squeeze(0)).unsqueeze(0)


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


def composite(cnn_phys, ens_phys, weight):
    return (1.0 - weight) * cnn_phys + weight * ens_phys


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


# ---------------------------------------------------------------------------
# Eddy detection + matching
# ---------------------------------------------------------------------------

def run_gamma1(vel):
    """Run Gamma1 eddy detection on (2, H, W) velocity field."""
    vel = torch.nan_to_num(vel, nan=0.0)
    eddies, gamma1, vort = detect_eddies_gamma(vel, **EDDY_PARAMS)
    return eddies


def match_eddies(gt_eddies, pred_eddies, dist_thresh=8.0):
    """Greedy nearest-center matching."""
    if not gt_eddies or not pred_eddies:
        return [], list(range(len(gt_eddies))), list(range(len(pred_eddies)))

    n_gt, n_pred = len(gt_eddies), len(pred_eddies)
    dist = np.zeros((n_gt, n_pred))
    for i, ge in enumerate(gt_eddies):
        for j, pe in enumerate(pred_eddies):
            dist[i, j] = np.sqrt((ge.center_y - pe.center_y) ** 2 +
                                  (ge.center_x - pe.center_x) ** 2)
    matches, used_gt, used_pred = [], set(), set()
    for _ in range(min(n_gt, n_pred)):
        best_d, bi, bj = float("inf"), -1, -1
        for i in range(n_gt):
            if i in used_gt:
                continue
            for j in range(n_pred):
                if j in used_pred:
                    continue
                if dist[i, j] < best_d:
                    best_d, bi, bj = dist[i, j], i, j
        if best_d <= dist_thresh:
            matches.append((bi, bj, best_d))
            used_gt.add(bi)
            used_pred.add(bj)
        else:
            break

    fn_list = [i for i in range(n_gt) if i not in used_gt]
    fp_list = [j for j in range(n_pred) if j not in used_pred]
    return matches, fn_list, fp_list


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--n-ensemble", type=int, default=10)
    parser.add_argument("--reveal-pct", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--timestep", type=int, default=None,
                        help="DDPM timestep for composite (default: auto-select by reveal-pct)")
    args = parser.parse_args()

    # Auto-select best timestep based on reveal percentage
    if args.timestep is not None:
        t_val = args.timestep
    elif args.reveal_pct <= 1.5:
        t_val = 200  # best for 1%
    else:
        t_val = 75   # best for 5% and 10%

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Load all models
    print("Loading DDPM...")
    ddpm = load_ddpm(device)

    print(f"Loading GP-CNN from {GP_CNN_CKPT}...")
    gpcnn, gpcnn_ckpt = load_gp_cnn(device)
    gpcnn_norm_mean = gpcnn_ckpt["norm_mean"].numpy()
    gpcnn_norm_std = gpcnn_ckpt["norm_std"].numpy()
    ocean_mask = gpcnn_ckpt["ocean_mask"]

    print(f"Loading Voronoi-CNN from {VCNN_CKPT}...")
    vcnn, vcnn_ckpt = load_vcnn(device)
    vcnn_norm_mean = vcnn_ckpt["norm_mean"].numpy()
    vcnn_norm_std = vcnn_ckpt["norm_std"].numpy()

    # Load data
    _, val_vel = load_pickle_data()

    # Same 100 eddy-balanced samples
    ddpm_data = torch.load(DDPM_EVAL_PT, map_location="cpu", weights_only=False)
    ddpm_samples = ddpm_data["samples"]
    val_indices = [s["val_idx"] for s in ddpm_samples]
    eddy_set = set(ddpm_data.get("eddy_indices", []))
    n_total = min(args.n_samples, len(val_indices))
    val_indices = val_indices[:n_total]

    n_ocean = int(ocean_mask.sum())
    n_reveal = max(1, round(n_ocean * args.reveal_pct / 100.0))
    print(f"\nRandom mask: {args.reveal_pct}% → {n_reveal}/{n_ocean} observed pixels")
    print(f"DDPM composite timestep: t={t_val}, ensemble N={args.n_ensemble}")
    print(f"Eddy params: {EDDY_PARAMS}")
    print(f"Samples: {n_total}")
    print(f"{'=' * 90}")

    rng = np.random.default_rng(args.seed)

    # Accumulators for each method
    methods = ["GP", "GP-CNN", "Composite", "VCNN"]
    stats = {m: {"tp": 0, "fp": 0, "fn": 0, "center_dists": [],
                 "mse_list": [], "eddy_mse": [], "clean_mse": []}
             for m in methods}

    total_gt_eddies = 0
    per_sample = []

    t0_global = time.time()

    for run_i, vi in enumerate(val_indices):
        t0 = time.time()
        gt_vel_small = val_vel[vi].numpy()  # (2, OCEAN_H, OCEAN_W)
        gt_t = torch.from_numpy(gt_vel_small)
        is_eddy = vi in eddy_set

        # Generate random mask
        obs_mask = generate_random_mask(ocean_mask, args.reveal_pct, rng)

        # Missing ocean mask for MSE computation
        known_mask = torch.from_numpy(obs_mask).bool()
        ocean_bool = torch.from_numpy(ocean_mask).bool()
        missing_ocean = ocean_bool & ~known_mask

        # ── Ground truth eddy detection ──
        gt_eddies = run_gamma1(gt_t)
        n_gt = len(gt_eddies)
        total_gt_eddies += n_gt

        # ── GP ──
        gp_mean, gp_var = compute_gp(gt_vel_small, obs_mask, ocean_mask)
        gp_t = torch.from_numpy(gp_mean)
        gp_mse = (gp_t - gt_t)[:, missing_ocean].pow(2).mean().item()
        gp_eddies = run_gamma1(gp_t)

        # ── GP-CNN ──
        gpcnn_phys, gpcnn_ddpm_std = gp_cnn_predict(
            gpcnn, gp_mean, gp_var, obs_mask, ocean_mask,
            gpcnn_norm_mean, gpcnn_norm_std, device
        )
        gpcnn_phys_cpu = gpcnn_phys.cpu()
        gpcnn_small = gpcnn_phys_cpu[0, :, :OCEAN_H, :OCEAN_W]
        gpcnn_mse = (gpcnn_small - gt_t)[:, missing_ocean].pow(2).mean().item()
        gpcnn_eddies = run_gamma1(gpcnn_small)

        # ── DDPM Composite ──
        missing_mask_1ch = build_ddpm_missing_mask(obs_mask, ocean_mask)
        missing_mask_dev = missing_mask_1ch.to(device)
        gpcnn_ddpm_std_dev = gpcnn_ddpm_std.to(device)

        ens_mean_std, _ = ddpm_ensemble(
            ddpm, gpcnn_ddpm_std_dev, missing_mask_dev,
            t_val, args.n_ensemble, device
        )
        ens_mean_phys = unstd_ddpm(ens_mean_std.cpu())
        w = compute_gp_var_weight(gp_var, ocean_mask)
        comp_phys = composite(gpcnn_phys_cpu, ens_mean_phys, w)
        comp_small = comp_phys[0, :, :OCEAN_H, :OCEAN_W]
        comp_mse = (comp_small - gt_t)[:, missing_ocean].pow(2).mean().item()
        comp_eddies = run_gamma1(comp_small)

        # ── Voronoi-CNN ──
        vcnn_pred = vcnn_predict(
            vcnn, gt_vel_small, obs_mask, ocean_mask,
            vcnn_norm_mean, vcnn_norm_std, device
        )
        vcnn_mse = (vcnn_pred - gt_t)[:, missing_ocean].pow(2).mean().item()
        vcnn_eddies = run_gamma1(vcnn_pred)

        # ── Match eddies for each method ──
        sample_info = {
            "val_idx": vi, "is_eddy": is_eddy, "n_gt_eddies": n_gt,
        }
        for method_name, method_eddies, method_mse in [
            ("GP", gp_eddies, gp_mse),
            ("GP-CNN", gpcnn_eddies, gpcnn_mse),
            ("Composite", comp_eddies, comp_mse),
            ("VCNN", vcnn_eddies, vcnn_mse),
        ]:
            if n_gt > 0:
                matches, fn_list, fp_list = match_eddies(gt_eddies, method_eddies)
                tp = len(matches)
                fn = len(fn_list)
                fp = len(fp_list)
                dists = [d for _, _, d in matches]
            else:
                tp, fn = 0, 0
                fp = len(method_eddies)
                dists = []

            stats[method_name]["tp"] += tp
            stats[method_name]["fp"] += fp
            stats[method_name]["fn"] += fn
            stats[method_name]["center_dists"].extend(dists)
            stats[method_name]["mse_list"].append(method_mse)
            if is_eddy:
                stats[method_name]["eddy_mse"].append(method_mse)
            else:
                stats[method_name]["clean_mse"].append(method_mse)

            sample_info[f"{method_name}_tp"] = tp
            sample_info[f"{method_name}_fp"] = fp
            sample_info[f"{method_name}_fn"] = fn
            sample_info[f"{method_name}_mse"] = method_mse
            sample_info[f"{method_name}_n_detected"] = len(method_eddies)

        per_sample.append(sample_info)

        elapsed = time.time() - t0
        tag = "EDDY" if is_eddy else "clean"
        gt_str = f"GT={n_gt}"
        det_str = " | ".join([
            f"GP={len(gp_eddies)}",
            f"CNN={len(gpcnn_eddies)}",
            f"Comp={len(comp_eddies)}",
            f"VCNN={len(vcnn_eddies)}",
        ])
        print(f"  [{run_i+1:3d}/{n_total}] vi={vi:5d} {tag:>5}  {gt_str}  {det_str}  ({elapsed:.1f}s)")

    elapsed_total = time.time() - t0_global
    print(f"\nDone: {n_total} samples in {elapsed_total:.1f}s")

    # ==================================================================
    # Results
    # ==================================================================
    print(f"\n{'=' * 100}")
    print(f"EDDY DETECTION COMPARISON — Random {args.reveal_pct}% masks, t={t_val}, N={args.n_ensemble}")
    print(f"{'=' * 100}")
    print(f"Total ground-truth eddies: {total_gt_eddies}")

    print(f"\n  {'Method':<12} {'TP':>4} {'FP':>4} {'FN':>4}  "
          f"{'Precision':>10} {'Recall':>8} {'F1':>8}  "
          f"{'Med Dist':>9}  {'Mean MSE':>10} {'MSE/GP':>8}")
    print(f"  {'-' * 95}")

    gp_mean_mse = np.mean(stats["GP"]["mse_list"])

    for m in methods:
        s = stats[m]
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        med_d = np.median(s["center_dists"]) if s["center_dists"] else float("nan")
        mean_mse = np.mean(s["mse_list"])
        ratio = mean_mse / gp_mean_mse

        print(f"  {m:<12} {tp:>4} {fp:>4} {fn:>4}  "
              f"{prec:>10.1%} {rec:>8.1%} {f1:>8.3f}  "
              f"{med_d:>8.2f}px  {mean_mse:>10.6f} {ratio:>7.3f}x")

    # Eddy vs non-eddy MSE breakdown
    print(f"\n  MSE breakdown:")
    print(f"  {'Method':<12} {'All':>10} {'Eddy':>10} {'Non-eddy':>10}")
    print(f"  {'-' * 45}")
    for m in methods:
        s = stats[m]
        all_mse = np.mean(s["mse_list"])
        eddy_mse = np.mean(s["eddy_mse"]) if s["eddy_mse"] else 0
        clean_mse = np.mean(s["clean_mse"]) if s["clean_mse"] else 0
        print(f"  {m:<12} {all_mse:>10.6f} {eddy_mse:>10.6f} {clean_mse:>10.6f}")

    # Head-to-head: Composite vs VCNN
    comp_wins_mse = sum(1 for r in per_sample
                        if r["Composite_mse"] < r["VCNN_mse"])
    vcnn_wins_mse = sum(1 for r in per_sample
                        if r["VCNN_mse"] < r["Composite_mse"])

    print(f"\n  HEAD-TO-HEAD: Composite vs VCNN")
    print(f"  MSE:  Composite wins {comp_wins_mse}/{n_total}, "
          f"VCNN wins {vcnn_wins_mse}/{n_total}")

    comp_tp_total = stats["Composite"]["tp"]
    vcnn_tp_total = stats["VCNN"]["tp"]
    comp_fp_total = stats["Composite"]["fp"]
    vcnn_fp_total = stats["VCNN"]["fp"]
    print(f"  Eddies found:  Composite TP={comp_tp_total} FP={comp_fp_total}, "
          f"VCNN TP={vcnn_tp_total} FP={vcnn_fp_total}")

    # Per-sample eddy detection comparison
    comp_better_eddy = 0
    vcnn_better_eddy = 0
    tied_eddy = 0
    for r in per_sample:
        if r["n_gt_eddies"] == 0:
            continue
        c_found = r["Composite_tp"]
        v_found = r["VCNN_tp"]
        if c_found > v_found:
            comp_better_eddy += 1
        elif v_found > c_found:
            vcnn_better_eddy += 1
        else:
            tied_eddy += 1

    n_eddy_samples = sum(1 for r in per_sample if r["n_gt_eddies"] > 0)
    print(f"  Per eddy-sample TP:  Composite better {comp_better_eddy}, "
          f"VCNN better {vcnn_better_eddy}, tied {tied_eddy}  "
          f"(of {n_eddy_samples} eddy samples)")

    # Save
    save_data = {
        "per_sample": per_sample,
        "stats": {m: {k: v for k, v in s.items()} for m, s in stats.items()},
        "config": {
            "reveal_pct": args.reveal_pct,
            "timestep": t_val,
            "n_ensemble": args.n_ensemble,
            "seed": args.seed,
            "n_samples": n_total,
            "eddy_params": EDDY_PARAMS,
        },
        "total_gt_eddies": total_gt_eddies,
    }
    save_path = OUT_DIR / f"eddy_compare_{args.reveal_pct}pct_t{t_val}.pt"
    torch.save(save_data, save_path)
    print(f"\nSaved to {save_path}")


if __name__ == "__main__":
    main()
