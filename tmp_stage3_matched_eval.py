#!/usr/bin/env python3
"""Matched 3-stage detached rollout inference + evaluation.

This inference pipeline mirrors how the stage-3 model was trained:
  - 3 stages with stage-aware embeddings (stage=0, 1, 2)
  - Stage 0: denoise from t_max=75, Stage 1-2: denoise from t_max=50
  - After each stage: repaste known observations
  - Voronoi fill as initialization (no GP)

Training does single-step x0 prediction per stage; inference uses the full
reverse chain for each stage (standard DDPM reverse from t_start to 0),
which is the matched generalization.

Usage:
    PYTHONPATH=. python3 tmp_stage3_matched_eval.py
    PYTHONPATH=. python3 tmp_stage3_matched_eval.py --n-samples 5 --coverage 1.0
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

BASE_DIR = Path(__file__).resolve().parent

from ddpm.neural_networks.ddpm import GaussianDDPM
from ddpm.neural_networks.unets.unet_xl_attn import MyUNet_Attn
from ddpm.helper_functions.standardize_data import ZScoreStandardizer
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

# Stage t_max values matching training config
MODEL_CONFIGS = {
    "baseline": {
        "fracs": [0.30, 0.20, 0.20],
        "weights": "experiments/11_bootstrap_rollout/voronoi_detached_stage3/results/inpaint_gaussian_t250_best_ema_weights.pt",
    },
    "maskxt_hightmax": {
        "fracs": [0.50, 0.40, 0.30],
        "weights": "experiments/11_bootstrap_rollout/voronoi_stage3_maskxt_hightmax/results/inpaint_gaussian_t250_best_ema_weights.pt",
    },
}
NUM_STAGES = 3
VCNN_CKPT = "results/voronoi_cnn/voronoi_cnn_best.pt"


def make_border_mask(shape, device):
    """Create border mask: 1 inside ocean subdomain, 0 outside.

    Replaces BorderMaskGenerator to avoid DDInitializer / data.pickle dependency.
    """
    _, _, h, w = shape
    m = torch.zeros(1, 1, h, w, device=device)
    m[:, :, :OCEAN_H, :OCEAN_W] = 1.0
    return m


# ── Args ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--coverage", type=float, default=0.5,
                    help="Percent of ocean cells observed (0.5 = 0.5%%)")
parser.add_argument("--n-samples", type=int, default=3)
parser.add_argument("--resample-steps", type=int, default=3,
                    help="RePaint resample iterations per timestep")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--no-vcnn", action="store_true",
                    help="Skip V-CNN comparison (faster)")
parser.add_argument("--model", choices=list(MODEL_CONFIGS.keys()), default="baseline",
                    help="Which model config to evaluate")
args = parser.parse_args()

_cfg = MODEL_CONFIGS[args.model]
STAGE_T_MAX_FRACS = _cfg["fracs"]
STAGE_T_MAX = [int(round(f * (N_STEPS - 1))) for f in STAGE_T_MAX_FRACS]
STAGE3_WEIGHTS = _cfg["weights"]

# ── Device ───────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Device: {device}")


# ── Data loading ─────────────────────────────────────────────────────
def load_val_data():
    """Load RAW validation data from data.pickle (true ground truth).

    data.pickle contains (94, 44, 2, N) arrays → transpose to (N, 2, 44, 94).
    NaN values (land) are replaced with 0.
    """
    import pickle
    with open(str(BASE_DIR / "data.pickle"), "rb") as f:
        _, _, test_raw = pickle.load(f)
    # (94, 44, 2, N) → (N, 2, 44, 94)
    test = np.transpose(test_raw, (3, 2, 1, 0)).astype(np.float32)
    test = np.nan_to_num(test, nan=0.0)
    return torch.from_numpy(test)


def get_ocean_mask(val_tensor):
    """Ocean mask from 44×94 domain (val_tensor is already in ocean subdomain)."""
    return (val_tensor[0].abs().sum(dim=0) > 1e-7).float().numpy()


def random_obs_mask(ocean_mask, pct, rng):
    """Generate random observation mask. 1 = observed, 0 = missing."""
    idx = np.argwhere(ocean_mask > 0.5)
    n = max(1, round(len(idx) * pct / 100.0))
    sel = rng.choice(len(idx), size=n, replace=False)
    m = np.zeros((OCEAN_H, OCEAN_W), dtype=np.float32)
    for i in sel:
        m[idx[i][0], idx[i][1]] = 1.0
    return m


# ── Voronoi fill ─────────────────────────────────────────────────────
def voronoi_fill(vel_obs, obs_mask, ocean_mask):
    """Voronoi nearest-neighbor fill. Returns (2, 44, 94) in physical space."""
    ky, kx = np.where(obs_mask > 0.5)
    if len(ky) == 0:
        return np.zeros_like(vel_obs)
    tree = cKDTree(np.stack([ky, kx], axis=1).astype(np.float64))
    gy, gx = np.mgrid[0:OCEAN_H, 0:OCEAN_W]
    _, idx = tree.query(np.stack([gy.ravel(), gx.ravel()], axis=1).astype(np.float64))
    idx = idx.reshape(OCEAN_H, OCEAN_W)
    filled = np.stack([vel_obs[0, ky, kx][idx], vel_obs[1, ky, kx][idx]], axis=0)
    return filled * ocean_mask


# ── V-CNN ────────────────────────────────────────────────────────────
def load_vcnn(dev):
    ck = torch.load(str(BASE_DIR / VCNN_CKPT), map_location="cpu", weights_only=False)
    m = VoronoiCNN(**ck["model_config"]).to(dev)
    m.load_state_dict(ck["model_state"])
    m.eval()
    return m


def predict_vcnn(model, vel_obs, obs_mask, ocean_mask, dev):
    vel_n = ((vel_obs - NM[:, None, None]) / NS[:, None, None]) * ocean_mask[None]
    vi = build_voronoi_input(vel_n, obs_mask, ocean_mask)
    with torch.no_grad():
        p = model(torch.from_numpy(vi).unsqueeze(0).to(dev))
    mt = torch.tensor(NM).view(1, 2, 1, 1).to(dev)
    st = torch.tensor(NS).view(1, 2, 1, 1).to(dev)
    om = torch.from_numpy(ocean_mask).float().to(dev).view(1, 1, OCEAN_H, OCEAN_W)
    phys = (p * st + mt) * om
    return phys.squeeze(0).cpu().numpy()


# ── Stage-3 DDPM model ──────────────────────────────────────────────
def load_stage3_ddpm():
    net = MyUNet_Attn(n_steps=N_STEPS, time_emb_dim=256, n_stage_tokens=3)
    ddpm = GaussianDDPM(net, n_steps=N_STEPS,
                        min_beta=0.0001, max_beta=0.02, device=device)
    state = torch.load(str(BASE_DIR / STAGE3_WEIGHTS), map_location="cpu",
                       weights_only=False)
    ddpm.load_state_dict(state)
    ddpm.to(device)
    ddpm.eval()
    return ddpm


# ── Matched 3-stage reverse diffusion ───────────────────────────────
def reverse_diffuse_stage(ddpm, x_init, t_start, stage_idx,
                          known_std, known_mask, miss_mask,
                          resample_steps=3):
    """Full DDPM reverse chain from t_start to 0 for one stage.

    Mirrors training: at each t, predicts x0 with the correct stage embedding.
    Uses RePaint-style known-region pasting at every step.

    Args:
        ddpm: GaussianDDPM model
        x_init: (1, 2, 64, 128) initial state in standardized space
        t_start: starting timestep (max t for this stage)
        stage_idx: 0, 1, or 2 — passed to network's stage embedding
        known_std: (1, 2, 64, 128) ground truth known values (standardized)
        known_mask: (1, 2, 64, 128) 1 = known, 0 = missing
        miss_mask: (1, 2, 64, 128) 1 = missing, 0 = known
        resample_steps: RePaint resample iterations

    Returns:
        (1, 2, 64, 128) denoised output in standardized space
    """
    stage_tensor = torch.tensor([stage_idx], device=device, dtype=torch.long)

    # Forward-diffuse the init to t_start
    alpha_bar_t = ddpm.alpha_bars[t_start]
    eps = torch.randn_like(x_init)
    x = alpha_bar_t.sqrt() * x_init + (1 - alpha_bar_t).sqrt() * eps

    for t in range(t_start, -1, -1):
        n_resample = resample_steps if t > 0 else 1

        for r in range(n_resample):
            alpha_t = ddpm.alphas[t]
            alpha_bar_t = ddpm.alpha_bars[t]
            beta_t = ddpm.betas[t]
            time_tensor = torch.full((1, 1), t, device=device, dtype=torch.long)

            # Network predicts x0 with stage embedding
            x0_pred = ddpm.network(x, time_tensor, stage=stage_tensor)

            if t > 0:
                alpha_bar_prev = ddpm.alpha_bars[t - 1]
                # Posterior mean
                coeff_x0 = (alpha_bar_prev.sqrt() * beta_t) / (1 - alpha_bar_t)
                coeff_xt = (alpha_t.sqrt() * (1 - alpha_bar_prev)) / (1 - alpha_bar_t)
                mu = coeff_x0 * x0_pred + coeff_xt * x
                # Posterior variance
                beta_tilde = ((1 - alpha_bar_prev) / (1 - alpha_bar_t)) * beta_t
                sigma_t = beta_tilde.sqrt()
                z = torch.randn_like(x)
                x_denoised = mu + sigma_t * z
            else:
                x_denoised = x0_pred

            # RePaint: paste forward-noised known region
            if t > 0:
                alpha_bar_prev = ddpm.alpha_bars[t - 1]
                noise_known = torch.randn_like(known_std)
                x_known = alpha_bar_prev.sqrt() * known_std + (1 - alpha_bar_prev).sqrt() * noise_known
                x = x_known * known_mask + x_denoised * miss_mask
            else:
                x = known_std * known_mask + x_denoised * miss_mask

            # Resample (RePaint noise-back step)
            if r < n_resample - 1 and t > 0:
                noise_back = torch.randn_like(x)
                x = alpha_t.sqrt() * x + (1 - alpha_t).sqrt() * noise_back

    return x


def single_step_denoise_stage(ddpm, x_init, t_val, stage_idx,
                               known_std, known_mask, miss_mask):
    """Single-step x0 prediction at a given t (exactly matches one training step).

    No reverse chain — just noise to t, predict x0, done.
    """
    stage_tensor = torch.tensor([stage_idx], device=device, dtype=torch.long)
    time_tensor = torch.full((1, 1), t_val, device=device, dtype=torch.long)

    alpha_bar_t = ddpm.alpha_bars[t_val]
    eps = torch.randn_like(x_init)
    x_t = alpha_bar_t.sqrt() * x_init + (1 - alpha_bar_t).sqrt() * eps

    x0_pred = ddpm.network(x_t, time_tensor, stage=stage_tensor)

    # Repaste known
    return known_std * known_mask + x0_pred * miss_mask


def run_single_step_3stage(ddpm, gt_ocean, obs_mask, ocean_mask, seed):
    """3-stage single-step inference (one x0 prediction per stage).

    Exactly mirrors one training forward pass:
      - Use t = t_max//2 (expected value of Uniform[0, t_max])
    """
    gt_full = np.zeros((1, 2, FULL_H, FULL_W), dtype=np.float32)
    gt_full[0, :, :OCEAN_H, :OCEAN_W] = gt_ocean * ocean_mask[None]
    known_std = standardizer(
        torch.from_numpy(gt_full).squeeze(0)).unsqueeze(0).to(device)

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

    vel_obs = gt_ocean * obs_mask[None]
    vor_fill = voronoi_fill(vel_obs, obs_mask, ocean_mask)
    vor_full = np.zeros((1, 2, FULL_H, FULL_W), dtype=np.float32)
    vor_full[0, :, :OCEAN_H, :OCEAN_W] = vor_fill
    vor_std = standardizer(
        torch.from_numpy(vor_full).squeeze(0)).unsqueeze(0).to(device)

    current = known_std * known_mask + vor_std * miss_mask

    torch.manual_seed(seed)
    with torch.no_grad():
        for stage_idx in range(NUM_STAGES):
            t_val = STAGE_T_MAX[stage_idx] // 2  # expected value of Uniform[0, t_max]
            current = single_step_denoise_stage(
                ddpm, current, t_val, stage_idx,
                known_std, known_mask, miss_mask,
            )

    result_phys = standardizer.unstandardize(current.squeeze(0).cpu())
    return result_phys[:, :OCEAN_H, :OCEAN_W].numpy()


def run_matched_3stage(ddpm, gt_ocean, obs_mask, ocean_mask, seed,
                       resample_steps=3):
    """Run the matched 3-stage detached rollout inference.

    Stage 0: Voronoi → noise to t=75 → full reverse → repaste known
    Stage 1: Stage 0 output → noise to t=50 → full reverse → repaste known
    Stage 2: Stage 1 output → noise to t=50 → full reverse → final output

    Args:
        ddpm: loaded stage-3 DDPM model
        gt_ocean: (2, 44, 94) ground truth velocity in physical space
        obs_mask: (44, 94) observation mask (1 = observed)
        ocean_mask: (44, 94) ocean mask (1 = ocean)
        seed: random seed for reproducibility
        resample_steps: RePaint resample steps per timestep

    Returns:
        (2, 44, 94) inpainted velocity in physical space
    """
    # Build known values in standardized full grid
    gt_full = np.zeros((1, 2, FULL_H, FULL_W), dtype=np.float32)
    gt_full[0, :, :OCEAN_H, :OCEAN_W] = gt_ocean * ocean_mask[None]
    known_std = standardizer(
        torch.from_numpy(gt_full).squeeze(0)).unsqueeze(0).to(device)

    # Build masks in full grid
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

    # Voronoi fill as initial state (in physical space → standardized)
    vel_obs = gt_ocean * obs_mask[None]
    vor_fill = voronoi_fill(vel_obs, obs_mask, ocean_mask)
    vor_full = np.zeros((1, 2, FULL_H, FULL_W), dtype=np.float32)
    vor_full[0, :, :OCEAN_H, :OCEAN_W] = vor_fill
    vor_std = standardizer(
        torch.from_numpy(vor_full).squeeze(0)).unsqueeze(0).to(device)

    # Build composite: known = GT, unknown = Voronoi (standardized)
    current = known_std * known_mask + vor_std * miss_mask

    torch.manual_seed(seed)
    with torch.no_grad():
        for stage_idx in range(NUM_STAGES):
            t_start = STAGE_T_MAX[stage_idx]
            current = reverse_diffuse_stage(
                ddpm, current, t_start, stage_idx,
                known_std, known_mask, miss_mask,
                resample_steps=resample_steps,
            )
            # Repaste known after each stage (matching training)
            current = known_std * known_mask + current * miss_mask

    # Convert back to physical space
    result_phys = standardizer.unstandardize(current.squeeze(0).cpu())
    return result_phys[:, :OCEAN_H, :OCEAN_W].numpy()


def run_single_stage(ddpm, gt_ocean, obs_mask, ocean_mask, seed,
                     t_start=75, stage_idx=0, resample_steps=3):
    """Run single-stage x0 reverse diffusion (baseline comparison).

    Same model but just one stage with stage=0, like the pre-trained model
    would do without rollout.
    """
    gt_full = np.zeros((1, 2, FULL_H, FULL_W), dtype=np.float32)
    gt_full[0, :, :OCEAN_H, :OCEAN_W] = gt_ocean * ocean_mask[None]
    known_std = standardizer(
        torch.from_numpy(gt_full).squeeze(0)).unsqueeze(0).to(device)

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

    vel_obs = gt_ocean * obs_mask[None]
    vor_fill = voronoi_fill(vel_obs, obs_mask, ocean_mask)
    vor_full = np.zeros((1, 2, FULL_H, FULL_W), dtype=np.float32)
    vor_full[0, :, :OCEAN_H, :OCEAN_W] = vor_fill
    vor_std = standardizer(
        torch.from_numpy(vor_full).squeeze(0)).unsqueeze(0).to(device)

    current = known_std * known_mask + vor_std * miss_mask

    torch.manual_seed(seed)
    with torch.no_grad():
        current = reverse_diffuse_stage(
            ddpm, current, t_start, stage_idx,
            known_std, known_mask, miss_mask,
            resample_steps=resample_steps,
        )
        current = known_std * known_mask + current * miss_mask

    result_phys = standardizer.unstandardize(current.squeeze(0).cpu())
    return result_phys[:, :OCEAN_H, :OCEAN_W].numpy()


# ── Metrics ──────────────────────────────────────────────────────────
def ocean_mse(pred, gt, ocean_mask):
    ocean_b = ocean_mask.astype(bool)
    diff = pred[:, ocean_b] - gt[:, ocean_b]
    return float((diff ** 2).mean())


# ══════════════════════════════════════════════════════════════════════
def main():
    coverage = args.coverage
    n_samples = args.n_samples

    print(f"\n{'='*72}")
    print(f"MATCHED 3-STAGE DETACHED ROLLOUT — EVALUATION")
    print(f"Coverage: {coverage}%  |  Samples: {n_samples}  |  "
          f"Resample: {args.resample_steps}")
    print(f"Stage t_max: {STAGE_T_MAX}  (fracs: {STAGE_T_MAX_FRACS})")
    print(f"{'='*72}")

    val = load_val_data()
    ocean_mask = get_ocean_mask(val)
    n_ocean = int(ocean_mask.sum())
    n_obs = max(1, round(n_ocean * coverage / 100.0))
    print(f"  Ocean cells: {n_ocean}, Observations: {n_obs}")

    rng = np.random.default_rng(seed=7777)
    val_indices = rng.choice(val.shape[0], size=min(n_samples, val.shape[0]),
                             replace=False)
    val_indices.sort()

    print("\nLoading models...")
    stage3_ddpm = load_stage3_ddpm()
    print(f"  Stage-3 DDPM loaded (n_stage_tokens=3, weights: {STAGE3_WEIGHTS})")

    vcnn_model = None
    if not args.no_vcnn:
        vcnn_model = load_vcnn(device)
        print("  V-CNN loaded")

    methods = ["Voronoi", "1-Step-3S", "1-Stage", "3-Stage"]
    if vcnn_model:
        methods.insert(1, "V-CNN")
    results = {m: [] for m in methods}

    header = f"{'#':>3} {'ValIdx':>7}"
    for m in methods:
        header += f" {m:>10}"
    header += f" {'Time':>7}"
    print(f"\n{header}")
    print("-" * (len(header) + 5))

    t0_global = time.time()
    for i, vi in enumerate(val_indices):
        gt = val[vi, :, :OCEAN_H, :OCEAN_W].numpy()
        seed = args.seed + vi
        obs_mask = random_obs_mask(ocean_mask, coverage,
                                   np.random.default_rng(seed=seed))
        vel_obs = gt * obs_mask[None]

        t0 = time.time()

        # Voronoi baseline
        vor = voronoi_fill(vel_obs, obs_mask, ocean_mask)
        vor_mse = ocean_mse(vor, gt, ocean_mask)
        results["Voronoi"].append(vor_mse)

        # V-CNN
        if vcnn_model:
            vcnn_pred = predict_vcnn(vcnn_model, vel_obs, obs_mask,
                                     ocean_mask, device)
            vcnn_mse = ocean_mse(vcnn_pred, gt, ocean_mask)
            results["V-CNN"].append(vcnn_mse)

        # Single-step 3-stage (one x0 pred per stage, no reverse chain)
        ss_pred = run_single_step_3stage(
            stage3_ddpm, gt, obs_mask, ocean_mask, seed,
        )
        ss_mse = ocean_mse(ss_pred, gt, ocean_mask)
        results["1-Step-3S"].append(ss_mse)

        # Single-stage reverse chain (same model, stage=0, t=75)
        s1_pred = run_single_stage(
            stage3_ddpm, gt, obs_mask, ocean_mask, seed,
            t_start=STAGE_T_MAX[0], stage_idx=0,
            resample_steps=args.resample_steps,
        )
        s1_mse = ocean_mse(s1_pred, gt, ocean_mask)
        results["1-Stage"].append(s1_mse)

        # Matched 3-stage
        s3_pred = run_matched_3stage(
            stage3_ddpm, gt, obs_mask, ocean_mask, seed,
            resample_steps=args.resample_steps,
        )
        s3_mse = ocean_mse(s3_pred, gt, ocean_mask)
        results["3-Stage"].append(s3_mse)

        elapsed = time.time() - t0
        row = f"{i+1:>3} {vi:>7}"
        for m in methods:
            row += f" {results[m][-1]:>10.5f}"
        row += f" {elapsed:>6.1f}s"
        print(row)

    total = time.time() - t0_global

    # Summary
    print(f"\n{'='*72}")
    print(f"SUMMARY — {coverage}% coverage, {len(val_indices)} samples, "
          f"{total:.0f}s total")
    print(f"{'='*72}")
    baseline_key = "V-CNN" if vcnn_model else "Voronoi"
    baseline_mean = np.mean(results[baseline_key])

    print(f"{'Method':<20} {'Mean MSE':<12} {'vs ' + baseline_key:<12}")
    print("-" * 45)
    for m in methods:
        arr = np.array(results[m])
        ratio = arr.mean() / baseline_mean if baseline_mean > 0 else float("inf")
        print(f"{m:<20} {arr.mean():.6f}     {ratio:.3f}x")

    # Win rates
    if len(val_indices) >= 2:
        print(f"\nWin rates (lower MSE per sample):")
        s3 = np.array(results["3-Stage"])
        s1 = np.array(results["1-Stage"])
        ss = np.array(results["1-Step-3S"])
        vor = np.array(results["Voronoi"])
        n = len(s3)
        print(f"  1-Step-3S beats Voronoi: {(ss < vor).sum()}/{n}")
        print(f"  1-Stage beats Voronoi:   {(s1 < vor).sum()}/{n}")
        print(f"  3-Stage beats 1-Stage:   {(s3 < s1).sum()}/{n}")
        print(f"  3-Stage beats Voronoi:   {(s3 < vor).sum()}/{n}")
        if vcnn_model:
            vcnn_arr = np.array(results["V-CNN"])
            print(f"  3-Stage beats V-CNN:   {(s3 < vcnn_arr).sum()}/{n}")


if __name__ == "__main__":
    main()
