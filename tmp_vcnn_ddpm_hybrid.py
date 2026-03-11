#!/usr/bin/env python3
"""
Hybrid test: VCNN + DDPM refinement.

Instead of GP → DDPM, we do VCNN → DDPM.
The VCNN provides a much better starting point (0.269x GP vs 1.0x GP),
so the DDPM refines from closer to ground truth.

Tests:
  1. VCNN-only (baseline for comparison)
  2. VCNN → DDPM single-step refinement (various t)
  3. VCNN → DDPM ensemble refinement
  4. Uncertainty quantification: ensemble std as confidence map
"""
import sys, os, pickle, time
import numpy as np
import torch

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from ddpm.neural_networks.ddpm import GaussianDDPM
from ddpm.neural_networks.unets.unet_film_attn import MyUNet_FiLM_Attn
from ddpm.helper_functions.masks.border_mask import BorderMaskGenerator
from ddpm.helper_functions.standardize_data import ZScoreStandardizer
from ddpm.helper_functions.interpolation_tool import gp_fill
from data_prep.data_initializer import DDInitializer
from scripts.voronoi_cnn_model import VoronoiCNN, build_voronoi_input

# ── Constants ──
N_STEPS = 250
U_MEAN, U_STD = -0.06929559429949586, 0.1358005549716049
V_MEAN, V_STD = -0.0323937796117541, 0.08899177232117582
OCEAN_H, OCEAN_W = 44, 94

standardizer = ZScoreStandardizer(U_MEAN, U_STD, V_MEAN, V_STD)
norm_mean = np.array([U_MEAN, V_MEAN], dtype=np.float32)
norm_std = np.array([U_STD, V_STD], dtype=np.float32)

dd = DDInitializer()
device = dd.get_device()
val_data = dd.get_validation_data()
dd_std = dd.get_standardizer()

RESULTS_DIR = os.path.join(BASE, "experiments/06_gp_forward/gp_conditioned/results")
GPDIFF_PT = os.path.join(BASE, "results/eddy_balanced_eval/bulk_eval_eddy_balanced_100.pt")
gpdiff_data = torch.load(GPDIFF_PT, map_location="cpu", weights_only=False)
val_indices = gpdiff_data["val_indices"]

n_test = 10

# ── Load DDPM ──
def load_ddpm(weight_path):
    network = MyUNet_FiLM_Attn(n_steps=N_STEPS, time_emb_dim=256, in_channels=5)
    ddpm = GaussianDDPM(network, n_steps=N_STEPS, min_beta=0.0001, max_beta=0.02, device=device)
    ddpm.load_state_dict(torch.load(weight_path, map_location="cpu", weights_only=False))
    ddpm = ddpm.to(device)
    ddpm.eval()
    return ddpm

# ── Load VCNN ──
def load_vcnn(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get('model_config', {})
    model = VoronoiCNN(
        in_channels=cfg.get('in_channels', 5),
        out_channels=cfg.get('out_channels', 2),
        base_ch=cfg.get('base_ch', 32),
        depth=cfg.get('depth', 3),
    )
    model.load_state_dict(ckpt['model_state'])
    model = model.to(device)
    model.eval()
    return model

# ── Load raw data for VCNN ──
with open("data.pickle", "rb") as f:
    train_np, val_np, _test_np = pickle.load(f)

def _to_tensor(arr):
    t = torch.from_numpy(np.ascontiguousarray(arr)).float()
    t = t.permute(3, 2, 1, 0)
    t = torch.nan_to_num(t, nan=0.0)
    return t

val_raw = _to_tensor(val_np)

# Build ocean mask from first sample
gt0 = val_raw[0]  # (2, 44, 94)
speed0 = (gt0[0]**2 + gt0[1]**2).sqrt()
ocean_mask_np = (speed0 > 1e-8).numpy().astype(np.float32)

# Build observation mask (row 22)
obs_mask_np = np.zeros((OCEAN_H, OCEAN_W), dtype=np.float32)
obs_mask_np[22, :] = 1.0
obs_mask_np *= ocean_mask_np


def prepare_sample(vi):
    """Prepare a sample, returning both GP and VCNN reconstructions."""
    # -- Standard DDPM pipeline data --
    input_image_dd = val_data[vi][0].unsqueeze(0)
    input_orig = dd_std.unstandardize(input_image_dd.squeeze(0)).unsqueeze(0).to(device)
    input_std = standardizer(input_orig.squeeze(0)).unsqueeze(0).to(device)

    land_mask = (input_orig.abs() > 1e-5).float().to(device)
    area_h, area_w = 44, 94
    raw_mask = torch.ones(1, 1, 64, 128, device=device)
    raw_mask[0, 0, area_h // 2, :area_w] = 0.0
    border = BorderMaskGenerator().generate_mask(input_std.shape).to(device)
    raw_mask = raw_mask * border
    missing_mask_1ch = raw_mask * land_mask[:, 0:1]
    missing_mask = missing_mask_1ch.expand(-1, 2, -1, -1)

    # GP fill
    gp_out, _ = gp_fill(
        input_orig, missing_mask,
        lengthscale=dd.get_attribute("gp_lengthscale"),
        variance=dd.get_attribute("gp_variance"),
        noise=dd.get_attribute("gp_noise"),
        use_double=True,
        kernel_type=dd.get_attribute("gp_kernel_type"),
        coord_system=dd.get_attribute("gp_coord_system"),
        return_variance=True,
    )
    gp_mse = ((gp_out - input_orig) * missing_mask).pow(2).sum() / (missing_mask.sum() + 1e-8)
    gp_std_field = standardizer(gp_out.squeeze(0)).unsqueeze(0).to(device)

    return {
        'input_orig': input_orig,
        'input_std': input_std,
        'missing_mask': missing_mask,
        'missing_mask_1ch': missing_mask_1ch,
        'gp_mse': gp_mse,
        'gp_std_field': gp_std_field,
    }


def get_vcnn_prediction(vcnn_model, vi):
    """Run VCNN on a validation sample, return in standardized space (matching DDPM)."""
    vel_phys = val_raw[vi].numpy()  # (2, 44, 94), physical units

    # Normalize for VCNN input
    vel_n = (vel_phys - norm_mean[:, None, None]) / norm_std[:, None, None]
    vel_n *= ocean_mask_np[None, :, :]

    # Build Voronoi input
    voronoi_in = build_voronoi_input(vel_n, obs_mask_np, ocean_mask_np)
    voronoi_t = torch.from_numpy(voronoi_in).unsqueeze(0).to(device)

    # Forward pass → normalized output
    with torch.no_grad():
        pred_n = vcnn_model(voronoi_t)  # (1, 2, 44, 94) in per-component z-score space

    # Un-normalize to physical units
    mean_t = torch.tensor(norm_mean, dtype=torch.float32).view(1, 2, 1, 1).to(device)
    std_t = torch.tensor(norm_std, dtype=torch.float32).view(1, 2, 1, 1).to(device)
    pred_phys = pred_n * std_t + mean_t

    # Zero land
    ocean_t = torch.from_numpy(ocean_mask_np).to(device).unsqueeze(0).unsqueeze(0)
    pred_phys = pred_phys * ocean_t  # (1, 2, 44, 94)

    # Convert to DDPM standardized space for conditioning
    # Need to embed in 64x128 grid first
    pred_full = torch.zeros(1, 2, 64, 128, device=device)
    pred_full[:, :, :44, :94] = pred_phys

    vcnn_std = standardizer(pred_full.squeeze(0)).unsqueeze(0)  # (1, 2, 64, 128)

    # Also return physical-space prediction for MSE computation
    return vcnn_std, pred_full


def compute_mse(pred_std, gt_orig, mask):
    """Compute MSE between prediction (standardized) and ground truth (physical)."""
    pred_phys = standardizer.unstandardize(pred_std.squeeze(0)).unsqueeze(0).to(device)
    return ((pred_phys - gt_orig) * mask).pow(2).sum() / (mask.sum() + 1e-8)


def compute_mse_phys(pred_phys, gt_orig, mask):
    """Compute MSE directly in physical space."""
    return ((pred_phys - gt_orig) * mask).pow(2).sum() / (mask.sum() + 1e-8)


# ── DDPM inference: condition on VCNN instead of GP ──
def ddpm_refine_vcnn(ddpm, vcnn_std_field, missing_mask, missing_mask_1ch, t_val, seed=42):
    """Single-step DDPM refinement: noise VCNN output → denoise.
    No mask_xt (best from ablation study)."""
    torch.manual_seed(seed)
    alpha_bar = ddpm.alpha_bars[t_val].to(device)
    noise = torch.randn_like(vcnn_std_field)
    noisy = alpha_bar.sqrt() * vcnn_std_field + (1 - alpha_bar).sqrt() * noise
    # Condition on VCNN field (replacing GP field in conditioning channels)
    x_cond = torch.cat([noisy, missing_mask_1ch, vcnn_std_field], dim=1)
    time_tensor = torch.full((1, 1), t_val, device=device, dtype=torch.long)
    with torch.no_grad():
        return ddpm.network(x_cond, time_tensor)


def ddpm_refine_gp(ddpm, gp_std_field, missing_mask, missing_mask_1ch, t_val, seed=42):
    """Single-step DDPM refinement with GP conditioning (reference)."""
    torch.manual_seed(seed)
    alpha_bar = ddpm.alpha_bars[t_val].to(device)
    noise = torch.randn_like(gp_std_field)
    noisy = alpha_bar.sqrt() * gp_std_field + (1 - alpha_bar).sqrt() * noise
    x_cond = torch.cat([noisy, missing_mask_1ch, gp_std_field], dim=1)
    time_tensor = torch.full((1, 1), t_val, device=device, dtype=torch.long)
    with torch.no_grad():
        return ddpm.network(x_cond, time_tensor)


def ensemble_refine(refine_fn, ddpm, cond_field, missing_mask, missing_mask_1ch, t_val, n_ens):
    """Ensemble of DDPM refinements. Returns (mean, std)."""
    preds = []
    for k in range(n_ens):
        pred = refine_fn(ddpm, cond_field, missing_mask, missing_mask_1ch, t_val, seed=42 + k * 1000)
        preds.append(pred)
    stack = torch.stack(preds, dim=0)  # (n_ens, 1, 2, 64, 128)
    return stack.mean(dim=0), stack.std(dim=0)


# ====================================================================
# Main experiment
# ====================================================================
print("\n" + "=" * 90)
print("HYBRID EXPERIMENT: VCNN + DDPM REFINEMENT")
print("=" * 90)

ddpm = load_ddpm(os.path.join(RESULTS_DIR, "inpaint_gaussian_t250_best_ema_weights.pt"))
vcnn = load_vcnn("results/voronoi_cnn/voronoi_cnn_best.pt")

# Pre-compute all samples
print("\nPreparing samples...")
samples = []
for i in range(n_test):
    vi = int(val_indices[i])
    data = prepare_sample(vi)
    vcnn_std, vcnn_phys = get_vcnn_prediction(vcnn, vi)
    data['vcnn_std'] = vcnn_std
    data['vcnn_phys'] = vcnn_phys
    data['vcnn_mse'] = compute_mse_phys(vcnn_phys, data['input_orig'], data['missing_mask']).item()
    samples.append((vi, data))

gp_mses = [s[1]['gp_mse'].item() for s in samples]
vcnn_mses = [s[1]['vcnn_mse'] for s in samples]
mean_gp = np.mean(gp_mses)
mean_vcnn = np.mean(vcnn_mses)

print(f"\nGP baseline:   {mean_gp:.6f}")
print(f"VCNN baseline: {mean_vcnn:.6f}  ({mean_vcnn/mean_gp:.3f}x GP)")

print(f"\n{'Strategy':<55} | {'Mean MSE':>10} | {'vsGP':>6} | {'vsVCNN':>7} | {'Wins/GP':>7} | {'Wins/V':>6}")
print("-" * 105)
print(f"{'GP baseline':<55} | {mean_gp:>10.6f} | {'1.000x':>6} | {'---':>7} | {'---':>7} | {'---':>6}")
print(f"{'VCNN baseline':<55} | {mean_vcnn:>10.6f} | {mean_vcnn/mean_gp:>5.3f}x | {'1.000x':>7} | {sum(1 for v, g in zip(vcnn_mses, gp_mses) if v < g):>3}/{n_test} | {'---':>6}")


def evaluate(name, strategy_fn):
    mses, wins_gp, wins_vcnn = [], 0, 0
    for i, (vi, d) in enumerate(samples):
        result = strategy_fn(i, d)
        mse = compute_mse(result, d['input_orig'], d['missing_mask']).item()
        mses.append(mse)
        if mse < gp_mses[i]:
            wins_gp += 1
        if mse < vcnn_mses[i]:
            wins_vcnn += 1
    m = np.mean(mses)
    print(f"{name:<55} | {m:>10.6f} | {m/mean_gp:>5.3f}x | {m/mean_vcnn:>6.3f}x | {wins_gp:>3}/{n_test} | {wins_vcnn:>2}/{n_test}")
    return mses


# ── Section 1: GP-conditioned DDPM (reference) ──
print("\n--- GP-conditioned DDPM (current best, reference) ---")
for t in [100, 150, 200]:
    evaluate(f"GP→DDPM no_maskxt @ t={t}",
        lambda i, d, _t=t: ddpm_refine_gp(ddpm, d['gp_std_field'], d['missing_mask'], d['missing_mask_1ch'], _t))

evaluate(f"GP→DDPM Ens10 no_maskxt @ t=200",
    lambda i, d: ensemble_refine(ddpm_refine_gp, ddpm, d['gp_std_field'], d['missing_mask'], d['missing_mask_1ch'], 200, 10)[0])

# ── Section 2: VCNN-conditioned DDPM (THE HYBRID) ──
print("\n--- VCNN-conditioned DDPM (hybrid, zero retraining) ---")

# Single-step at various noise levels
for t in [25, 50, 75, 100, 150, 200]:
    evaluate(f"VCNN→DDPM no_maskxt @ t={t}",
        lambda i, d, _t=t: ddpm_refine_vcnn(ddpm, d['vcnn_std'], d['missing_mask'], d['missing_mask_1ch'], _t))

# ── Section 3: VCNN-conditioned DDPM ensemble ──
print("\n--- VCNN→DDPM ensemble (hybrid + uncertainty) ---")

for t in [25, 50, 75, 100, 150, 200]:
    evaluate(f"VCNN→DDPM Ens10 @ t={t}",
        lambda i, d, _t=t: ensemble_refine(ddpm_refine_vcnn, ddpm, d['vcnn_std'], d['missing_mask'], d['missing_mask_1ch'], _t, 10)[0])

# ── Section 4: Uncertainty quantification ──
print("\n\n--- UNCERTAINTY QUANTIFICATION (ensemble std) ---")
print("  Showing mean pixel-wise std across missing region for each ensemble.")

for t in [50, 100, 200]:
    stds_gp, stds_vcnn = [], []
    for i, (vi, d) in enumerate(samples):
        _, std_gp = ensemble_refine(ddpm_refine_gp, ddpm, d['gp_std_field'], d['missing_mask'], d['missing_mask_1ch'], t, 10)
        _, std_vcnn = ensemble_refine(ddpm_refine_vcnn, ddpm, d['vcnn_std'], d['missing_mask'], d['missing_mask_1ch'], t, 10)
        # Mean std over missing region
        mask = d['missing_mask']
        stds_gp.append((std_gp * mask).sum() / (mask.sum() + 1e-8))
        stds_vcnn.append((std_vcnn * mask).sum() / (mask.sum() + 1e-8))
    print(f"  t={t:>3}: GP-cond std = {torch.stack(stds_gp).mean():.5f}  |  VCNN-cond std = {torch.stack(stds_vcnn).mean():.5f}  |  ratio = {torch.stack(stds_vcnn).mean() / torch.stack(stds_gp).mean():.3f}x")

print("\nDone.")
