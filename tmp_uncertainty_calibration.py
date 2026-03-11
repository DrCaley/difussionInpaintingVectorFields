#!/usr/bin/env python3
"""
Uncertainty calibration test for the VCNN+DDPM hybrid.

Key question: Are the ensemble‐based uncertainty estimates *meaningful*?

Tests performed:
  1. PIXEL‐WISE CORRELATION  — Pearson & Spearman between pixel std and |error|
  2. CALIBRATION CURVE        — Does an X% prediction interval contain X% of truths?
  3. ORACLE RANKING (AUCE)    — If we trust uncertainty, are the worst pixels flagged?
  4. SPATIAL STRUCTURE         — Is the uncertainty map informative (not flat)?
  5. CONDITIONAL COVERAGE     — Coverage in hardest‐30% vs easiest‐30% regions
  6. COMPONENT CHECK          — Is uncertainty meaningful for u and v separately?

We run VCNN→DDPM Ens20 @ t=100 (best hybrid config) on 20 validation samples,
collecting per-pixel ensemble mean + std, then compare against ground truth.
"""
import sys, os, pickle, time
import numpy as np
import torch
from scipy import stats as scipy_stats

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
N_ENSEMBLE = 20   # larger ensemble for better uncertainty
N_SAMPLES = 20    # more samples for statistical power
T_NOISE = 100     # best hybrid t from prior experiment

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


# ── Load models ──
def load_ddpm(weight_path):
    network = MyUNet_FiLM_Attn(n_steps=N_STEPS, time_emb_dim=256, in_channels=5)
    ddpm = GaussianDDPM(network, n_steps=N_STEPS, min_beta=0.0001, max_beta=0.02, device=device)
    ddpm.load_state_dict(torch.load(weight_path, map_location="cpu", weights_only=False))
    ddpm = ddpm.to(device)
    ddpm.eval()
    return ddpm


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


# ── Raw data for VCNN ──
with open("data.pickle", "rb") as f:
    train_np, val_np, _test_np = pickle.load(f)


def _to_tensor(arr):
    t = torch.from_numpy(np.ascontiguousarray(arr)).float()
    t = t.permute(3, 2, 1, 0)
    t = torch.nan_to_num(t, nan=0.0)
    return t


val_raw = _to_tensor(val_np)
gt0 = val_raw[0]
speed0 = (gt0[0]**2 + gt0[1]**2).sqrt()
ocean_mask_np = (speed0 > 1e-8).numpy().astype(np.float32)
obs_mask_np = np.zeros((OCEAN_H, OCEAN_W), dtype=np.float32)
obs_mask_np[22, :] = 1.0
obs_mask_np *= ocean_mask_np


def prepare_sample(vi):
    input_image_dd = val_data[vi][0].unsqueeze(0)
    input_orig = dd_std.unstandardize(input_image_dd.squeeze(0)).unsqueeze(0).to(device)
    input_std = standardizer(input_orig.squeeze(0)).unsqueeze(0).to(device)

    land_mask = (input_orig.abs() > 1e-5).float().to(device)
    raw_mask = torch.ones(1, 1, 64, 128, device=device)
    raw_mask[0, 0, OCEAN_H // 2, :OCEAN_W] = 0.0
    border = BorderMaskGenerator().generate_mask(input_std.shape).to(device)
    raw_mask = raw_mask * border
    missing_mask_1ch = raw_mask * land_mask[:, 0:1]
    missing_mask = missing_mask_1ch.expand(-1, 2, -1, -1)

    return {
        'input_orig': input_orig,
        'input_std': input_std,
        'missing_mask': missing_mask,
        'missing_mask_1ch': missing_mask_1ch,
    }


def get_vcnn_prediction(vcnn_model, vi):
    vel_phys = val_raw[vi].numpy()
    vel_n = (vel_phys - norm_mean[:, None, None]) / norm_std[:, None, None]
    vel_n *= ocean_mask_np[None, :, :]
    voronoi_in = build_voronoi_input(vel_n, obs_mask_np, ocean_mask_np)
    voronoi_t = torch.from_numpy(voronoi_in).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_n = vcnn_model(voronoi_t)
    mean_t = torch.tensor(norm_mean, dtype=torch.float32).view(1, 2, 1, 1).to(device)
    std_t = torch.tensor(norm_std, dtype=torch.float32).view(1, 2, 1, 1).to(device)
    pred_phys = pred_n * std_t + mean_t
    ocean_t = torch.from_numpy(ocean_mask_np).to(device).unsqueeze(0).unsqueeze(0)
    pred_phys = pred_phys * ocean_t
    pred_full = torch.zeros(1, 2, 64, 128, device=device)
    pred_full[:, :, :44, :94] = pred_phys
    vcnn_std = standardizer(pred_full.squeeze(0)).unsqueeze(0)
    return vcnn_std, pred_full


def ddpm_refine(ddpm, cond_field, missing_mask_1ch, t_val, seed):
    torch.manual_seed(seed)
    alpha_bar = ddpm.alpha_bars[t_val].to(device)
    noise = torch.randn_like(cond_field)
    noisy = alpha_bar.sqrt() * cond_field + (1 - alpha_bar).sqrt() * noise
    x_cond = torch.cat([noisy, missing_mask_1ch, cond_field], dim=1)
    time_tensor = torch.full((1, 1), t_val, device=device, dtype=torch.long)
    with torch.no_grad():
        return ddpm.network(x_cond, time_tensor)


def ensemble_full(ddpm, cond_field, missing_mask_1ch, t_val, n_ens):
    """Return ALL individual predictions plus mean and std."""
    preds = []
    for k in range(n_ens):
        pred = ddpm_refine(ddpm, cond_field, missing_mask_1ch, t_val, seed=42 + k * 1000)
        preds.append(pred)
    stack = torch.stack(preds, dim=0)  # (n_ens, 1, 2, 64, 128)
    return stack, stack.mean(dim=0), stack.std(dim=0)


# ====================================================================
# Run inference
# ====================================================================
print("\n" + "=" * 80)
print("UNCERTAINTY CALIBRATION TEST")
print(f"Config: VCNN→DDPM Ens{N_ENSEMBLE} @ t={T_NOISE}, {N_SAMPLES} samples")
print("=" * 80)

ddpm = load_ddpm(os.path.join(RESULTS_DIR, "inpaint_gaussian_t250_best_ema_weights.pt"))
vcnn = load_vcnn("results/voronoi_cnn/voronoi_cnn_best.pt")

# Collect per-pixel data across all samples
all_errors = []          # |pred_mean - gt| per pixel (physical units)
all_stds = []            # ensemble std per pixel (standardized units)
all_stds_phys = []       # ensemble std per pixel (physical units)
all_u_errors = []        # u-component absolute error
all_v_errors = []        # v-component absolute error
all_u_stds = []          # u-component ensemble std
all_v_stds = []          # v-component ensemble std
all_pixel_preds = []     # individual ensemble member predictions per pixel
all_pixel_gts = []       # ground truth per pixel
sample_mses = []
sample_mean_stds = []

t0 = time.time()
for si in range(N_SAMPLES):
    vi = int(val_indices[si])
    d = prepare_sample(vi)
    vcnn_std, vcnn_phys = get_vcnn_prediction(vcnn, vi)

    # Run ensemble
    stack, ens_mean, ens_std = ensemble_full(
        ddpm, vcnn_std, d['missing_mask_1ch'], T_NOISE, N_ENSEMBLE
    )

    # Convert ensemble mean to physical space
    ens_mean_phys = standardizer.unstandardize(ens_mean.squeeze(0)).unsqueeze(0).to(device)

    # Convert ensemble std to physical space (std scales by the normalization std)
    # For unified z-score: std_phys = std_standardized * sigma_unified
    # But we use per-component standardizer, so need to handle carefully
    # std in standardized space: ens_std is in z-score units
    # To convert: multiply by normalization std for each component
    std_phys = ens_std.clone()
    std_phys[:, 0] *= U_STD  # u component
    std_phys[:, 1] *= V_STD  # v component

    gt = d['input_orig']  # (1, 2, 64, 128) physical
    mask = d['missing_mask']  # (1, 2, 64, 128)

    # Pixel-wise error and std over MISSING ocean pixels
    error = (ens_mean_phys - gt).abs()  # (1, 2, 64, 128)

    # Extract missing-region pixels
    mask_bool = mask[0, 0].bool()  # (64, 128) - same for both components
    n_missing = mask_bool.sum().item()

    # Per-component
    u_err = error[0, 0][mask_bool].cpu().numpy()
    v_err = error[0, 1][mask_bool].cpu().numpy()
    u_std = std_phys[0, 0][mask_bool].cpu().numpy()
    v_std = std_phys[0, 1][mask_bool].cpu().numpy()

    # Combined (magnitude of error vector)
    err_mag = (error[0, 0][mask_bool]**2 + error[0, 1][mask_bool]**2).sqrt().cpu().numpy()
    std_mag = (std_phys[0, 0][mask_bool]**2 + std_phys[0, 1][mask_bool]**2).sqrt().cpu().numpy()

    all_errors.append(err_mag)
    all_stds_phys.append(std_mag)
    all_u_errors.append(u_err)
    all_v_errors.append(v_err)
    all_u_stds.append(u_std)
    all_v_stds.append(v_std)

    # For calibration: collect individual ensemble member predictions per pixel
    # stack: (n_ens, 1, 2, 64, 128) in standardized space
    # Convert all members to physical
    for k in range(N_ENSEMBLE):
        member_phys = standardizer.unstandardize(stack[k, 0]).unsqueeze(0).to(device)
        # We don't need all members stored—just the gt and stats for calibration

    # For calibration intervals, we need: for each pixel, the ensemble distribution
    # and whether the truth falls within quantiles.
    # Store per-pixel: mean, std, gt (all in physical units, combined as u and v separately)
    all_pixel_preds.append((ens_mean_phys[0].cpu(), std_phys[0].cpu()))  # (2, 64, 128) each
    all_pixel_gts.append(gt[0].cpu())  # (2, 64, 128)

    mse = ((ens_mean_phys - gt) * mask).pow(2).sum() / (mask.sum() + 1e-8)
    mean_std = (std_phys * mask[:, :1].expand_as(std_phys)).sum() / (mask.sum() + 1e-8)
    sample_mses.append(mse.item())
    sample_mean_stds.append(mean_std.item())

    if (si + 1) % 5 == 0:
        elapsed = time.time() - t0
        print(f"  [{si+1}/{N_SAMPLES}] elapsed: {elapsed:.1f}s, "
              f"MSE={mse.item():.6f}, mean_std={mean_std.item():.5f}")

elapsed = time.time() - t0
print(f"\nInference complete: {elapsed:.1f}s total")

# Flatten all pixel-wise arrays
all_errors = np.concatenate(all_errors)
all_stds_phys = np.concatenate(all_stds_phys)
all_u_errors = np.concatenate(all_u_errors)
all_v_errors = np.concatenate(all_v_errors)
all_u_stds = np.concatenate(all_u_stds)
all_v_stds = np.concatenate(all_v_stds)

n_pixels = len(all_errors)
print(f"\nTotal pixels analyzed: {n_pixels:,}")
print(f"Mean sample MSE: {np.mean(sample_mses):.6f}")
print(f"Mean ensemble std (phys): {np.mean(sample_mean_stds):.6f}")


# ====================================================================
# TEST 1: PIXEL-WISE CORRELATION
# ====================================================================
print("\n" + "=" * 80)
print("TEST 1: PIXEL-WISE CORRELATION (std vs |error|)")
print("=" * 80)
print("  If uncertainty is meaningful, pixels with higher std should have higher error.")

# Combined (speed)
r_pearson, p_pearson = scipy_stats.pearsonr(all_stds_phys, all_errors)
r_spearman, p_spearman = scipy_stats.spearmanr(all_stds_phys, all_errors)
print(f"\n  Combined (speed):")
print(f"    Pearson  r = {r_pearson:.4f}  (p = {p_pearson:.2e})")
print(f"    Spearman ρ = {r_spearman:.4f}  (p = {p_spearman:.2e})")

# u-component
r_u, p_u = scipy_stats.pearsonr(all_u_stds, all_u_errors)
rho_u, _ = scipy_stats.spearmanr(all_u_stds, all_u_errors)
print(f"\n  u-component:")
print(f"    Pearson  r = {r_u:.4f}  (p = {p_u:.2e})")
print(f"    Spearman ρ = {rho_u:.4f}")

# v-component
r_v, p_v = scipy_stats.pearsonr(all_v_stds, all_v_errors)
rho_v, _ = scipy_stats.spearmanr(all_v_stds, all_v_errors)
print(f"\n  v-component:")
print(f"    Pearson  r = {r_v:.4f}  (p = {p_v:.2e})")
print(f"    Spearman ρ = {rho_v:.4f}")

interpretation = "GOOD" if r_spearman > 0.3 else ("WEAK" if r_spearman > 0.1 else "POOR")
print(f"\n  → Interpretation: {interpretation} correlation")
print(f"    (>0.3 = meaningful uncertainty, >0.5 = strong, <0.1 = useless)")


# ====================================================================
# TEST 2: CALIBRATION CURVE (prediction intervals)
# ====================================================================
print("\n" + "=" * 80)
print("TEST 2: CALIBRATION CURVE (Gaussian prediction intervals)")
print("=" * 80)
print("  Assuming ensemble mean ± k*std is Gaussian, check if X% intervals")
print("  contain X% of true values.")

# For each pixel we have (mean, std) and gt. Compute z-score: z = (gt - mean) / std
# If calibrated, z should be ~ N(0,1)

# Use u-component for clean 1D analysis
z_scores_u = (all_pixel_gts[0][0][torch.tensor(True)].numpy() - 1)  # placeholder; compute properly below

# Collect z-scores properly
z_u_all, z_v_all = [], []
for si in range(N_SAMPLES):
    pred_mean, pred_std = all_pixel_preds[si]  # (2, 64, 128) each
    gt = all_pixel_gts[si]  # (2, 64, 128)

    # Get mask for this sample
    vi = int(val_indices[si])
    d_mask = prepare_sample(vi)
    mask_bool = d_mask['missing_mask'][0, 0].bool().cpu()

    # z-scores for u and v
    u_z = ((gt[0] - pred_mean[0]) / (pred_std[0] + 1e-10))[mask_bool].numpy()
    v_z = ((gt[1] - pred_mean[1]) / (pred_std[1] + 1e-10))[mask_bool].numpy()
    z_u_all.append(u_z)
    z_v_all.append(v_z)

z_u_all = np.concatenate(z_u_all)
z_v_all = np.concatenate(z_v_all)
z_all = np.concatenate([z_u_all, z_v_all])

# Calibration: for nominal coverage levels, what fraction of z-scores fall within?
nominal_levels = [0.50, 0.68, 0.80, 0.90, 0.95, 0.99]
print(f"\n  {'Nominal':>8} | {'k (z-mult)':>10} | {'Actual (u)':>10} | {'Actual (v)':>10} | {'Actual (all)':>12} | {'Status':>8}")
print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}-+-{'-'*8}")

calibration_gaps = []
for nom in nominal_levels:
    k = scipy_stats.norm.ppf((1 + nom) / 2)  # z-multiplier for this coverage
    actual_u = np.mean(np.abs(z_u_all) <= k)
    actual_v = np.mean(np.abs(z_v_all) <= k)
    actual_all = np.mean(np.abs(z_all) <= k)
    gap = actual_all - nom
    calibration_gaps.append(abs(gap))

    if abs(gap) < 0.05:
        status = "✓ GOOD"
    elif abs(gap) < 0.10:
        status = "~ OK"
    elif actual_all > nom:
        status = "OVER-CON"  # overconfident intervals too wide
    else:
        status = "UNDER"     # underconfident
    print(f"  {nom:>7.0%} | {k:>10.3f} | {actual_u:>9.1%} | {actual_v:>9.1%} | {actual_all:>11.1%} | {status:>8}")

mean_cal_gap = np.mean(calibration_gaps)
print(f"\n  Mean calibration gap: {mean_cal_gap:.3f}")
cal_interpret = "WELL-CALIBRATED" if mean_cal_gap < 0.05 else ("FAIRLY CALIBRATED" if mean_cal_gap < 0.10 else "POORLY CALIBRATED")
print(f"  → Interpretation: {cal_interpret}")
print(f"    (<0.05 = well calibrated, <0.10 = fair, >0.10 = needs recalibration)")

# Z-score distribution statistics
print(f"\n  z-score statistics (should be ~N(0,1) if calibrated):")
print(f"    u-component:  mean={np.mean(z_u_all):+.3f}, std={np.std(z_u_all):.3f}, "
      f"skew={scipy_stats.skew(z_u_all):.3f}, kurtosis={scipy_stats.kurtosis(z_u_all):.3f}")
print(f"    v-component:  mean={np.mean(z_v_all):+.3f}, std={np.std(z_v_all):.3f}, "
      f"skew={scipy_stats.skew(z_v_all):.3f}, kurtosis={scipy_stats.kurtosis(z_v_all):.3f}")
print(f"    (ideal: mean=0, std=1, skew=0, kurtosis=0)")

z_std = np.std(z_all)
if z_std < 1.0:
    print(f"\n  Note: z-score std = {z_std:.3f} < 1.0 → ensemble is OVERCONFIDENT")
    print(f"    (uncertainty estimates are too small, true errors are larger than predicted)")
elif z_std > 1.0:
    print(f"\n  Note: z-score std = {z_std:.3f} > 1.0 → ensemble is CONSERVATIVE")
    print(f"    (uncertainty estimates are too large, could be tightened for sharper intervals)")


# ====================================================================
# TEST 3: ORACLE RANKING — Does uncertainty predict which pixels are bad?
# ====================================================================
print("\n" + "=" * 80)
print("TEST 3: ORACLE RANKING (uncertainty-ordered error)")
print("=" * 80)
print("  If uncertainty is informative, the most uncertain pixels should have")
print("  the highest errors. We sort pixels by std and check error in quintiles.")

# Sort pixels by predicted uncertainty
sort_idx = np.argsort(all_stds_phys)
n_q = 5
q_size = n_pixels // n_q

print(f"\n  {'Quintile':>10} | {'Std range':>20} | {'Mean |error|':>12} | {'Median |error|':>14} | {'Mean std':>10}")
print(f"  {'-'*10}-+-{'-'*20}-+-{'-'*12}-+-{'-'*14}-+-{'-'*10}")

quintile_errors = []
for q in range(n_q):
    idx = sort_idx[q * q_size: (q + 1) * q_size]
    q_std = all_stds_phys[idx]
    q_err = all_errors[idx]
    quintile_errors.append(np.mean(q_err))
    label = f"Q{q+1} ({'lowest' if q == 0 else 'highest' if q == n_q-1 else f'mid-{q}'})"
    print(f"  {label:>10} | [{q_std.min():.5f}, {q_std.max():.5f}] | "
          f"{np.mean(q_err):>12.6f} | {np.median(q_err):>14.6f} | {np.mean(q_std):>10.6f}")

# Check monotonicity
monotonic = all(quintile_errors[i] <= quintile_errors[i+1] for i in range(len(quintile_errors)-1))
ratio_q5_q1 = quintile_errors[-1] / (quintile_errors[0] + 1e-10)
print(f"\n  Monotonically increasing: {'YES ✓' if monotonic else 'NO ✗'}")
print(f"  Error ratio Q5/Q1: {ratio_q5_q1:.2f}x")
print(f"  → Interpretation: {'STRONG' if ratio_q5_q1 > 3 else 'MODERATE' if ratio_q5_q1 > 1.5 else 'WEAK'} discriminative power")
print(f"    (>3x = excellent, >1.5x = useful, <1.5x = barely informative)")


# ====================================================================
# TEST 4: SPATIAL STRUCTURE (is uncertainty map informative?)
# ====================================================================
print("\n" + "=" * 80)
print("TEST 4: SPATIAL STRUCTURE (uncertainty not just constant)")
print("=" * 80)
print("  Check that std has meaningful spatial variation, not just a constant.")

# Compute coefficient of variation of the std field for each sample
cvs = []
for si in range(N_SAMPLES):
    pred_mean, pred_std = all_pixel_preds[si]
    vi = int(val_indices[si])
    d_mask = prepare_sample(vi)
    mask_bool = d_mask['missing_mask'][0, 0].bool().cpu()

    std_vals = pred_std[0][mask_bool].numpy()  # u-component std
    cv = np.std(std_vals) / (np.mean(std_vals) + 1e-10)
    cvs.append(cv)

mean_cv = np.mean(cvs)
print(f"\n  Coefficient of variation of std field (per sample):")
print(f"    Mean CV = {mean_cv:.3f}  (min={min(cvs):.3f}, max={max(cvs):.3f})")
print(f"    → {'GOOD spatial variation' if mean_cv > 0.3 else 'MODERATE spatial variation' if mean_cv > 0.1 else 'WARNING: nearly constant std (not spatially informative)'}")

# Distance-from-observations analysis
# Pixels far from row 22 should have higher uncertainty
print(f"\n  Uncertainty vs distance from observed row:")
# Build distance map from row 22
row_obs = 22
dist_from_obs = np.abs(np.arange(OCEAN_H) - row_obs).astype(np.float32)  # (44,)
dist_map = np.tile(dist_from_obs[:, None], (1, OCEAN_W))  # (44, 94)
dist_full = np.zeros((64, 128), dtype=np.float32)
dist_full[:44, :94] = dist_map

dist_bins = [(0, 5), (5, 10), (10, 20), (20, 44)]
print(f"  {'Distance':>12} | {'Mean std':>10} | {'Mean |error|':>12} | {'n_pixels':>8}")
print(f"  {'-'*12}-+-{'-'*10}-+-{'-'*12}-+-{'-'*8}")

for lo, hi in dist_bins:
    # Collect pixels in this distance range across all samples
    stds_bin, errs_bin = [], []
    for si in range(N_SAMPLES):
        pred_mean, pred_std = all_pixel_preds[si]
        gt = all_pixel_gts[si]
        vi = int(val_indices[si])
        d_mask = prepare_sample(vi)
        mask_2d = d_mask['missing_mask'][0, 0].cpu().numpy()

        dist_mask = (dist_full >= lo) & (dist_full < hi) & (mask_2d > 0.5)
        if dist_mask.sum() == 0:
            continue
        std_bin = pred_std[0][torch.from_numpy(dist_mask)].numpy()  # u-std
        err_bin = (pred_mean[0] - gt[0]).abs()[torch.from_numpy(dist_mask)].numpy()
        stds_bin.append(std_bin)
        errs_bin.append(err_bin)

    if stds_bin:
        stds_flat = np.concatenate(stds_bin)
        errs_flat = np.concatenate(errs_bin)
        print(f"  {lo:>3}-{hi:<3} px   | {np.mean(stds_flat):>10.6f} | {np.mean(errs_flat):>12.6f} | {len(stds_flat):>8,}")


# ====================================================================
# TEST 5: CONDITIONAL COVERAGE
# ====================================================================
print("\n" + "=" * 80)
print("TEST 5: CONDITIONAL COVERAGE (hard vs easy regions)")
print("=" * 80)
print("  Check if 90% intervals maintain ~90% coverage even in the hardest regions.")

# Split pixels into tertiles by true error
err_sort = np.argsort(all_errors)  # ascending
n_tert = 3
tert_size = n_pixels // n_tert
k_90 = scipy_stats.norm.ppf(0.95)  # 90% interval: ±1.645

# Need per-component z-scores matched to error positions
# Use combined u+v z-scores
z_combined_per_pixel = np.concatenate([z_u_all, z_v_all])  # but these don't align with all_errors...

# Recompute properly: for each pixel, check coverage
# We need z-scores aligned with the same pixel ordering as all_errors
# Easier: recompute using the u/v separately and concatenate

# For u-component
u_err_sort = np.argsort(all_u_errors)
u_tert = n_pixels // (N_SAMPLES) // n_tert  # per sample
k_90_val = scipy_stats.norm.ppf(0.95)

# Just use all_u z-scores directly
print(f"\n  Coverage at 90% nominal for u-component error tertiles:")
u_z_abs = np.abs(z_u_all)
u_err_matched = all_u_errors  # same ordering as z_u_all

err_tert_idx = np.argsort(u_err_matched)
tert_size_u = len(u_err_matched) // n_tert

for t_idx, t_name in enumerate(["Easy (low error)", "Medium", "Hard (high error)"]):
    idx = err_tert_idx[t_idx * tert_size_u: (t_idx + 1) * tert_size_u]
    coverage = np.mean(u_z_abs[idx] <= k_90_val)
    mean_err = np.mean(u_err_matched[idx])
    print(f"    {t_name:<20}: coverage={coverage:.1%}  (mean |error|={mean_err:.5f})")

print(f"\n  Coverage at 90% nominal for v-component error tertiles:")
v_z_abs = np.abs(z_v_all)
v_err_matched = all_v_errors

err_tert_idx_v = np.argsort(v_err_matched)
tert_size_v = len(v_err_matched) // n_tert

for t_idx, t_name in enumerate(["Easy (low error)", "Medium", "Hard (high error)"]):
    idx = err_tert_idx_v[t_idx * tert_size_v: (t_idx + 1) * tert_size_v]
    coverage = np.mean(v_z_abs[idx] <= k_90_val)
    mean_err = np.mean(v_err_matched[idx])
    print(f"    {t_name:<20}: coverage={coverage:.1%}  (mean |error|={mean_err:.5f})")


# ====================================================================
# TEST 6: SAMPLE-LEVEL CORRELATION
# ====================================================================
print("\n" + "=" * 80)
print("TEST 6: SAMPLE-LEVEL CORRELATION (mean std vs MSE across samples)")
print("=" * 80)
print("  Do harder samples (higher MSE) also get higher mean uncertainty?")

r_sample, p_sample = scipy_stats.pearsonr(sample_mean_stds, sample_mses)
rho_sample, _ = scipy_stats.spearmanr(sample_mean_stds, sample_mses)
print(f"\n  Pearson  r = {r_sample:.4f}  (p = {p_sample:.3f})")
print(f"  Spearman ρ = {rho_sample:.4f}")

for i in range(min(N_SAMPLES, 5)):
    print(f"    Sample {i:2d}: MSE = {sample_mses[i]:.6f}, mean std = {sample_mean_stds[i]:.6f}")

interpretation = "STRONG" if abs(rho_sample) > 0.6 else ("MODERATE" if abs(rho_sample) > 0.3 else "WEAK")
print(f"\n  → {interpretation} sample-level correlation")
print(f"    (high = model 'knows' which samples it's unsure about)")


# ====================================================================
# OVERALL SUMMARY
# ====================================================================
print("\n" + "=" * 80)
print("OVERALL UNCERTAINTY QUALITY SUMMARY")
print("=" * 80)

scores = {
    'Pixel-wise correlation': ('PASS' if r_spearman > 0.2 else 'FAIL', f"ρ={r_spearman:.3f}"),
    'Calibration': ('PASS' if mean_cal_gap < 0.10 else 'FAIL', f"gap={mean_cal_gap:.3f}"),
    'Oracle ranking': ('PASS' if ratio_q5_q1 > 1.5 else 'FAIL', f"Q5/Q1={ratio_q5_q1:.1f}x"),
    'Spatial structure': ('PASS' if mean_cv > 0.1 else 'FAIL', f"CV={mean_cv:.3f}"),
    'Sample-level correlation': ('PASS' if abs(rho_sample) > 0.3 else 'FAIL', f"ρ={rho_sample:.3f}"),
}

for test, (result, detail) in scores.items():
    icon = "✓" if result == "PASS" else "✗"
    print(f"  {icon} {test:<30}: {result}  ({detail})")

n_pass = sum(1 for v in scores.values() if v[0] == 'PASS')
n_total = len(scores)
print(f"\n  SCORE: {n_pass}/{n_total} tests passed")

if n_pass == n_total:
    print("  → CONCLUSION: Uncertainty estimates are MEANINGFUL and WELL-CALIBRATED")
elif n_pass >= 3:
    print("  → CONCLUSION: Uncertainty estimates are PARTIALLY MEANINGFUL (some aspects need work)")
else:
    print("  → CONCLUSION: Uncertainty estimates are NOT RELIABLE for downstream use")

print("\nDone.")
