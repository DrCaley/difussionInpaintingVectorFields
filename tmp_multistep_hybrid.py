#!/usr/bin/env python3
"""
Multi-step DDPM reverse process starting from VCNN output.

Previous hybrid tests used single-step x0 prediction:
  VCNN → noise to t → one network pass → predicted x0

This test uses the proper iterative DDPM reverse process:
  VCNN → noise to t_start → walk back t_start → t_start-1 → ... → 0

Each step computes the DDPM posterior q(x_{t-1} | x_t, x̂₀) and samples,
which is what diffusion models are actually designed to do.

We also test with RePaint resampling (re-noise + re-denoise) for better
boundary coherence between known and unknown regions.

Comparisons:
  1. Single-step (current best): VCNN→noise→one pass→x0
  2. Multi-step (no resample): VCNN→noise→walk back→x0
  3. Multi-step + RePaint (resample_steps=3,5): adds re-noising
  4. Ensemble versions of the above for uncertainty
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


def compute_mse(pred_std, gt_orig, mask):
    pred_phys = standardizer.unstandardize(pred_std.squeeze(0)).unsqueeze(0).to(device)
    return ((pred_phys - gt_orig) * mask).pow(2).sum() / (mask.sum() + 1e-8)


def compute_mse_phys(pred_phys, gt_orig, mask):
    return ((pred_phys - gt_orig) * mask).pow(2).sum() / (mask.sum() + 1e-8)


# =========================================================================
# INFERENCE STRATEGIES
# =========================================================================

def single_step_x0(ddpm, cond_field, missing_mask_1ch, t_val, seed=42):
    """Original single-step: noise to t → one forward pass → x0 prediction."""
    torch.manual_seed(seed)
    alpha_bar = ddpm.alpha_bars[t_val].to(device)
    noise = torch.randn_like(cond_field)
    noisy = alpha_bar.sqrt() * cond_field + (1 - alpha_bar).sqrt() * noise
    x_cond = torch.cat([noisy, missing_mask_1ch, cond_field], dim=1)
    time_tensor = torch.full((1, 1), t_val, device=device, dtype=torch.long)
    with torch.no_grad():
        return ddpm.network(x_cond, time_tensor)


def multi_step_reverse(ddpm, cond_field, input_std, missing_mask, missing_mask_1ch,
                       t_start, resample_steps=1, mask_xt=False, seed=42):
    """Full multi-step DDPM reverse process from t_start down to 0.

    At each step:
      1. Build 5-ch input: [x_t, mask, conditioning_field]
      2. Network predicts x̂₀
      3. Compute DDPM posterior q(x_{t-1} | x_t, x̂₀)
      4. Sample x_{t-1}
      5. RePaint paste: replace known region with forward-noised GT
      6. (Optional) RePaint resample: re-noise back to t, repeat

    Args:
        ddpm: trained DDPM model
        cond_field: (1, 2, 64, 128) VCNN output in standardized space (used as conditioning)
        input_std: (1, 2, 64, 128) ground truth in standardized space (for paste-back)
        missing_mask: (1, 2, 64, 128) 1=missing/unknown
        missing_mask_1ch: (1, 1, 64, 128) single-channel mask
        t_start: starting timestep
        resample_steps: RePaint resample iterations (1 = no resample)
        mask_xt: replace known region with independent noise before network
        seed: random seed
    """
    torch.manual_seed(seed)
    known_mask = 1.0 - missing_mask  # 1 where known

    # Known values for conditioning (observed data in standardized space)
    known_values = input_std * known_mask  # (1, 2, 64, 128)

    # Build composite starting point: known=GT, unknown=VCNN
    composite = input_std * known_mask + cond_field * missing_mask

    # Forward diffuse to t_start
    alpha_bar_start = ddpm.alpha_bars[t_start].to(device)
    noise_init = torch.randn_like(composite)
    x = alpha_bar_start.sqrt() * composite + (1 - alpha_bar_start).sqrt() * noise_init

    with torch.no_grad():
        for t in range(t_start, -1, -1):
            n_resample = resample_steps if t > 0 else 1

            for r in range(n_resample):
                alpha_t = ddpm.alphas[t].to(device)
                alpha_bar_t = ddpm.alpha_bars[t].to(device)
                beta_t = ddpm.betas[t].to(device)

                time_tensor = torch.full((1, 1), t, device=device, dtype=torch.long)

                # Build 5-channel input
                if mask_xt:
                    indep_noise = torch.randn_like(x)
                    x_for_model = x * missing_mask[:, :2] + indep_noise * known_mask[:, :2]
                else:
                    x_for_model = x

                x_cond = torch.cat([x_for_model, missing_mask_1ch, cond_field], dim=1)

                # Network predicts x̂₀
                x0_pred = ddpm.network(x_cond, time_tensor)

                # DDPM posterior q(x_{t-1} | x_t, x̂₀)
                if t > 0:
                    alpha_bar_prev = ddpm.alpha_bars[t - 1].to(device)

                    coeff_x0 = (alpha_bar_prev.sqrt() * beta_t) / (1 - alpha_bar_t)
                    coeff_xt = (alpha_t.sqrt() * (1 - alpha_bar_prev)) / (1 - alpha_bar_t)
                    mu = coeff_x0 * x0_pred + coeff_xt * x

                    beta_tilde = ((1 - alpha_bar_prev) / (1 - alpha_bar_t)) * beta_t
                    sigma_t = beta_tilde.sqrt()

                    z = torch.randn_like(x)
                    x_denoised = mu + sigma_t * z
                else:
                    x_denoised = x0_pred

                # RePaint: paste forward-noised known region
                if t > 0:
                    alpha_bar_prev = ddpm.alpha_bars[t - 1].to(device)
                    noise_known = torch.randn_like(input_std)
                    x_known = alpha_bar_prev.sqrt() * input_std + (1 - alpha_bar_prev).sqrt() * noise_known
                    x = x_known * known_mask + x_denoised * missing_mask
                else:
                    x = input_std * known_mask + x_denoised * missing_mask

                # RePaint resample: re-noise back to t if not last iteration
                if r < n_resample - 1 and t > 0:
                    alpha_bar_prev = ddpm.alpha_bars[t - 1].to(device)
                    noise_back = torch.randn_like(x)
                    # Re-noise from t-1 back to t
                    x = (alpha_t.sqrt() * x + (1 - alpha_t).sqrt() * noise_back)

    return x  # This is in standardized space, paste-backed


def multi_step_reverse_cond_only(ddpm, cond_field, missing_mask, missing_mask_1ch,
                                  t_start, seed=42):
    """Multi-step but simpler: no paste-back, just denoise from VCNN starting point.

    Uses cond_field as both the initialization AND the conditioning channel.
    No known-value paste-back at each step.
    """
    torch.manual_seed(seed)

    # Forward diffuse VCNN output to t_start
    alpha_bar_start = ddpm.alpha_bars[t_start].to(device)
    noise_init = torch.randn_like(cond_field)
    x = alpha_bar_start.sqrt() * cond_field + (1 - alpha_bar_start).sqrt() * noise_init

    with torch.no_grad():
        for t in range(t_start, -1, -1):
            alpha_t = ddpm.alphas[t].to(device)
            alpha_bar_t = ddpm.alpha_bars[t].to(device)
            beta_t = ddpm.betas[t].to(device)

            time_tensor = torch.full((1, 1), t, device=device, dtype=torch.long)

            # 5-channel input: [x_t, mask, cond_field]
            x_cond = torch.cat([x, missing_mask_1ch, cond_field], dim=1)

            # Predict x̂₀
            x0_pred = ddpm.network(x_cond, time_tensor)

            # DDPM step
            if t > 0:
                alpha_bar_prev = ddpm.alpha_bars[t - 1].to(device)
                coeff_x0 = (alpha_bar_prev.sqrt() * beta_t) / (1 - alpha_bar_t)
                coeff_xt = (alpha_t.sqrt() * (1 - alpha_bar_prev)) / (1 - alpha_bar_t)
                mu = coeff_x0 * x0_pred + coeff_xt * x
                beta_tilde = ((1 - alpha_bar_prev) / (1 - alpha_bar_t)) * beta_t
                sigma_t = beta_tilde.sqrt()
                z = torch.randn_like(x)
                x = mu + sigma_t * z
            else:
                x = x0_pred

    return x


def ensemble(infer_fn, n_ens=10, **kwargs):
    """Run inference n_ens times with different seeds, return (mean, std, all)."""
    preds = []
    for k in range(n_ens):
        pred = infer_fn(seed=42 + k * 1000, **kwargs)
        preds.append(pred)
    stack = torch.stack(preds, dim=0)
    return stack.mean(dim=0), stack.std(dim=0), stack


# ====================================================================
# Main experiment
# ====================================================================
print("\n" + "=" * 90)
print("MULTI-STEP DDPM REVERSE WITH VCNN INITIALIZATION")
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

print(f"\n{'Strategy':<60} | {'Mean MSE':>10} | {'vsGP':>6} | {'vsVCNN':>7} | {'W/GP':>5} | {'W/V':>4} | {'Time':>5}")
print("-" * 115)
print(f"{'GP baseline':<60} | {mean_gp:>10.6f} | {'1.000x':>6} | {'---':>7} | {'---':>5} | {'---':>4} | {'---':>5}")
print(f"{'VCNN baseline':<60} | {mean_vcnn:>10.6f} | {mean_vcnn/mean_gp:>5.3f}x | {'1.000x':>7} | "
      f"{sum(1 for v, g in zip(vcnn_mses, gp_mses) if v < g):>2}/{n_test} | {'---':>4} | {'---':>5}")


def evaluate(name, strategy_fn):
    mses, wins_gp, wins_vcnn = [], 0, 0
    t0 = time.time()
    for i, (vi, d) in enumerate(samples):
        result = strategy_fn(i, d)
        mse = compute_mse(result, d['input_orig'], d['missing_mask']).item()
        mses.append(mse)
        if mse < gp_mses[i]:
            wins_gp += 1
        if mse < vcnn_mses[i]:
            wins_vcnn += 1
    elapsed = time.time() - t0
    m = np.mean(mses)
    print(f"{name:<60} | {m:>10.6f} | {m/mean_gp:>5.3f}x | {m/mean_vcnn:>6.3f}x | "
          f"{wins_gp:>2}/{n_test} | {wins_vcnn:>1}/{n_test} | {elapsed:>4.1f}s")
    return mses


# ── Section 1: Single-step baselines (reference) ──
print("\n--- Single-step x0 prediction (current best, for comparison) ---")
for t in [50, 75, 100]:
    evaluate(f"VCNN→single-step @ t={t}",
        lambda i, d, _t=t: single_step_x0(ddpm, d['vcnn_std'], d['missing_mask_1ch'], _t))

# ── Section 2: Multi-step reverse (no paste-back, conditioning only) ──
print("\n--- Multi-step reverse, cond-only (no paste-back) ---")
for t in [25, 50, 75, 100]:
    evaluate(f"VCNN→multi-step cond-only @ t={t}",
        lambda i, d, _t=t: multi_step_reverse_cond_only(
            ddpm, d['vcnn_std'], d['missing_mask'], d['missing_mask_1ch'], _t))

# ── Section 3: Multi-step reverse WITH paste-back ──
print("\n--- Multi-step reverse + paste-back (RePaint, resample=1) ---")
for t in [25, 50, 75, 100]:
    evaluate(f"VCNN→multi-step+paste @ t={t}, r=1",
        lambda i, d, _t=t: multi_step_reverse(
            ddpm, d['vcnn_std'], d['input_std'], d['missing_mask'], d['missing_mask_1ch'],
            _t, resample_steps=1, mask_xt=False))

# ── Section 4: Multi-step + paste + mask_xt ──
print("\n--- Multi-step + paste + mask_xt ---")
for t in [50, 75, 100]:
    evaluate(f"VCNN→multi-step+paste+mask_xt @ t={t}, r=1",
        lambda i, d, _t=t: multi_step_reverse(
            ddpm, d['vcnn_std'], d['input_std'], d['missing_mask'], d['missing_mask_1ch'],
            _t, resample_steps=1, mask_xt=True))

# ── Section 5: Multi-step + paste + RePaint resampling ──
print("\n--- Multi-step + paste + RePaint resample ---")
for t in [50, 75]:
    for r in [3, 5]:
        evaluate(f"VCNN→multi-step+paste @ t={t}, r={r}",
            lambda i, d, _t=t, _r=r: multi_step_reverse(
                ddpm, d['vcnn_std'], d['input_std'], d['missing_mask'], d['missing_mask_1ch'],
                _t, resample_steps=_r, mask_xt=False))

# ── Section 6: Best multi-step config as ensemble ──
print("\n--- Ensemble comparisons (Ens10) ---")
# Find best single-step config to compare against
evaluate(f"VCNN→single-step Ens10 @ t=100",
    lambda i, d: ensemble(
        lambda seed: single_step_x0(ddpm, d['vcnn_std'], d['missing_mask_1ch'], 100, seed),
        n_ens=10)[0])

# Test multi-step ensembles at the most promising t values
for t in [50, 75]:
    evaluate(f"VCNN→multi-step+paste Ens10 @ t={t}, r=1",
        lambda i, d, _t=t: ensemble(
            lambda seed, _t2=_t: multi_step_reverse(
                ddpm, d['vcnn_std'], d['input_std'], d['missing_mask'], d['missing_mask_1ch'],
                _t2, resample_steps=1, mask_xt=False, seed=seed),
            n_ens=10)[0])

    evaluate(f"VCNN→multi-step cond-only Ens10 @ t={t}",
        lambda i, d, _t=t: ensemble(
            lambda seed, _t2=_t: multi_step_reverse_cond_only(
                ddpm, d['vcnn_std'], d['missing_mask'], d['missing_mask_1ch'],
                _t2, seed=seed),
            n_ens=10)[0])

print("\nDone.")
