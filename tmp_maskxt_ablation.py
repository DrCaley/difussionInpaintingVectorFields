#!/usr/bin/env python3
"""Ablation: mask_xt vs no mask_xt, iterative noise buildup, and t=75.

Tests:
  A) Original: 1-shot noise to t, mask_xt=True  (current best)
  B) No mask_xt: 1-shot noise to t, leave noised GP in known region
  C) Iterative buildup: add noise step-by-step from t=0→t_target, mask_xt
  D) Iterative buildup + no mask_xt
  E) Low noise: t=75 variants
  F) Paste-back: paste known observations into output after prediction
"""
import sys, os, torch
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from ddpm.neural_networks.ddpm import GaussianDDPM
from ddpm.neural_networks.unets.unet_film_attn import MyUNet_FiLM_Attn
from ddpm.helper_functions.masks.border_mask import BorderMaskGenerator
from ddpm.helper_functions.standardize_data import ZScoreStandardizer
from ddpm.utils.noise_utils import get_noise_strategy
from ddpm.helper_functions.interpolation_tool import gp_fill
from data_prep.data_initializer import DDInitializer

N_STEPS = 250
U_MEAN, U_STD = -0.06929559429949586, 0.1358005549716049
V_MEAN, V_STD = -0.0323937796117541, 0.08899177232117582
standardizer = ZScoreStandardizer(U_MEAN, U_STD, V_MEAN, V_STD)

dd = DDInitializer()
device = dd.get_device()
val_data = dd.get_validation_data()
dd_std = dd.get_standardizer()

RESULTS_DIR = os.path.join(BASE, "experiments/06_gp_forward/gp_conditioned/results")
GPDIFF_PT = os.path.join(BASE, "results/eddy_balanced_eval/bulk_eval_eddy_balanced_100.pt")
gpdiff_data = torch.load(GPDIFF_PT, map_location="cpu", weights_only=False)
val_indices = gpdiff_data["val_indices"]

n_test = 10


def load_model(weight_path):
    network = MyUNet_FiLM_Attn(n_steps=N_STEPS, time_emb_dim=256, in_channels=5)
    ddpm = GaussianDDPM(network, n_steps=N_STEPS, min_beta=0.0001, max_beta=0.02, device=device)
    ddpm.load_state_dict(torch.load(weight_path, map_location="cpu", weights_only=False))
    ddpm = ddpm.to(device)
    ddpm.eval()
    return ddpm


def prepare_sample(vi):
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
    # Also get the known observations in standardized space for paste-back
    known_std = input_std  # ground truth standardized
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
    return input_orig, input_std, missing_mask, missing_mask_1ch, gp_mse, gp_std_field, known_std


def compute_mse(pred_std, gt_orig, mask):
    pred_phys = standardizer.unstandardize(pred_std.squeeze(0)).unsqueeze(0).to(device)
    return ((pred_phys - gt_orig) * mask).pow(2).sum() / (mask.sum() + 1e-8)


# ── Strategy A: Original with mask_xt (1-shot noise) ──
def single_step_maskxt(ddpm, start_field, missing_mask, missing_mask_1ch, gp_std_field, t_val, seed=42):
    """Original: 1-shot noise to t, replace known region with indep noise."""
    torch.manual_seed(seed)
    alpha_bar = ddpm.alpha_bars[t_val].to(device)
    noise = torch.randn_like(start_field)
    noisy = alpha_bar.sqrt() * start_field + (1 - alpha_bar).sqrt() * noise
    known_mask = 1.0 - missing_mask
    indep_noise = torch.randn_like(noisy)
    x_for_model = noisy * missing_mask[:, :2] + indep_noise * known_mask[:, :2]
    x_cond = torch.cat([x_for_model, missing_mask_1ch, gp_std_field], dim=1)
    time_tensor = torch.full((1, 1), t_val, device=device, dtype=torch.long)
    with torch.no_grad():
        return ddpm.network(x_cond, time_tensor)


# ── Strategy B: No mask_xt (1-shot noise, keep noised GP everywhere) ──
def single_step_no_maskxt(ddpm, start_field, missing_mask, missing_mask_1ch, gp_std_field, t_val, seed=42):
    """No mask_xt: 1-shot noise, leave noised GP in known region."""
    torch.manual_seed(seed)
    alpha_bar = ddpm.alpha_bars[t_val].to(device)
    noise = torch.randn_like(start_field)
    noisy = alpha_bar.sqrt() * start_field + (1 - alpha_bar).sqrt() * noise
    # NO replacement — model sees noised GP everywhere
    x_cond = torch.cat([noisy, missing_mask_1ch, gp_std_field], dim=1)
    time_tensor = torch.full((1, 1), t_val, device=device, dtype=torch.long)
    with torch.no_grad():
        return ddpm.network(x_cond, time_tensor)


# ── Strategy C: Iterative noise buildup + mask_xt ──
def iterative_buildup_maskxt(ddpm, start_field, missing_mask, missing_mask_1ch, gp_std_field, t_target, seed=42):
    """Build noise iteratively from t=0 to t_target using the forward process step-by-step."""
    torch.manual_seed(seed)
    x = start_field.clone()
    # Forward diffuse step by step: x_t = sqrt(alpha_t) * x_{t-1} + sqrt(1-alpha_t) * eps
    betas = ddpm.betas.to(device)
    for t in range(t_target):
        beta_t = betas[t]
        alpha_t = 1.0 - beta_t
        noise = torch.randn_like(x)
        x = alpha_t.sqrt() * x + beta_t.sqrt() * noise
    # Now x is at ~t_target. Apply mask_xt.
    known_mask = 1.0 - missing_mask
    indep_noise = torch.randn_like(x)
    x_for_model = x * missing_mask[:, :2] + indep_noise * known_mask[:, :2]
    x_cond = torch.cat([x_for_model, missing_mask_1ch, gp_std_field], dim=1)
    time_tensor = torch.full((1, 1), t_target, device=device, dtype=torch.long)
    with torch.no_grad():
        return ddpm.network(x_cond, time_tensor)


# ── Strategy D: Iterative noise buildup + NO mask_xt ──
def iterative_buildup_no_maskxt(ddpm, start_field, missing_mask, missing_mask_1ch, gp_std_field, t_target, seed=42):
    """Build noise iteratively, no mask_xt."""
    torch.manual_seed(seed)
    x = start_field.clone()
    betas = ddpm.betas.to(device)
    for t in range(t_target):
        beta_t = betas[t]
        alpha_t = 1.0 - beta_t
        noise = torch.randn_like(x)
        x = alpha_t.sqrt() * x + beta_t.sqrt() * noise
    x_cond = torch.cat([x, missing_mask_1ch, gp_std_field], dim=1)
    time_tensor = torch.full((1, 1), t_target, device=device, dtype=torch.long)
    with torch.no_grad():
        return ddpm.network(x_cond, time_tensor)


# ── Ensemble wrapper ──
def ensemble(fn, ddpm, start_field, missing_mask, missing_mask_1ch, gp_std_field, t_val, n_avg):
    total = torch.zeros_like(start_field)
    for k in range(n_avg):
        pred = fn(ddpm, start_field, missing_mask, missing_mask_1ch, gp_std_field, t_val, seed=42 + k * 1000)
        total += pred
    return total / n_avg


# ── Paste-back wrapper: paste known observations into prediction ──
def with_pasteback(fn, ddpm, start_field, missing_mask, missing_mask_1ch, gp_std_field, known_std, t_val, seed=42):
    pred = fn(ddpm, start_field, missing_mask, missing_mask_1ch, gp_std_field, t_val, seed=seed)
    known_mask = 1.0 - missing_mask[:, :2]
    # Paste in the actual known observations (standardized)
    return pred * missing_mask[:, :2] + known_std * known_mask


# Pre-compute samples
samples = []
for i in range(n_test):
    vi = val_indices[i]
    data = prepare_sample(vi)
    samples.append((vi, data))

gp_mses = [s[1][4].item() for s in samples]
mean_gp = np.mean(gp_mses)


def evaluate_strategy(ddpm, name, strategy_fn):
    mses, wins = [], 0
    for i, (vi, (input_orig, input_std, missing_mask, missing_mask_1ch, gp_mse, gp_std_field, known_std)) in enumerate(samples):
        result = strategy_fn(ddpm, gp_std_field, missing_mask, missing_mask_1ch, known_std, i)
        mse = compute_mse(result, input_orig, missing_mask).item()
        mses.append(mse)
        if mse < gp_mse.item():
            wins += 1
    mean_mse = np.mean(mses)
    ratio = mean_mse / mean_gp
    print(f"{name:<50} | {mean_mse:>10.6f} | {ratio:>5.2f}x | {wins}/{n_test}")
    return mean_mse


# ── Load model ──
print("\n" + "=" * 85)
print("mask_xt ablation + iterative buildup + t=75 + paste-back")
print("=" * 85)
model = load_model(os.path.join(RESULTS_DIR, "inpaint_gaussian_t250_best_ema_weights.pt"))

print(f"\n{'Strategy':<50} | {'Mean MSE':>10} | {'vsGP':>5} | {'Wins':>5}")
print("-" * 85)
print(f"{'GP baseline':<50} | {mean_gp:>10.6f} | {'1.00x':>5} | {'---':>5}")

# ─── Section 1: mask_xt ON vs OFF at different t values ───
print("\n--- mask_xt ON (original) vs OFF (noised GP everywhere) ---")

for t in [75, 150, 200, 225]:
    evaluate_strategy(model, f"mask_xt ON  @ t={t}",
        lambda m, gp, mm, mm1, ks, i, _t=t: single_step_maskxt(m, gp, mm, mm1, gp, _t))
    evaluate_strategy(model, f"mask_xt OFF @ t={t}",
        lambda m, gp, mm, mm1, ks, i, _t=t: single_step_no_maskxt(m, gp, mm, mm1, gp, _t))

# ─── Section 2: Iterative buildup vs 1-shot ───
print("\n--- Iterative noise buildup vs 1-shot ---")

for t in [75, 150, 200]:
    evaluate_strategy(model, f"1-shot + mask_xt @ t={t}",
        lambda m, gp, mm, mm1, ks, i, _t=t: single_step_maskxt(m, gp, mm, mm1, gp, _t))
    evaluate_strategy(model, f"iterative + mask_xt @ t={t}",
        lambda m, gp, mm, mm1, ks, i, _t=t: iterative_buildup_maskxt(m, gp, mm, mm1, gp, _t))
    evaluate_strategy(model, f"iterative + no mask_xt @ t={t}",
        lambda m, gp, mm, mm1, ks, i, _t=t: iterative_buildup_no_maskxt(m, gp, mm, mm1, gp, _t))

# ─── Section 3: Ensemble at t=75 ───
print("\n--- Ensemble at t=75 ---")

evaluate_strategy(model, "Ens 3x mask_xt @ t=75",
    lambda m, gp, mm, mm1, ks, i: ensemble(single_step_maskxt, m, gp, mm, mm1, gp, 75, 3))
evaluate_strategy(model, "Ens 10x mask_xt @ t=75",
    lambda m, gp, mm, mm1, ks, i: ensemble(single_step_maskxt, m, gp, mm, mm1, gp, 75, 10))
evaluate_strategy(model, "Ens 3x no_maskxt @ t=75",
    lambda m, gp, mm, mm1, ks, i: ensemble(single_step_no_maskxt, m, gp, mm, mm1, gp, 75, 3))
evaluate_strategy(model, "Ens 10x no_maskxt @ t=75",
    lambda m, gp, mm, mm1, ks, i: ensemble(single_step_no_maskxt, m, gp, mm, mm1, gp, 75, 10))

# ─── Section 4: Ensemble at t=200 for comparison ───
print("\n--- Ensemble at t=200 (reference) ---")

evaluate_strategy(model, "Ens 10x mask_xt @ t=200",
    lambda m, gp, mm, mm1, ks, i: ensemble(single_step_maskxt, m, gp, mm, mm1, gp, 200, 10))
evaluate_strategy(model, "Ens 10x no_maskxt @ t=200",
    lambda m, gp, mm, mm1, ks, i: ensemble(single_step_no_maskxt, m, gp, mm, mm1, gp, 200, 10))

# ─── Section 5: Paste-back (paste known obs into output) ───
print("\n--- Paste-back: paste known observations into DDPM output ---")

for t in [75, 200]:
    evaluate_strategy(model, f"mask_xt @ t={t} + pasteback",
        lambda m, gp, mm, mm1, ks, i, _t=t: with_pasteback(
            single_step_maskxt, m, gp, mm, mm1, gp, ks, _t))
    evaluate_strategy(model, f"no_maskxt @ t={t} + pasteback",
        lambda m, gp, mm, mm1, ks, i, _t=t: with_pasteback(
            single_step_no_maskxt, m, gp, mm, mm1, gp, ks, _t))

# Best combo: ensemble + paste-back
print("\n--- Best combos: ensemble + paste-back ---")

def ens_pasteback(fn, ddpm, gp, mm, mm1, gp_cond, known_std, t_val, n_avg):
    total = torch.zeros_like(gp)
    for k in range(n_avg):
        pred = fn(ddpm, gp, mm, mm1, gp_cond, t_val, seed=42 + k * 1000)
        total += pred
    avg = total / n_avg
    known_mask = 1.0 - mm[:, :2]
    return avg * mm[:, :2] + known_std * known_mask

evaluate_strategy(model, "Ens10x mask_xt@200 + pasteback",
    lambda m, gp, mm, mm1, ks, i: ens_pasteback(
        single_step_maskxt, m, gp, mm, mm1, gp, ks, 200, 10))
evaluate_strategy(model, "Ens10x no_maskxt@200 + pasteback",
    lambda m, gp, mm, mm1, ks, i: ens_pasteback(
        single_step_no_maskxt, m, gp, mm, mm1, gp, ks, 200, 10))
evaluate_strategy(model, "Ens10x mask_xt@75 + pasteback",
    lambda m, gp, mm, mm1, ks, i: ens_pasteback(
        single_step_maskxt, m, gp, mm, mm1, gp, ks, 75, 10))
evaluate_strategy(model, "Ens10x no_maskxt@75 + pasteback",
    lambda m, gp, mm, mm1, ks, i: ens_pasteback(
        single_step_no_maskxt, m, gp, mm, mm1, gp, ks, 75, 10))
