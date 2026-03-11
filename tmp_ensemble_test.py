#!/usr/bin/env python3
"""Test ensemble averaging and two-stage refinement with v2 weights.

Best single-step strategy: noise GP at t=200, predict x0 → 0.88x GP.
Can we improve with:
  1) Average K single-step predictions (different noise seeds)
  2) Two-stage: refine GP at t=200, then refine result at t=50
  3) Non-EMA vs EMA weights
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
noise_strategy = get_noise_strategy("gaussian")

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
    return input_orig, input_std, missing_mask, missing_mask_1ch, gp_mse, gp_std_field


def compute_mse(pred_std, gt_orig, mask):
    pred_phys = standardizer.unstandardize(pred_std.squeeze(0)).unsqueeze(0).to(device)
    return ((pred_phys - gt_orig) * mask).pow(2).sum() / (mask.sum() + 1e-8)


def single_step(ddpm, start_field, missing_mask, missing_mask_1ch, gp_std_field, t_val, seed=42):
    """Single-step refinement: noise start_field at t_val, predict x0."""
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


def ensemble_avg(ddpm, start_field, missing_mask, missing_mask_1ch, gp_std_field, t_val, n_avg):
    """Average n_avg single-step predictions with different noise seeds."""
    total = torch.zeros_like(start_field)
    for k in range(n_avg):
        pred = single_step(ddpm, start_field, missing_mask, missing_mask_1ch, gp_std_field, t_val, seed=42 + k * 1000)
        total += pred
    return total / n_avg


def two_stage(ddpm, gp_field, missing_mask, missing_mask_1ch, t1=200, t2=50, n_avg1=1, n_avg2=1):
    """Two-stage: refine GP at t1, then refine result at t2."""
    # Stage 1: GP -> refined
    if n_avg1 > 1:
        refined = ensemble_avg(ddpm, gp_field, missing_mask, missing_mask_1ch, gp_field, t1, n_avg1)
    else:
        refined = single_step(ddpm, gp_field, missing_mask, missing_mask_1ch, gp_field, t1)
    # Stage 2: refined -> final
    if n_avg2 > 1:
        final = ensemble_avg(ddpm, refined, missing_mask, missing_mask_1ch, gp_field, t2, n_avg2)
    else:
        final = single_step(ddpm, refined, missing_mask, missing_mask_1ch, gp_field, t2, seed=999)
    return final


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
    for i, (vi, (input_orig, input_std, missing_mask, missing_mask_1ch, gp_mse, gp_std_field)) in enumerate(samples):
        result = strategy_fn(ddpm, gp_std_field, missing_mask, missing_mask_1ch, i)
        mse = compute_mse(result, input_orig, missing_mask).item()
        mses.append(mse)
        if mse < gp_mse.item():
            wins += 1
    mean_mse = np.mean(mses)
    ratio = mean_mse / mean_gp
    print(f"{name:<40} | {mean_mse:>10.6f} | {ratio:>5.2f}x | {wins}/{n_test}")
    return mean_mse


# ── Test with v2 EMA weights (epoch ~157) ────────────────────────
print("\n" + "=" * 75)
print("v2 EMA weights (epoch ~157, EMA test loss 0.024)")
print("=" * 75)
ema_model = load_model(os.path.join(RESULTS_DIR, "inpaint_gaussian_t250_best_ema_weights.pt"))

print(f"\n{'Strategy':<40} | {'Mean MSE':>10} | {'vsGP':>5} | {'Wins':>5}")
print("-" * 75)
print(f"{'GP baseline':<40} | {mean_gp:>10.6f} | {'1.00x':>5} | {'---':>5}")

# Single step baselines
evaluate_strategy(ema_model, "1-step GP @ t=200",
    lambda m, gp, mm, mm1, i: single_step(m, gp, mm, mm1, gp, 200))
evaluate_strategy(ema_model, "1-step GP @ t=225",
    lambda m, gp, mm, mm1, i: single_step(m, gp, mm, mm1, gp, 225))
evaluate_strategy(ema_model, "1-step GP @ t=240",
    lambda m, gp, mm, mm1, i: single_step(m, gp, mm, mm1, gp, 240))

# Ensemble averaging (t=200)
evaluate_strategy(ema_model, "Ensemble 3x GP @ t=200",
    lambda m, gp, mm, mm1, i: ensemble_avg(m, gp, mm, mm1, gp, 200, 3))
evaluate_strategy(ema_model, "Ensemble 5x GP @ t=200",
    lambda m, gp, mm, mm1, i: ensemble_avg(m, gp, mm, mm1, gp, 200, 5))
evaluate_strategy(ema_model, "Ensemble 10x GP @ t=200",
    lambda m, gp, mm, mm1, i: ensemble_avg(m, gp, mm, mm1, gp, 200, 10))
evaluate_strategy(ema_model, "Ensemble 20x GP @ t=200",
    lambda m, gp, mm, mm1, i: ensemble_avg(m, gp, mm, mm1, gp, 200, 20))

# Two-stage
evaluate_strategy(ema_model, "2-stage: ens3@200 → ens3@50",
    lambda m, gp, mm, mm1, i: two_stage(m, gp, mm, mm1, 200, 50, 3, 3))

# ── Compare with v2 EMA epoch-32 weights ────────────────────────
print("\n" + "=" * 75)
print("v2 EMA weights (epoch ~32, EMA test loss 0.027) [PREVIOUS BEST]")
print("=" * 75)
old_model = load_model(os.path.join(RESULTS_DIR, "v2_epoch32_best_ema_weights.pt"))

print(f"\n{'Strategy':<40} | {'Mean MSE':>10} | {'vsGP':>5} | {'Wins':>5}")
print("-" * 75)
evaluate_strategy(old_model, "1-step GP @ t=200",
    lambda m, gp, mm, mm1, i: single_step(m, gp, mm, mm1, gp, 200))
evaluate_strategy(old_model, "Ensemble 10x GP @ t=200",
    lambda m, gp, mm, mm1, i: ensemble_avg(m, gp, mm, mm1, gp, 200, 10))
evaluate_strategy(old_model, "Ensemble 20x GP @ t=200",
    lambda m, gp, mm, mm1, i: ensemble_avg(m, gp, mm, mm1, gp, 200, 20))
