#!/usr/bin/env python3
"""Test GP → GT refinement: noise GP field, apply model to predict GT.

Key insight: single-step at t=50 with noised GT gives 0.10x GP (incredible).
At inference we don't have GT, but we have GP. What if we noise the GP field
and ask the model to refine it to GT in one (or few) steps?

Also test multi-step DDIM-like approaches with various step counts.
"""
import sys, os, torch
import numpy as np
from tqdm import tqdm

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

# Load model
WEIGHTS = os.path.join(BASE, "experiments/06_gp_forward/gp_conditioned/results/inpaint_gaussian_t250_best_ema_weights.pt")
network = MyUNet_FiLM_Attn(n_steps=N_STEPS, time_emb_dim=256, in_channels=5)
ddpm = GaussianDDPM(network, n_steps=N_STEPS, min_beta=0.0001, max_beta=0.02, device=device)
ddpm.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=False))
ddpm = ddpm.to(device)
ddpm.eval()

GPDIFF_PT = os.path.join(BASE, "results/eddy_balanced_eval/bulk_eval_eddy_balanced_100.pt")
gpdiff_data = torch.load(GPDIFF_PT, map_location="cpu", weights_only=False)
val_indices = gpdiff_data["val_indices"]

n_test = 10


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


def single_step_refine(ddpm, start_field_std, missing_mask, missing_mask_1ch, gp_std_field, t_val):
    """Single-step: noise start_field at t_val, predict x0."""
    alpha_bar = ddpm.alpha_bars[t_val].to(device)
    noise = torch.randn_like(start_field_std)
    noisy = alpha_bar.sqrt() * start_field_std + (1 - alpha_bar).sqrt() * noise
    
    # mask_xt
    known_mask = 1.0 - missing_mask
    indep_noise = torch.randn_like(noisy)
    x_for_model = noisy * missing_mask[:, :2] + indep_noise * known_mask[:, :2]
    
    x_cond = torch.cat([x_for_model, missing_mask_1ch, gp_std_field], dim=1)
    time_tensor = torch.full((1, 1), t_val, device=device, dtype=torch.long)
    
    with torch.no_grad():
        x0_pred = ddpm.network(x_cond, time_tensor)
    return x0_pred


def multi_step_refine(ddpm, start_field_std, missing_mask, missing_mask_1ch, gp_std_field, t_start, n_steps):
    """Multi-step reverse from t_start with n_steps evenly spaced."""
    known_mask = 1.0 - missing_mask
    
    # Generate schedule
    timesteps = list(range(t_start, -1, -max(1, t_start // n_steps)))[:n_steps]
    if timesteps[-1] != 0:
        timesteps.append(0)
    
    # Start from noised start field
    alpha_bar_start = ddpm.alpha_bars[t_start].to(device)
    noise = torch.randn_like(start_field_std)
    x = alpha_bar_start.sqrt() * start_field_std + (1 - alpha_bar_start).sqrt() * noise
    
    with torch.no_grad():
        for step_idx in range(len(timesteps)):
            t = timesteps[step_idx]
            alpha_bar_t = ddpm.alpha_bars[t].to(device)
            
            time_tensor = torch.full((1, 1), t, device=device, dtype=torch.long)
            
            indep_noise = torch.randn_like(x)
            x_for_model = x * missing_mask[:, :2] + indep_noise * known_mask[:, :2]
            x_cond = torch.cat([x_for_model, missing_mask_1ch, gp_std_field], dim=1)
            x0_pred = ddpm.network(x_cond, time_tensor)
            
            if step_idx < len(timesteps) - 1:
                t_next = timesteps[step_idx + 1]
                if t_next == 0:
                    x = x0_pred
                else:
                    # DDIM-style: deterministic step to t_next
                    alpha_bar_next = ddpm.alpha_bars[t_next].to(device)
                    # Reconstruct eps from x0_pred
                    eps = (x - alpha_bar_t.sqrt() * x0_pred) / (1 - alpha_bar_t).sqrt().clamp(min=1e-8)
                    x = alpha_bar_next.sqrt() * x0_pred + (1 - alpha_bar_next).sqrt() * eps
            else:
                x = x0_pred
    return x


# Pre-compute
samples = []
for i in range(n_test):
    vi = val_indices[i]
    data = prepare_sample(vi)
    samples.append((vi, data))

gp_mses = [s[1][4].item() for s in samples]
mean_gp = np.mean(gp_mses)

print(f"\n{'Strategy':<35} | {'Mean MSE':>10} | {'vs GP':>6} | {'Win%':>5}")
print("-" * 70)

strategies = [
    ("GP baseline", "gp"),
    # Single-step from pure noise
    ("1-step from noise (t=249)", ("noise", 249, 1)),
    ("1-step from noise (t=200)", ("noise", 200, 1)),
    ("1-step from noise (t=150)", ("noise", 150, 1)),
    ("1-step from noise (t=100)", ("noise", 100, 1)),
    # Single-step from noised GP
    ("1-step from GP (t=200)", ("gp_refine", 200, 1)),
    ("1-step from GP (t=150)", ("gp_refine", 150, 1)),
    ("1-step from GP (t=100)", ("gp_refine", 100, 1)),
    ("1-step from GP (t=50)", ("gp_refine", 50, 1)),
    ("1-step from GP (t=25)", ("gp_refine", 25, 1)),
    ("1-step from GP (t=10)", ("gp_refine", 10, 1)),
    # Multi-step from noised GP
    ("5-step DDIM from GP (t=100)", ("gp_multi", 100, 5)),
    ("10-step DDIM from GP (t=100)", ("gp_multi", 100, 10)),
    ("5-step DDIM from GP (t=50)", ("gp_multi", 50, 5)),
]

for strat_name, strat_spec in strategies:
    mses = []
    wins = 0
    for i, (vi, (input_orig, input_std, missing_mask, missing_mask_1ch, gp_mse, gp_std_field)) in enumerate(samples):
        torch.manual_seed(42 + i)
        
        if strat_spec == "gp":
            mse = gp_mse.item()
        elif strat_spec[0] == "noise":
            _, t_val, _ = strat_spec
            # Start from pure noise
            x_start = torch.zeros(1, 2, 64, 128, device=device)
            result = single_step_refine(ddpm, x_start, missing_mask, missing_mask_1ch, gp_std_field, t_val)
            mse = compute_mse(result, input_orig, missing_mask).item()
        elif strat_spec[0] == "gp_refine":
            _, t_val, _ = strat_spec
            result = single_step_refine(ddpm, gp_std_field, missing_mask, missing_mask_1ch, gp_std_field, t_val)
            mse = compute_mse(result, input_orig, missing_mask).item()
        elif strat_spec[0] == "gp_multi":
            _, t_start, n_steps = strat_spec
            result = multi_step_refine(ddpm, gp_std_field, missing_mask, missing_mask_1ch, gp_std_field, t_start, n_steps)
            mse = compute_mse(result, input_orig, missing_mask).item()
        
        mses.append(mse)
        if strat_spec != "gp" and mse < gp_mse.item():
            wins += 1
    
    mean_mse = np.mean(mses)
    ratio = mean_mse / mean_gp
    win_pct = f"{wins}/{n_test}" if strat_spec != "gp" else "---"
    print(f"{strat_name:<35} | {mean_mse:>10.6f} | {ratio:>5.2f}x | {win_pct}")
