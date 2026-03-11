#!/usr/bin/env python3
"""Diagnostic: check model's x0 prediction quality at individual timesteps.

If single-step predictions at low t are good but full reverse is bad,
the problem is in the chain. If single-step predictions are also bad,
the model hasn't converged.
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

# Config
N_STEPS = 250
U_MEAN, U_STD = -0.06929559429949586, 0.1358005549716049
V_MEAN, V_STD = -0.0323937796117541, 0.08899177232117582
standardizer = ZScoreStandardizer(U_MEAN, U_STD, V_MEAN, V_STD)
noise_strategy = get_noise_strategy("gaussian")

dd = DDInitializer()
device = dd.get_device()
val_data = dd.get_validation_data()
dd_std = dd.get_standardizer()

# Load model (EMA weights)
WEIGHTS = os.path.join(BASE, "experiments/06_gp_forward/gp_conditioned/results/inpaint_gaussian_t250_best_ema_weights.pt")
network = MyUNet_FiLM_Attn(n_steps=N_STEPS, time_emb_dim=256, in_channels=5)
ddpm = GaussianDDPM(network, n_steps=N_STEPS, min_beta=0.0001, max_beta=0.02, device=device)
ddpm.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=False))
ddpm = ddpm.to(device)
ddpm.eval()

# Reference data
GPDIFF_PT = os.path.join(BASE, "results/eddy_balanced_eval/bulk_eval_eddy_balanced_100.pt")
gpdiff_data = torch.load(GPDIFF_PT, map_location="cpu", weights_only=False)
val_indices = gpdiff_data["val_indices"]

# Use first 3 samples
n_test = 3
test_timesteps = [1, 5, 10, 25, 50, 100, 150, 200, 249]

print(f"{'t':>5} | {'abar':>8} | {'MSE(pred_x0, GT)':>18} | {'MSE(GP, GT)':>12} | Ratio")
print("-" * 75)

for vi_idx in range(n_test):
    vi = val_indices[vi_idx]
    
    # Prepare data
    input_image_dd = val_data[vi][0].unsqueeze(0)
    input_orig = dd_std.unstandardize(input_image_dd.squeeze(0)).unsqueeze(0).to(device)
    input_std = standardizer(input_orig.squeeze(0)).unsqueeze(0).to(device)

    # Mask
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

    known_mask = 1.0 - missing_mask
    known_values = gp_std_field

    print(f"\n=== Sample {vi_idx+1} (val {vi}) — GP MSE: {gp_mse.item():.6f} ===")

    for t_val in test_timesteps:
        t_tensor = torch.tensor([t_val], device=device)
        alpha_bar_t = ddpm.alpha_bars[t_val].to(device)

        # Forward diffusion: add noise to GT at timestep t
        torch.manual_seed(42 + vi_idx * 1000 + t_val)
        noise = torch.randn_like(input_std)
        noisy = alpha_bar_t.sqrt() * input_std + (1 - alpha_bar_t).sqrt() * noise

        # Apply mask_xt: replace known region with independent noise
        indep_noise = torch.randn_like(noisy)
        x_for_model = noisy * missing_mask[:, :2] + indep_noise * known_mask[:, :2]

        # Build 5-ch input
        x_cond = torch.cat([x_for_model, missing_mask_1ch, known_values], dim=1)

        # Model predicts x0
        with torch.no_grad():
            time_tensor = torch.full((1, 1), t_val, device=device, dtype=torch.long)
            x0_pred = ddpm.network(x_cond, time_tensor)

        # Compute MSE of prediction in physical space (missing region only)
        x0_pred_phys = standardizer.unstandardize(x0_pred.squeeze(0)).unsqueeze(0).to(device)
        pred_mse = ((x0_pred_phys - input_orig) * missing_mask).pow(2).sum() / (missing_mask.sum() + 1e-8)
        
        ratio = pred_mse.item() / gp_mse.item()
        print(f"{t_val:>5} | {alpha_bar_t.item():>8.4f} | {pred_mse.item():>18.6f} | {gp_mse.item():>12.6f} | {ratio:.3f}x")

print("\n\nInterpretation:")
print("  t=1: x_t ≈ x0 (almost no noise). Model should easily predict x0.")
print("  t=249: x_t ≈ pure noise. Model must rely entirely on GP conditioning.")
print("  If ratio > 1 at low t: model hasn't converged (can't even copy near-clean input)")
print("  If ratio < 1 at low t but > 1 at high t: model works locally but breaks at high noise")
