#!/usr/bin/env python3
"""Test alternative inference strategies to fix chain error amplification.

Finding: Model's single-step x0 predictions are excellent (0.05-1.0x GP)
but the 250-step reverse chain gives 1.9x worse. The chain compounds errors.

Test strategies:
  A) Single-step from t=249 (just predict x0 from pure noise)
  B) Full reverse with RPaste (paste known GT at each step)
  C) Full reverse without mask_xt (let model see x_t in known region)
  D) Truncated reverse from t=50 (start from lightly noised GT)
"""
import sys, os, time, torch
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

# Load model
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

n_test = 5

results = {
    "A_singlestep_t249": [],
    "B_repaste_250": [],
    "C_nomaskxt_250": [],
    "D_truncated_t50": [],
    "E_singlestep_t50": [],
    "Full_reverse_baseline": [],
    "GP": [],
}


def prepare_sample(vi):
    """Prepare input data for a validation sample."""
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
    """Compute physical-space MSE in missing region."""
    pred_phys = standardizer.unstandardize(pred_std.squeeze(0)).unsqueeze(0).to(device)
    return ((pred_phys - gt_orig) * mask).pow(2).sum() / (mask.sum() + 1e-8)


def strategy_A_singlestep(ddpm, input_std, missing_mask, missing_mask_1ch, gp_std_field, t_val=249):
    """Single-step: predict x0 from pure noise at t=t_val."""
    torch.manual_seed(42)
    noise = torch.randn(1, 2, 64, 128, device=device)
    alpha_bar = ddpm.alpha_bars[t_val].to(device)
    noisy = alpha_bar.sqrt() * input_std + (1 - alpha_bar).sqrt() * noise
    
    # mask_xt
    known_mask = 1.0 - missing_mask
    indep_noise = torch.randn_like(noisy)
    x_for_model = noisy * missing_mask[:, :2] + indep_noise * known_mask[:, :2]
    
    x_cond = torch.cat([x_for_model, missing_mask_1ch, gp_std_field], dim=1)
    time_tensor = torch.full((1, 1), t_val, device=device, dtype=torch.long)
    
    with torch.no_grad():
        x0_pred = ddpm.network(x_cond, time_tensor)
    return x0_pred


def strategy_B_repaste(ddpm, input_std, missing_mask, missing_mask_1ch, gp_std_field):
    """Full 250 reverse with GT known-region paste at each step (RePaint-style)."""
    torch.manual_seed(42)
    x = torch.randn(1, 2, 64, 128, device=device)
    known_mask = 1.0 - missing_mask
    
    with torch.no_grad():
        for t in range(N_STEPS - 1, -1, -1):
            alpha_t = ddpm.alphas[t].to(device)
            alpha_bar_t = ddpm.alpha_bars[t].to(device)
            beta_t = ddpm.betas[t].to(device)
            
            time_tensor = torch.full((1, 1), t, device=device, dtype=torch.long)
            
            # mask_xt
            indep_noise = torch.randn_like(x)
            x_for_model = x * missing_mask[:, :2] + indep_noise * known_mask[:, :2]
            
            x_cond = torch.cat([x_for_model, missing_mask_1ch, gp_std_field], dim=1)
            x0_pred = ddpm.network(x_cond, time_tensor)
            
            if t > 0:
                alpha_bar_prev = ddpm.alpha_bars[t - 1].to(device)
                coeff_x0 = (alpha_bar_prev.sqrt() * beta_t) / (1 - alpha_bar_t)
                coeff_xt = (alpha_t.sqrt() * (1 - alpha_bar_prev)) / (1 - alpha_bar_t)
                mu = coeff_x0 * x0_pred + coeff_xt * x
                beta_tilde = ((1 - alpha_bar_prev) / (1 - alpha_bar_t)) * beta_t
                z = torch.randn_like(x)
                x = mu + beta_tilde.sqrt() * z
                
                # PASTE known region from noised GT
                noise_known = torch.randn_like(input_std)
                x_known = alpha_bar_prev.sqrt() * input_std + (1 - alpha_bar_prev).sqrt() * noise_known
                x = x_known * known_mask + x * missing_mask
            else:
                x = x0_pred
    return x


def strategy_C_nomaskxt(ddpm, input_std, missing_mask, missing_mask_1ch, gp_std_field):
    """Full 250 reverse WITHOUT mask_xt."""
    torch.manual_seed(42)
    x = torch.randn(1, 2, 64, 128, device=device)
    known_mask = 1.0 - missing_mask
    
    with torch.no_grad():
        for t in range(N_STEPS - 1, -1, -1):
            alpha_t = ddpm.alphas[t].to(device)
            alpha_bar_t = ddpm.alpha_bars[t].to(device)
            beta_t = ddpm.betas[t].to(device)
            
            time_tensor = torch.full((1, 1), t, device=device, dtype=torch.long)
            
            # NO mask_xt — model sees full x_t
            x_cond = torch.cat([x, missing_mask_1ch, gp_std_field], dim=1)
            x0_pred = ddpm.network(x_cond, time_tensor)
            
            if t > 0:
                alpha_bar_prev = ddpm.alpha_bars[t - 1].to(device)
                coeff_x0 = (alpha_bar_prev.sqrt() * beta_t) / (1 - alpha_bar_t)
                coeff_xt = (alpha_t.sqrt() * (1 - alpha_bar_prev)) / (1 - alpha_bar_t)
                mu = coeff_x0 * x0_pred + coeff_xt * x
                beta_tilde = ((1 - alpha_bar_prev) / (1 - alpha_bar_t)) * beta_t
                z = torch.randn_like(x)
                x = mu + beta_tilde.sqrt() * z
            else:
                x = x0_pred
    return x


def strategy_D_truncated(ddpm, input_std, missing_mask, missing_mask_1ch, gp_std_field, t_start=50):
    """Start from lightly noised GT at t=t_start, reverse to t=0."""
    torch.manual_seed(42)
    alpha_bar_start = ddpm.alpha_bars[t_start].to(device)
    noise = torch.randn_like(input_std)
    x = alpha_bar_start.sqrt() * input_std + (1 - alpha_bar_start).sqrt() * noise
    known_mask = 1.0 - missing_mask
    
    with torch.no_grad():
        for t in range(t_start, -1, -1):
            alpha_t = ddpm.alphas[t].to(device)
            alpha_bar_t = ddpm.alpha_bars[t].to(device)
            beta_t = ddpm.betas[t].to(device)
            
            time_tensor = torch.full((1, 1), t, device=device, dtype=torch.long)
            
            # mask_xt
            indep_noise = torch.randn_like(x)
            x_for_model = x * missing_mask[:, :2] + indep_noise * known_mask[:, :2]
            
            x_cond = torch.cat([x_for_model, missing_mask_1ch, gp_std_field], dim=1)
            x0_pred = ddpm.network(x_cond, time_tensor)
            
            if t > 0:
                alpha_bar_prev = ddpm.alpha_bars[t - 1].to(device)
                coeff_x0 = (alpha_bar_prev.sqrt() * beta_t) / (1 - alpha_bar_t)
                coeff_xt = (alpha_t.sqrt() * (1 - alpha_bar_prev)) / (1 - alpha_bar_t)
                mu = coeff_x0 * x0_pred + coeff_xt * x
                beta_tilde = ((1 - alpha_bar_prev) / (1 - alpha_bar_t)) * beta_t
                z = torch.randn_like(x)
                x = mu + beta_tilde.sqrt() * z
            else:
                x = x0_pred
    return x


def strategy_E_singlestep_t50(ddpm, input_std, missing_mask, missing_mask_1ch, gp_std_field):
    """Single step from lightly noised GT at t=50."""
    return strategy_A_singlestep(ddpm, input_std, missing_mask, missing_mask_1ch, gp_std_field, t_val=50)


# Full reverse (baseline)
from ddpm.utils.inpainting_utils import x0_full_reverse_inpaint

print(f"\n{'Strategy':<25} | {'Samp1':>8} {'Samp2':>8} {'Samp3':>8} {'Samp4':>8} {'Samp5':>8} | {'Mean':>8} {'vsGP':>6}")
print("-" * 95)

strategies = [
    ("GP baseline", None),
    ("Full reverse (baseline)", "Full_reverse_baseline"),
    ("A: Single-step t=249", "A_singlestep_t249"),
    ("B: RePaste (250 steps)", "B_repaste_250"),
    ("C: No mask_xt (250)", "C_nomaskxt_250"),
    ("D: Truncated t=50", "D_truncated_t50"),
    ("E: Single-step t=50", "E_singlestep_t50"),
]

# Pre-compute sample data
samples = []
for i in range(n_test):
    vi = val_indices[i]
    data = prepare_sample(vi)
    samples.append((vi, data))

gp_mses = [s[1][4].item() for s in samples]
mean_gp_mse = np.mean(gp_mses)

for strat_name, strat_key in strategies:
    mses = []
    for i, (vi, (input_orig, input_std, missing_mask, missing_mask_1ch, gp_mse, gp_std_field)) in enumerate(samples):
        torch.manual_seed(42 + i)
        
        if strat_key is None:  # GP baseline
            mse = gp_mse.item()
        elif strat_key == "Full_reverse_baseline":
            result = x0_full_reverse_inpaint(
                ddpm, input_std, missing_mask,
                n_samples=1, device=device,
                noise_strategy=noise_strategy,
                mask_xt=True,
                known_values_override=gp_std_field,
            )
            mse = compute_mse(result, input_orig, missing_mask).item()
        elif strat_key == "A_singlestep_t249":
            result = strategy_A_singlestep(ddpm, input_std, missing_mask, missing_mask_1ch, gp_std_field, 249)
            mse = compute_mse(result, input_orig, missing_mask).item()
        elif strat_key == "B_repaste_250":
            result = strategy_B_repaste(ddpm, input_std, missing_mask, missing_mask_1ch, gp_std_field)
            mse = compute_mse(result, input_orig, missing_mask).item()
        elif strat_key == "C_nomaskxt_250":
            result = strategy_C_nomaskxt(ddpm, input_std, missing_mask, missing_mask_1ch, gp_std_field)
            mse = compute_mse(result, input_orig, missing_mask).item()
        elif strat_key == "D_truncated_t50":
            result = strategy_D_truncated(ddpm, input_std, missing_mask, missing_mask_1ch, gp_std_field, 50)
            mse = compute_mse(result, input_orig, missing_mask).item()
        elif strat_key == "E_singlestep_t50":
            result = strategy_E_singlestep_t50(ddpm, input_std, missing_mask, missing_mask_1ch, gp_std_field)
            mse = compute_mse(result, input_orig, missing_mask).item()
        
        mses.append(mse)

    mean_mse = np.mean(mses)
    ratio = mean_mse / mean_gp_mse
    vals = " ".join(f"{m:>8.5f}" for m in mses)
    print(f"{strat_name:<25} | {vals} | {mean_mse:>8.5f} {ratio:>5.2f}x")
