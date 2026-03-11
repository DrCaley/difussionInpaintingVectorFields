#!/usr/bin/env python3
"""Systematic A/B/C experiment sweep: comparing FiLM+GP inference strategies.

Experiment A: Best EMA weights (epoch 58) vs epoch 161 (current)
Experiment B: Low-t sweep (t_start = 1,2,5,10,25,50,75) — fewer diffusion steps
Experiment C: DDIM deterministic sampling vs DDPM stochastic

All experiments use the same 100-sample eddy-balanced evaluation set.

Usage:
    PYTHONPATH=. python3 scripts/eval_abc_sweep.py --quick 10    # fast 10-sample test
    PYTHONPATH=. python3 scripts/eval_abc_sweep.py               # full 100 samples
"""
import argparse, os, sys, time
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import matplotlib
matplotlib.use("Agg")

from data_prep.data_initializer import DDInitializer
from ddpm.neural_networks.ddpm import GaussianDDPM
from ddpm.neural_networks.unets.unet_film_attn import MyUNet_FiLM_Attn
from ddpm.helper_functions.masks.border_mask import BorderMaskGenerator
from ddpm.utils.noise_utils import get_noise_strategy
from ddpm.helper_functions.interpolation_tool import gp_fill

# ── args ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--quick", type=int, default=0, help="Run only N samples (0=all)")
args = parser.parse_args()

# ── paths ─────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(BASE_DIR, "experiments/03_conditioning/film_attn_divfree/results")
EPOCH161_WEIGHTS = os.path.join(RESULTS_DIR, "model_weights_epoch161.pt")
BEST_EMA_WEIGHTS = os.path.join(RESULTS_DIR, "inpaint_forward_diff_div_free_t250_best_ema_weights.pt")
BEST_WEIGHTS     = os.path.join(RESULTS_DIR, "inpaint_forward_diff_div_free_t250_best_weights.pt")
GPDIFF_PT = os.path.join(BASE_DIR, "results/eddy_balanced_eval/bulk_eval_eddy_balanced_100.pt")
OUT_DIR = os.path.join(BASE_DIR, "results/abc_sweep")
os.makedirs(OUT_DIR, exist_ok=True)

# ── model config ──────────────────────────────────────────────────────
N_STEPS = 250
MIN_BETA = 0.0001
MAX_BETA = 0.02
NOISE_FN = "forward_diff_div_free"


# ── fixed center mask (row 22) ───────────────────────────────────────
_fixed_mask = None
def get_fixed_center_mask(image_shape):
    global _fixed_mask
    if _fixed_mask is not None:
        return _fixed_mask
    _, _, h, w = image_shape
    area_height, area_width = 44, 94
    mid_row = area_height // 2
    mask = np.ones((h, w), dtype=np.float32)
    mask[mid_row:mid_row + 1, 0:area_width] = 0.0
    mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    border = BorderMaskGenerator().generate_mask(image_shape)
    mask = mask.to(border.device) * border
    _fixed_mask = mask
    return mask


# ═══════════════════════════════════════════════════════════════════════
#  INFERENCE METHODS
# ═══════════════════════════════════════════════════════════════════════

def single_step_x0_predict(ddpm, input_image, missing_mask, gp_std, t_val,
                            noise_strategy, device, seed, mask_xt=True):
    """Single-step x0 prediction: noise GP at level t, predict x0 in one shot.
    
    This is the most direct analogy to Voronoi-CNN: the model sees a
    slightly-noised GP field and predicts the clean field in one forward pass.
    """
    torch.manual_seed(seed)
    ddpm.eval()
    
    known_mask = 1.0 - missing_mask
    known_values = input_image * known_mask
    mask_single = missing_mask[:, 0:1]
    composite = input_image * known_mask + gp_std * missing_mask
    
    with torch.no_grad():
        alpha_bar_t = ddpm.alpha_bars[t_val].to(device)
        noise = noise_strategy(
            torch.zeros(1, 2, 64, 128, device=device),
            torch.tensor([t_val], device=device),
        )
        x_t = alpha_bar_t.sqrt() * composite + (1 - alpha_bar_t).sqrt() * noise
        
        # Build 5-channel FiLM input
        if mask_xt:
            indep_noise = torch.randn_like(x_t)
            x_for_model = x_t * missing_mask[:, :2] + indep_noise * known_mask[:, :2]
        else:
            x_for_model = x_t
        
        x_cond = torch.cat([x_for_model, mask_single, known_values], dim=1)
        time_tensor = torch.full((1, 1), t_val, device=device, dtype=torch.long)
        
        x0_pred = ddpm.network(x_cond, time_tensor)
    
    # Paste known region
    result = input_image * known_mask + x0_pred * missing_mask
    return result


def multistep_ddpm_x0(ddpm, input_image, missing_mask, gp_std, gp_var_map,
                       t_start, noise_strategy, device, seed,
                       noise_floor=0.2, gamma=3.0, resample_steps=0,
                       mask_xt=True):
    """Multi-step DDPM reverse with GP warm-start (stochastic, x0-prediction).
    
    resample_steps=0 means no RePaint resampling (pure reverse).
    """
    torch.manual_seed(seed)
    ddpm.eval()
    
    known_mask = 1.0 - missing_mask
    known_values = input_image * known_mask
    mask_single = missing_mask[:, 0:1]
    composite = input_image * known_mask + gp_std * missing_mask
    
    # Build variance-adaptive noise weight
    gp_var = gp_var_map.to(device)
    if gp_var.shape[1] == 1:
        gp_var = gp_var.expand_as(missing_mask)
    masked_var = gp_var * missing_mask
    var_max = masked_var.max()
    var_min = (masked_var[missing_mask > 0.5].min()
               if (missing_mask > 0.5).any() else torch.tensor(0.0))
    var_range = var_max - var_min
    if var_range < 1e-12:
        noise_weight = torch.ones_like(missing_mask)
    else:
        var_norm = ((masked_var - var_min) / var_range).clamp(0, 1)
        noise_weight = noise_floor + (1.0 - noise_floor) * var_norm ** gamma
    noise_weight = noise_weight * missing_mask + known_mask
    
    # Forward-diffuse composite to t_start
    t_start = min(t_start, ddpm.n_steps - 1)
    alpha_bar_t = ddpm.alpha_bars[t_start].to(device)
    noise_init = noise_strategy(
        torch.zeros(1, 2, 64, 128, device=device),
        torch.tensor([t_start], device=device),
    )
    x = alpha_bar_t.sqrt() * composite + noise_weight * (1 - alpha_bar_t).sqrt() * noise_init
    
    with torch.no_grad():
        for t in range(t_start, -1, -1):
            n_resample = max(1, resample_steps) if t > 0 else 1
            
            for r in range(n_resample):
                alpha_t = ddpm.alphas[t].to(device)
                alpha_bar_cur = ddpm.alpha_bars[t].to(device)
                beta_t = ddpm.betas[t].to(device)
                
                time_tensor = torch.full((1, 1), t, device=device, dtype=torch.long)
                
                if mask_xt:
                    indep_noise = torch.randn_like(x)
                    x_for_model = x * missing_mask[:, :2] + indep_noise * known_mask[:, :2]
                else:
                    x_for_model = x
                
                x_cond = torch.cat([x_for_model, mask_single, known_values], dim=1)
                x0_pred = ddpm.network(x_cond, time_tensor)
                
                if t > 0:
                    alpha_bar_prev = ddpm.alpha_bars[t - 1].to(device)
                    coeff_x0 = (alpha_bar_prev.sqrt() * beta_t) / (1 - alpha_bar_cur)
                    coeff_xt = (alpha_t.sqrt() * (1 - alpha_bar_prev)) / (1 - alpha_bar_cur)
                    mu = coeff_x0 * x0_pred + coeff_xt * x
                    beta_tilde = ((1 - alpha_bar_prev) / (1 - alpha_bar_cur)) * beta_t
                    sigma_t = beta_tilde.sqrt()
                    z = noise_strategy(
                        torch.zeros_like(x),
                        torch.tensor([t], device=device),
                    )
                    x_denoised = mu + noise_weight * sigma_t * z
                else:
                    x_denoised = x0_pred
                
                # Paste known region
                if t > 0:
                    alpha_bar_prev = ddpm.alpha_bars[t - 1].to(device)
                    noise_known = noise_strategy(
                        torch.zeros_like(input_image),
                        torch.tensor([t - 1], device=device),
                    )
                    x_known = alpha_bar_prev.sqrt() * input_image + (1 - alpha_bar_prev).sqrt() * noise_known
                    x = x_known * known_mask + x_denoised * missing_mask
                else:
                    x = input_image * known_mask + x_denoised * missing_mask
                
                # RePaint re-noise
                if resample_steps > 0 and r < n_resample - 1 and t > 0:
                    noise_back = noise_strategy(
                        torch.zeros_like(x),
                        torch.tensor([t], device=device),
                    )
                    x = alpha_t.sqrt() * x + noise_weight * (1 - alpha_t).sqrt() * noise_back
    
    return x


def ddim_x0(ddpm, input_image, missing_mask, gp_std, gp_var_map,
            t_start, noise_strategy, device, seed,
            noise_floor=0.2, gamma=3.0, ddim_steps=None,
            eta=0.0, mask_xt=True):
    """DDIM deterministic sampling with GP warm-start (x0-prediction model).
    
    eta=0.0 → fully deterministic (DDIM)
    eta=1.0 → equivalent to DDPM stochastic
    
    ddim_steps: number of evenly-spaced steps to use (None = use all t_start steps)
    """
    torch.manual_seed(seed)
    ddpm.eval()
    
    known_mask = 1.0 - missing_mask
    known_values = input_image * known_mask
    mask_single = missing_mask[:, 0:1]
    composite = input_image * known_mask + gp_std * missing_mask
    
    # Build variance-adaptive noise weight
    gp_var = gp_var_map.to(device)
    if gp_var.shape[1] == 1:
        gp_var = gp_var.expand_as(missing_mask)
    masked_var = gp_var * missing_mask
    var_max = masked_var.max()
    var_min = (masked_var[missing_mask > 0.5].min()
               if (missing_mask > 0.5).any() else torch.tensor(0.0))
    var_range = var_max - var_min
    if var_range < 1e-12:
        noise_weight = torch.ones_like(missing_mask)
    else:
        var_norm = ((masked_var - var_min) / var_range).clamp(0, 1)
        noise_weight = noise_floor + (1.0 - noise_floor) * var_norm ** gamma
    noise_weight = noise_weight * missing_mask + known_mask
    
    # Forward-diffuse composite to t_start
    t_start = min(t_start, ddpm.n_steps - 1)
    alpha_bar_t = ddpm.alpha_bars[t_start].to(device)
    noise_init = noise_strategy(
        torch.zeros(1, 2, 64, 128, device=device),
        torch.tensor([t_start], device=device),
    )
    x = alpha_bar_t.sqrt() * composite + noise_weight * (1 - alpha_bar_t).sqrt() * noise_init
    
    # Build timestep schedule
    if ddim_steps is not None and ddim_steps < t_start:
        # Evenly spaced sub-sequence ending at 0
        timesteps = np.linspace(t_start, 0, ddim_steps + 1, dtype=int)
        timesteps = list(timesteps)
    else:
        timesteps = list(range(t_start, -1, -1))
    
    with torch.no_grad():
        for i in range(len(timesteps) - 1):
            t = int(timesteps[i])
            t_prev = int(timesteps[i + 1])
            
            alpha_bar_cur = ddpm.alpha_bars[t].to(device)
            alpha_bar_prev = ddpm.alpha_bars[t_prev].to(device) if t_prev >= 0 else torch.tensor(1.0, device=device)
            
            time_tensor = torch.full((1, 1), t, device=device, dtype=torch.long)
            
            if mask_xt:
                indep_noise = torch.randn_like(x)
                x_for_model = x * missing_mask[:, :2] + indep_noise * known_mask[:, :2]
            else:
                x_for_model = x
            
            x_cond = torch.cat([x_for_model, mask_single, known_values], dim=1)
            x0_pred = ddpm.network(x_cond, time_tensor)
            
            # DDIM update: x_{t-1} = √ᾱ_{t-1} · x̂₀ + √(1-ᾱ_{t-1}-σ²) · ε_pred + σ · z
            # where ε_pred = (x_t - √ᾱ_t · x̂₀) / √(1-ᾱ_t)
            eps_pred = (x - alpha_bar_cur.sqrt() * x0_pred) / ((1 - alpha_bar_cur).sqrt() + 1e-8)
            
            if t_prev > 0:
                sigma = eta * ((1 - alpha_bar_prev) / (1 - alpha_bar_cur) * (1 - alpha_bar_cur / alpha_bar_prev)).sqrt()
                dir_xt = (1 - alpha_bar_prev - sigma ** 2).clamp(min=0).sqrt()
                
                if eta > 0:
                    z = noise_strategy(torch.zeros_like(x), torch.tensor([t_prev], device=device))
                else:
                    z = torch.zeros_like(x)
                
                x_denoised = alpha_bar_prev.sqrt() * x0_pred + dir_xt * eps_pred + noise_weight * sigma * z
                
                # Paste known region
                noise_known = noise_strategy(
                    torch.zeros_like(input_image),
                    torch.tensor([t_prev], device=device),
                )
                x_known = alpha_bar_prev.sqrt() * input_image + (1 - alpha_bar_prev).sqrt() * noise_known
                x = x_known * known_mask + x_denoised * missing_mask
            else:
                # Final step: use x0 prediction directly
                x = input_image * known_mask + x0_pred * missing_mask
    
    return x


# ═══════════════════════════════════════════════════════════════════════
#  MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════

def load_model(weights_path, device):
    """Load FiLM+Attn model with given weights."""
    network = MyUNet_FiLM_Attn(n_steps=N_STEPS, time_emb_dim=256, in_channels=5)
    ddpm = GaussianDDPM(
        network, n_steps=N_STEPS,
        min_beta=MIN_BETA, max_beta=MAX_BETA, device=device,
    )
    state = torch.load(weights_path, map_location=device, weights_only=False)
    ddpm.load_state_dict(state)
    ddpm = ddpm.to(device)
    ddpm.eval()
    return ddpm


# ═══════════════════════════════════════════════════════════════════════
#  MAIN SWEEP
# ═══════════════════════════════════════════════════════════════════════

def main():
    # Load eval set indices
    gpdiff_data = torch.load(GPDIFF_PT, map_location="cpu", weights_only=False)
    val_indices = gpdiff_data["val_indices"]
    eddy_set = set(gpdiff_data.get("eddy_indices", []))
    noeddy_set = set(gpdiff_data.get("noeddy_indices", []))
    n_total = len(val_indices)
    if args.quick > 0:
        n_total = min(args.quick, n_total)
    
    dd = DDInitializer()
    device = dd.get_device()
    standardizer = dd.get_standardizer()
    noise_strategy = get_noise_strategy(NOISE_FN)
    val_data = dd.get_validation_data()
    
    print(f"Device: {device}")
    print(f"Samples: {n_total} / {len(val_indices)}")
    print(f"Noise: {NOISE_FN}")
    
    # ═══════════════════════════════════════════════════════════════════
    #  Define experiment configurations
    # ═══════════════════════════════════════════════════════════════════
    
    # All configs: (name, weights_path, method, kwargs)
    configs = []
    
    # --- Experiment A: Weight comparison (S1 single-stage for speed) ---
    # A1: Current epoch 161 weights (baseline — matches previous eval)
    configs.append(("A1_epoch161_S1_t75", EPOCH161_WEIGHTS, "multistep_ddpm", {
        "t_start": 75, "noise_floor": 0.2, "gamma": 3.0, "resample_steps": 0,
        "stages": 1,
    }))
    # A2: Best EMA weights (epoch ~58)
    configs.append(("A2_bestEMA_S1_t75", BEST_EMA_WEIGHTS, "multistep_ddpm", {
        "t_start": 75, "noise_floor": 0.2, "gamma": 3.0, "resample_steps": 0,
        "stages": 1,
    }))
    # A3: Best raw weights (epoch ~58, no EMA)
    configs.append(("A3_bestRaw_S1_t75", BEST_WEIGHTS, "multistep_ddpm", {
        "t_start": 75, "noise_floor": 0.2, "gamma": 3.0, "resample_steps": 0,
        "stages": 1,
    }))
    
    # --- Experiment B: Low-t sweep (single-step x0 prediction, like CNN) ---
    for t_val in [1, 2, 5, 10, 25, 50]:
        configs.append((f"B_1step_t{t_val}", BEST_EMA_WEIGHTS, "single_step", {
            "t_val": t_val,
        }))
    
    # Few-step DDPM reverse from low t (GP warm-start, variance-adaptive)
    for t_start in [5, 10, 25]:
        configs.append((f"B_ddpm_t{t_start}_S1", BEST_EMA_WEIGHTS, "multistep_ddpm", {
            "t_start": t_start, "noise_floor": 0.2, "gamma": 3.0, "resample_steps": 0,
            "stages": 1,
        }))
    
    # --- Experiment C: DDIM deterministic (best EMA, GP warm-start) ---
    for ddim_steps in [5, 10, 25]:
        configs.append((f"C_ddim_s{ddim_steps}_t75", BEST_EMA_WEIGHTS, "ddim", {
            "t_start": 75, "ddim_steps": ddim_steps, "eta": 0.0,
            "noise_floor": 0.2, "gamma": 3.0,
        }))
    # DDIM from lower t_start
    configs.append(("C_ddim_s10_t25", BEST_EMA_WEIGHTS, "ddim", {
        "t_start": 25, "ddim_steps": 10, "eta": 0.0,
        "noise_floor": 0.2, "gamma": 3.0,
    }))
    # DDIM with partial stochasticity
    configs.append(("C_ddim_s10_t75_eta05", BEST_EMA_WEIGHTS, "ddim", {
        "t_start": 75, "ddim_steps": 10, "eta": 0.5,
        "noise_floor": 0.2, "gamma": 3.0,
    }))
    # Best raw weights with DDIM (combining best of A and C)
    configs.append(("AC_raw_ddim_s10_t75", BEST_WEIGHTS, "ddim", {
        "t_start": 75, "ddim_steps": 10, "eta": 0.0,
        "noise_floor": 0.2, "gamma": 3.0,
    }))
    # Best raw weights single-step
    configs.append(("AB_raw_1step_t1", BEST_WEIGHTS, "single_step", {
        "t_val": 1,
    }))
    configs.append(("AB_raw_1step_t10", BEST_WEIGHTS, "single_step", {
        "t_val": 10,
    }))
    
    print(f"\n{'='*100}")
    print(f"Running {len(configs)} configurations × {n_total} samples")
    print(f"{'='*100}")
    
    # ═══════════════════════════════════════════════════════════════════
    #  Pre-compute GP baselines for all samples
    # ═══════════════════════════════════════════════════════════════════
    print("\n--- Pre-computing GP baselines ---")
    sample_data = []
    for run_i in range(n_total):
        vi = val_indices[run_i]
        is_eddy = vi in eddy_set
        input_image = val_data[vi][0].unsqueeze(0).to(device)
        input_orig = standardizer.unstandardize(input_image.squeeze(0)).to(device).unsqueeze(0)
        
        land_mask = (input_orig.abs() > 1e-5).float().to(device)
        raw_mask = get_fixed_center_mask(input_image.shape).to(device)
        missing_mask = (raw_mask * land_mask[:, 0:1]).expand(-1, 2, -1, -1)
        
        gp_out, gp_var_map = gp_fill(
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
        gp_std = standardizer(gp_out.squeeze(0)).to(device).unsqueeze(0)
        
        sample_data.append({
            "vi": vi, "is_eddy": is_eddy,
            "input_image": input_image,  # standardized
            "input_orig": input_orig,    # physical
            "missing_mask": missing_mask,
            "gp_out": gp_out,
            "gp_std": gp_std,
            "gp_var_map": gp_var_map,
            "gp_mse": gp_mse.item(),
        })
        if (run_i + 1) % 10 == 0:
            print(f"  GP: {run_i+1}/{n_total}")
    
    gp_mses = [s["gp_mse"] for s in sample_data]
    print(f"  GP baseline: mean={np.mean(gp_mses):.6f}, median={np.median(gp_mses):.6f}")
    
    # ═══════════════════════════════════════════════════════════════════
    #  Run each configuration
    # ═══════════════════════════════════════════════════════════════════
    
    all_results = {}
    loaded_models = {}  # cache: weights_path → ddpm
    
    for ci, (name, weights_path, method, kwargs) in enumerate(configs):
        # Load model (cached)
        if weights_path not in loaded_models:
            wname = os.path.basename(weights_path)
            print(f"\n  Loading model: {wname}")
            loaded_models[weights_path] = load_model(weights_path, device)
        ddpm = loaded_models[weights_path]
        
        mses = []
        t0_config = time.time()
        
        for run_i, sd in enumerate(sample_data):
            seed = args.seed + run_i
            
            if method == "single_step":
                out_std = single_step_x0_predict(
                    ddpm, sd["input_image"], sd["missing_mask"],
                    sd["gp_std"], kwargs["t_val"],
                    noise_strategy, device, seed,
                )
            
            elif method == "multistep_ddpm":
                stages = kwargs.get("stages", 1)
                current_prior = sd["gp_std"].clone()
                current_var = sd["gp_var_map"].clone()
                
                for stage in range(1, stages + 1):
                    if stage == 1:
                        t_s = kwargs["t_start"]
                        nf = kwargs["noise_floor"]
                    else:
                        t_s = kwargs["t_refine"]
                        nf = kwargs["noise_floor_refine"]
                        current_var = current_var * kwargs["var_decay"]
                    
                    out_std = multistep_ddpm_x0(
                        ddpm, sd["input_image"], sd["missing_mask"],
                        current_prior, current_var,
                        t_s, noise_strategy, device,
                        seed + stage * 10000,
                        noise_floor=nf, gamma=kwargs["gamma"],
                        resample_steps=kwargs["resample_steps"],
                    )
                    current_prior = out_std.clone()
            
            elif method == "ddim":
                out_std = ddim_x0(
                    ddpm, sd["input_image"], sd["missing_mask"],
                    sd["gp_std"], sd["gp_var_map"],
                    kwargs["t_start"], noise_strategy, device, seed,
                    noise_floor=kwargs["noise_floor"], gamma=kwargs["gamma"],
                    ddim_steps=kwargs.get("ddim_steps"),
                    eta=kwargs.get("eta", 0.0),
                )
            
            elif method == "ddim_multistage":
                stages = kwargs.get("stages", 1)
                current_prior = sd["gp_std"].clone()
                current_var = sd["gp_var_map"].clone()
                
                for stage in range(1, stages + 1):
                    if stage == 1:
                        t_s = kwargs["t_start"]
                        nf = kwargs["noise_floor"]
                        ds = kwargs.get("ddim_steps", 25)
                    else:
                        t_s = kwargs["t_refine"]
                        nf = kwargs["noise_floor_refine"]
                        ds = kwargs.get("ddim_steps_refine", 15)
                        current_var = current_var * kwargs["var_decay"]
                    
                    out_std = ddim_x0(
                        ddpm, sd["input_image"], sd["missing_mask"],
                        current_prior, current_var,
                        t_s, noise_strategy, device,
                        seed + stage * 10000,
                        noise_floor=nf, gamma=kwargs["gamma"],
                        ddim_steps=ds, eta=kwargs.get("eta", 0.0),
                    )
                    current_prior = out_std.clone()
            
            # Convert to physical space and compute MSE
            film_phys = standardizer.unstandardize(out_std.squeeze(0)).to(device).unsqueeze(0)
            mse_val = ((film_phys - sd["input_orig"]) * sd["missing_mask"]).pow(2).sum() / (
                sd["missing_mask"].sum() + 1e-8)
            mses.append(mse_val.item())
        
        elapsed = time.time() - t0_config
        
        # Compute summary stats
        ratios = [m / (g + 1e-12) for m, g in zip(mses, gp_mses)]
        wins = sum(1 for m, g in zip(mses, gp_mses) if m < g)
        
        all_results[name] = {
            "mses": mses,
            "mean_mse": np.mean(mses),
            "median_mse": np.median(mses),
            "mean_ratio": np.mean(ratios),
            "median_ratio": np.median(ratios),
            "wins": wins,
            "n": n_total,
            "time": elapsed,
            "method": method,
            "kwargs": kwargs,
            "weights": os.path.basename(weights_path),
        }
        
        print(f"[{ci+1:>2}/{len(configs)}] {name:35s}  "
              f"MSE={np.mean(mses):.6f}  ratio={np.mean(ratios):.3f}x  "
              f"wins={wins}/{n_total}  ({elapsed:.1f}s)")
    
    # ═══════════════════════════════════════════════════════════════════
    #  Print summary table
    # ═══════════════════════════════════════════════════════════════════
    
    print(f"\n{'='*120}")
    print(f"SUMMARY TABLE — {n_total} samples, GP baseline MSE = {np.mean(gp_mses):.6f}")
    print(f"{'='*120}")
    print(f"{'Config':35s}  {'Weights':20s}  {'Mean MSE':>10s}  {'Med MSE':>10s}  "
          f"{'Ratio':>7s}  {'Med Ratio':>9s}  {'Wins':>8s}  {'Time':>7s}")
    print(f"{'-'*120}")
    
    # Sort by mean MSE
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]["mean_mse"])
    
    for name, r in sorted_results:
        print(f"{name:35s}  {r['weights']:20s}  {r['mean_mse']:>10.6f}  "
              f"{r['median_mse']:>10.6f}  {r['mean_ratio']:>6.3f}x  "
              f"{r['median_ratio']:>8.3f}x  {r['wins']:>3d}/{r['n']:<4d}  "
              f"{r['time']:>6.1f}s")
    
    # Print experiment-group summaries
    print(f"\n{'='*120}")
    print("EXPERIMENT GROUP SUMMARIES")
    print(f"{'='*120}")
    
    print("\n--- A: Weight Selection ---")
    for name, r in sorted_results:
        if name.startswith("A"):
            print(f"  {name:35s}  MSE={r['mean_mse']:.6f}  ratio={r['mean_ratio']:.3f}x  wins={r['wins']}/{r['n']}")
    
    print("\n--- B: Low-t Sweep (single-step & few-step) ---")
    for name, r in sorted_results:
        if name.startswith("B"):
            print(f"  {name:35s}  MSE={r['mean_mse']:.6f}  ratio={r['mean_ratio']:.3f}x  wins={r['wins']}/{r['n']}")
    
    print("\n--- C: DDIM Deterministic ---")
    for name, r in sorted_results:
        if name.startswith("C"):
            print(f"  {name:35s}  MSE={r['mean_mse']:.6f}  ratio={r['mean_ratio']:.3f}x  wins={r['wins']}/{r['n']}")
    
    # Voronoi-CNN reference
    print(f"\n--- Reference ---")
    print(f"  {'GP baseline':35s}  MSE={np.mean(gp_mses):.6f}  ratio=1.000x")
    print(f"  {'Voronoi-CNN (paper)':35s}  MSE=0.001400  ratio=0.269x")
    print(f"  {'GP-Diff uncond (paper)':35s}  MSE=0.004370  ratio=0.837x")
    
    # ── Save results ──────────────────────────────────────────────────
    save_dict = {
        "configs": {name: {k: v for k, v in r.items() if k != "mses"}
                    for name, r in all_results.items()},
        "per_sample": {name: r["mses"] for name, r in all_results.items()},
        "gp_mses": gp_mses,
        "n_samples": n_total,
        "sample_indices": [sd["vi"] for sd in sample_data],
    }
    pt_path = os.path.join(OUT_DIR, f"abc_sweep_{n_total}samples.pt")
    torch.save(save_dict, pt_path)
    print(f"\nResults saved to: {pt_path}")


if __name__ == "__main__":
    main()
