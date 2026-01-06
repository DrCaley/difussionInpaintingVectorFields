"""
Diagnostic script to track vector magnitudes at each denoising step.
Compares Gaussian vs Div-Free noise to see where vectors diverge.
"""

import sys
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from data_prep.data_initializer import DDInitializer
from ddpm.neural_networks.ddpm import GaussianDDPM
from ddpm.neural_networks.unets.unet_xl import MyUNet
from ddpm.helper_functions.masks.straigth_line import StraightLineMaskGenerator


def diagnose_denoising(noise_type='gaussian'):
    """Run denoising and track statistics at each step."""
    
    # Load config
    dd = DDInitializer()
    device = dd.get_device()
    
    # Select model based on noise type
    if noise_type == 'gaussian':
        model_path = BASE_DIR / "ddpm/Trained_Models/weekend_ddpm_ocean_model.pt"
    else:
        model_path = BASE_DIR / "ddpm/Trained_Models/div_free_model.pt"
    
    print(f"\n{'='*60}")
    print(f"Diagnosing {noise_type.upper()} noise denoising")
    print(f"Model: {model_path.name}")
    print(f"{'='*60}\n")
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model_state_dict = checkpoint.get('model_state_dict', checkpoint)
    n_steps = checkpoint.get('n_steps', 100)
    
    model = GaussianDDPM(MyUNet(n_steps), n_steps=n_steps, device=device)
    model.load_state_dict(model_state_dict)
    model.eval()
    
    # Reinitialize dd with model params
    min_beta = checkpoint.get('min_beta', 0.0001)
    max_beta = checkpoint.get('max_beta', 0.02)
    dd.reinitialize(min_beta, max_beta, n_steps, dd.get_standardizer())
    
    # Get noise strategy
    if noise_type == 'gaussian':
        from ddpm.utils.noise_utils import GaussianNoise
        noise_strat = GaussianNoise()
    else:
        noise_strat = dd.get_noise_strategy()
    
    # Load a sample
    val_loader = DataLoader(dd.get_validation_data(), batch_size=1, shuffle=False)
    batch = next(iter(val_loader))
    input_image = batch[0].to(device)
    
    # Unstandardize for proper comparison
    standardizer = dd.get_standardizer()
    input_original = standardizer.unstandardize(input_image.squeeze(0)).unsqueeze(0).to(device)
    
    # Create mask
    mask_gen = StraightLineMaskGenerator(1, 1)
    land_mask = (input_original.abs() > 1e-5).float().to(device)
    raw_mask = mask_gen.generate_mask(input_image.shape).to(device)
    mask = raw_mask * land_mask
    
    print(f"Input image - mean: {input_image.mean():.4f}, std: {input_image.std():.4f}")
    print(f"Input original - mean: {input_original.mean():.4f}, std: {input_original.std():.4f}")
    print(f"Mask coverage: {mask.mean():.2%}\n")
    
    # Forward noising - track stats
    print("FORWARD NOISING (adding noise):")
    print("-" * 50)
    noised_images = [None] * (n_steps + 1)
    noised_images[0] = input_image.clone()
    
    for t in range(n_steps):
        epsilon = noise_strat(noised_images[t], None)
        noised_images[t + 1] = model(noised_images[t], t, epsilon, one_step=True)
        
        if t % 20 == 0 or t == n_steps - 1:
            img = noised_images[t + 1]
            masked_vals = img * mask
            print(f"  t={t:3d}: mean={img.mean():8.4f}, std={img.std():8.4f}, "
                  f"min={img.min():8.4f}, max={img.max():8.4f}, "
                  f"masked_mean={masked_vals.sum()/mask.sum():8.4f}")
    
    # Initial state for denoising
    noise = noise_strat(input_image, torch.tensor([n_steps], device=device))
    x = noised_images[n_steps] * (1 - mask) + (noise * mask)
    
    print(f"\nInitial x (before denoising):")
    print(f"  mean={x.mean():8.4f}, std={x.std():8.4f}, min={x.min():8.4f}, max={x.max():8.4f}")
    
    # Denoising - track stats
    print(f"\nBACKWARD DENOISING (removing noise):")
    print("-" * 50)
    
    stats = []
    
    for t in range(n_steps - 1, -1, -1):
        # Denoise one step
        time_tensor = torch.full((1, 1), t, device=device, dtype=torch.long)
        epsilon_theta = model.backward(x, time_tensor)
        
        alpha_t = model.alphas[t].to(device)
        alpha_t_bar = model.alpha_bars[t].to(device)
        
        if noise_strat.get_gaussian_scaling():
            less_noised = (1 / alpha_t.sqrt()) * (
                x - ((1 - alpha_t) / (1 - alpha_t_bar).sqrt()) * epsilon_theta
            )
        else:
            less_noised = (1 / alpha_t.sqrt()) * (x - epsilon_theta)
        
        # Add noise back if not final step
        if t > 0:
            z = noise_strat(torch.zeros_like(x), torch.tensor([t], device=device))
            beta_t = model.betas[t].to(device)
            sigma_t = beta_t.sqrt()
            less_noised = less_noised + sigma_t * z
        
        # Combine with known values
        known = noised_images[t]
        inpainted = less_noised
        x = known * (1 - mask) + (inpainted * mask)
        
        # Track stats
        masked_inpainted = inpainted * mask
        masked_sum = mask.sum()
        
        stat = {
            't': t,
            'epsilon_mean': epsilon_theta.mean().item(),
            'epsilon_std': epsilon_theta.std().item(),
            'epsilon_max': epsilon_theta.abs().max().item(),
            'inpainted_mean': (masked_inpainted.sum() / masked_sum).item() if masked_sum > 0 else 0,
            'inpainted_std': inpainted.std().item(),
            'inpainted_max': inpainted.abs().max().item(),
            'x_mean': x.mean().item(),
            'x_std': x.std().item(),
        }
        stats.append(stat)
        
        if t % 20 == 0 or t == 0:
            print(f"  t={t:3d}: eps_mean={stat['epsilon_mean']:8.4f}, eps_std={stat['epsilon_std']:8.4f}, "
                  f"eps_max={stat['epsilon_max']:8.4f} | "
                  f"inp_mean={stat['inpainted_mean']:8.4f}, inp_max={stat['inpainted_max']:8.4f}")
    
    # Final result
    result = input_image * (1 - mask) + x * mask
    result_unstd = standardizer.unstandardize(result.squeeze(0)).unsqueeze(0)
    
    print(f"\nFINAL RESULT:")
    print(f"  Standardized   - mean={result.mean():8.4f}, std={result.std():8.4f}")
    print(f"  Unstandardized - mean={result_unstd.mean():8.4f}, std={result_unstd.std():8.4f}")
    print(f"  Ground truth   - mean={input_original.mean():8.4f}, std={input_original.std():8.4f}")
    
    # Check for explosion
    final_inpainted_region = (result * mask).sum() / mask.sum()
    ground_truth_region = (input_image * mask).sum() / mask.sum()
    print(f"\n  Masked region mean - Predicted: {final_inpainted_region:.4f}, Ground truth: {ground_truth_region:.4f}")
    
    return stats


if __name__ == '__main__':
    print("\n" + "="*70)
    print("DENOISING DIAGNOSTICS")
    print("="*70)
    
    # Run for both noise types
    gaussian_stats = diagnose_denoising('gaussian')
    divfree_stats = diagnose_denoising('div_free')
    
    # Compare epsilon statistics
    print("\n" + "="*70)
    print("COMPARISON: Epsilon (predicted noise) statistics")
    print("="*70)
    print(f"{'Step':>5} | {'Gaussian eps_std':>16} | {'DivFree eps_std':>16} | {'Ratio':>8}")
    print("-" * 55)
    
    for g, d in zip(gaussian_stats[::20], divfree_stats[::20]):
        ratio = d['epsilon_std'] / g['epsilon_std'] if g['epsilon_std'] > 0 else float('inf')
        print(f"{g['t']:5d} | {g['epsilon_std']:16.4f} | {d['epsilon_std']:16.4f} | {ratio:8.2f}x")
