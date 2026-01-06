#!/usr/bin/env python3
"""
Test actual denoising + inpainting to see divergence at boundaries.
This uses the real model and inpainting process.
"""

import torch
import numpy as np
import sys
sys.path.insert(0, '.')

from data_prep.data_initializer import DDInitializer
from ddpm.neural_networks.ddpm import GaussianDDPM
from ddpm.neural_networks.unets.unet_xl import MyUNet
from ddpm.utils.noise_utils import DivergenceFreeNoise, GaussianNoise
from ddpm.helper_functions.compute_divergence import compute_divergence
from data_prep.ocean_image_dataset import OceanImageDataset
from ddpm.helper_functions.masks.straigth_line import StraightLineMaskGenerator
from torch.nn.functional import conv2d

# Initialize
dd = DDInitializer('data.yaml')
device = torch.device('cpu')
n_steps = dd.get_attribute('noise_steps')

# Load both models
print("Loading div-free model...")
ddpm_divfree = GaussianDDPM(MyUNet(n_steps), n_steps=n_steps, device=device)
checkpoint = torch.load('ddpm/Trained_Models/div_free_model.pt', map_location=device, weights_only=False)
ddpm_divfree.load_state_dict(checkpoint['model_state_dict'])
ddpm_divfree.eval()

print("Loading gaussian model...")
ddpm_gaussian = GaussianDDPM(MyUNet(n_steps), n_steps=n_steps, device=device)
checkpoint = torch.load('ddpm/Trained_Models/weekend_ddpm_ocean_model.pt', map_location=device, weights_only=False)
ddpm_gaussian.load_state_dict(checkpoint['model_state_dict'])
ddpm_gaussian.eval()

# Create noise strategies
noise_divfree = DivergenceFreeNoise()
noise_gaussian = GaussianNoise()

# Get sample from dd's test data
dataset = dd.test_data
x0, _, _ = dataset[5]  # Returns (tensor, t, noise)
# x0 is [2, H, W] with u, v
standardizer = dd.get_standardizer()
original = standardizer(x0).unsqueeze(0)  # [1, 2, H, W]
# Land mask: where data is non-zero
land_mask = (x0.abs().sum(dim=0) > 1e-5).float()  # [H, W]

# Create mask (mask is 1 for inpainted, 0 for known)
mask_gen = StraightLineMaskGenerator(1)
raw_mask = mask_gen.generate_mask(original.shape)  # [1, 1, H, W]
mask = raw_mask * land_mask.unsqueeze(0).unsqueeze(0)

# Compute boundary
kernel = torch.ones(1, 1, 3, 3)
dilated = conv2d(mask.float(), kernel, padding=1).clamp(0, 1)
boundary = ((dilated - mask.float()) > 0).squeeze()

print(f"\nMask coverage: {mask.mean()*100:.1f}%")
print(f"Boundary pixels: {boundary.sum().item()}")

def forward_noise(ddpm, x0, t, noise_strat):
    """Forward noise using specific noise strategy."""
    n, c, h, w = x0.shape
    a_bar = ddpm.alpha_bars[t]
    
    # Generate noise using strategy
    epsilon = noise_strat.generate((n, c, h, w), t, device)
    
    # Add noise
    if noise_strat.get_gaussian_scaling():
        noisy = a_bar.sqrt().reshape(n, 1, 1, 1) * x0 + (1 - a_bar).sqrt().reshape(n, 1, 1, 1) * epsilon
    else:
        noisy = a_bar.sqrt().reshape(n, 1, 1, 1) * x0 + epsilon
    
    return noisy, epsilon

def run_one_denoising_step(ddpm, noise_strat, original, mask, t_val):
    """Run one denoising step and measure divergence."""
    
    t = torch.tensor([t_val])
    
    # Forward noise
    noised, noise = forward_noise(ddpm, original, t, noise_strat)
    
    # Predict epsilon
    with torch.no_grad():
        epsilon_theta = ddpm.network(noised, t)  # t should be int tensor
    
    # Denoise based on noise type
    alpha_t = ddpm.alphas[t]
    alpha_t_bar = ddpm.alpha_bars[t]
    
    if noise_strat.get_gaussian_scaling():
        less_noised = (1 / alpha_t.sqrt()) * (
            noised - ((1 - alpha_t) / (1 - alpha_t_bar).sqrt()) * epsilon_theta
        )
    else:
        less_noised = (1 / alpha_t.sqrt()) * (noised - epsilon_theta)
    
    # Forward noise original at t-1 for known region
    t_prev = t_val - 1
    if t_prev < 0:
        known = original
    else:
        known, _ = forward_noise(ddpm, original, torch.tensor([t_prev]), noise_strat)
    
    # NAIVE STITCH
    combined = known * (1 - mask) + less_noised * mask
    
    # Compute divergences
    div_known = compute_divergence(known[0, 0], known[0, 1])
    div_inpainted = compute_divergence(less_noised[0, 0], less_noised[0, 1])
    div_combined = compute_divergence(combined[0, 0], combined[0, 1])
    
    return {
        'div_known': div_known.abs().mean().item(),
        'div_inpainted': div_inpainted.abs().mean().item(),
        'div_combined': div_combined.abs().mean().item(),
        'div_at_boundary': div_combined[boundary].abs().mean().item(),
        'div_away': div_combined[~boundary].abs().mean().item(),
    }

# Test at multiple timesteps
timesteps = [80, 60, 40, 20, 5]

print("\n" + "="*70)
print("=== DIVERGENCE AFTER DENOISING + NAIVE STITCH ===")
print("="*70)

print("\n--- DIV-FREE MODEL + DIV-FREE NOISE ---")
print(f"{'t':>5} | {'div_known':>10} | {'div_inp':>10} | {'div_comb':>10} | {'at_bnd':>10} | {'away':>10} | {'ratio':>6}")
print("-"*70)
for t in timesteps:
    r = run_one_denoising_step(ddpm_divfree, noise_divfree, original, mask, t)
    ratio = r['div_at_boundary'] / r['div_away']
    print(f"{t:>5} | {r['div_known']:>10.4f} | {r['div_inpainted']:>10.4f} | {r['div_combined']:>10.4f} | {r['div_at_boundary']:>10.4f} | {r['div_away']:>10.4f} | {ratio:>6.2f}x")

print("\n--- GAUSSIAN MODEL + GAUSSIAN NOISE ---")
print(f"{'t':>5} | {'div_known':>10} | {'div_inp':>10} | {'div_comb':>10} | {'at_bnd':>10} | {'away':>10} | {'ratio':>6}")
print("-"*70)
for t in timesteps:
    r = run_one_denoising_step(ddpm_gaussian, noise_gaussian, original, mask, t)
    ratio = r['div_at_boundary'] / r['div_away']
    print(f"{t:>5} | {r['div_known']:>10.4f} | {r['div_inpainted']:>10.4f} | {r['div_combined']:>10.4f} | {r['div_at_boundary']:>10.4f} | {r['div_away']:>10.4f} | {ratio:>6.2f}x")

# Summary
print("\n" + "="*70)
print("=== SUMMARY: Average boundary/away ratio ===")
print("="*70)

divfree_ratios = []
for t in timesteps:
    r = run_one_denoising_step(ddpm_divfree, noise_divfree, original, mask, t)
    divfree_ratios.append(r['div_at_boundary'] / r['div_away'])

gaussian_ratios = []
for t in timesteps:
    r = run_one_denoising_step(ddpm_gaussian, noise_gaussian, original, mask, t)
    gaussian_ratios.append(r['div_at_boundary'] / r['div_away'])

print(f"\nDiv-Free avg boundary/away ratio: {np.mean(divfree_ratios):.2f}x")
print(f"Gaussian avg boundary/away ratio: {np.mean(gaussian_ratios):.2f}x")
print(f"\nDiv-Free creates {np.mean(divfree_ratios)/np.mean(gaussian_ratios):.2f}x more boundary divergence than Gaussian")
