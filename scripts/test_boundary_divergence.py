#!/usr/bin/env python3
"""
Test script to verify if naive stitching creates high divergence at boundaries.
Tests the theory that hard mask combining breaks div-free spatial correlation.
"""

import torch
import numpy as np
import sys
sys.path.insert(0, '.')

from ddpm.neural_networks.ddpm import GaussianDDPM
from ddpm.utils.noise_utils import DivergenceFreeNoise, GaussianNoise
from ddpm.helper_functions.compute_divergence import compute_divergence
from data_prep.ocean_image_dataset import OceanImageDataset
from ddpm.helper_functions.masks.straigth_line import StraightLineMaskGenerator
from torch.nn.functional import conv2d
import yaml

def analyze_boundary_divergence(noise_type='div_free'):
    """Analyze divergence at boundaries after naive stitching."""
    
    # Setup - update yaml and reimport
    with open('data.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Temporarily modify noise_function
    original_noise = config.get('noise_function', 'gaussian')
    config['noise_function'] = noise_type
    
    # Import DDInitializer after modifying yaml (singleton needs reset)
    from data_prep.data_initializer import DDInitializer
    DDInitializer._instance = None  # Reset singleton
    
    # Write temp config
    with open('data_temp.yaml', 'w') as f:
        yaml.dump(config, f)
    
    dd = DDInitializer('data_temp.yaml')
    device = torch.device('cpu')
    
    # Load model
    if noise_type == 'div_free':
        model_path = 'ddpm/Trained_Models/div_free_model.pt'
    else:
        model_path = 'ddpm/Trained_Models/weekend_ddpm_ocean_model.pt'
    
    ddpm = GaussianDDPM(dd, device)
    ddpm.load_model(model_path)
    
    # Get noise strategy
    if noise_type == 'div_free':
        noise_strat = DivergenceFreeNoise(dd)
    else:
        noise_strat = GaussianNoise(dd)
    
    # Get a sample - use the standardizer from dd
    dataset = OceanImageDataset('data/rams_head', device, dd)
    sample = dataset[5]
    u_raw, v_raw = sample['u'], sample['v']
    land_mask = sample['land_mask']
    
    # Standardize using dd's standardizer
    tensor = torch.stack([u_raw, v_raw], dim=0)
    standardizer = dd.get_standardizer()
    tensor_std = standardizer(tensor)
    original = tensor_std.unsqueeze(0)  # Add batch dim
    
    # Create mask
    mask_gen = StraightLineMaskGenerator(1)
    raw_mask = mask_gen(original.shape[-2:]).unsqueeze(0).unsqueeze(0)
    mask = raw_mask * land_mask.unsqueeze(0).unsqueeze(0)
    
    print(f'=== {noise_type.upper()} Noise: Divergence Analysis at t=80 ===')
    print()
    
    # Noise the image at t=80
    t = torch.tensor([80])
    noised, noise = ddpm.forward_noise(original, t, noise_strat)
    
    # 1. Original image divergence
    div_orig = compute_divergence(original[0, 0], original[0, 1])
    print(f'Original image mean |div|: {div_orig.abs().mean():.4f}')
    
    # 2. Noised image divergence
    div_noised = compute_divergence(noised[0, 0], noised[0, 1])
    print(f'Noised image mean |div|: {div_noised.abs().mean():.4f}')
    
    # 3. Denoise one step
    alpha_t = ddpm.alpha[t]
    with torch.no_grad():
        epsilon_theta = ddpm.model(noised, t.float())
    
    # Use the denoising formula
    if noise_strat.get_gaussian_scaling():
        alpha_t_bar = ddpm.alpha_bar[t]
        less_noised = (1 / alpha_t.sqrt()) * (
            noised - ((1 - alpha_t) / (1 - alpha_t_bar).sqrt()) * epsilon_theta
        )
    else:
        less_noised = (1 / alpha_t.sqrt()) * (noised - epsilon_theta)
    
    div_denoised = compute_divergence(less_noised[0, 0], less_noised[0, 1])
    print(f'Denoised (inpainted) mean |div|: {div_denoised.abs().mean():.4f}')
    
    # 4. Known region at t-1
    t_prev = torch.tensor([79])
    known_noised, _ = ddpm.forward_noise(original, t_prev, noise_strat)
    div_known = compute_divergence(known_noised[0, 0], known_noised[0, 1])
    print(f'Known region mean |div|: {div_known.abs().mean():.4f}')
    
    # 5. NAIVE COMBINATION
    mask_2d = mask[0, 0]
    combined_u = known_noised[0, 0] * (1 - mask_2d) + less_noised[0, 0] * mask_2d
    combined_v = known_noised[0, 1] * (1 - mask_2d) + less_noised[0, 1] * mask_2d
    div_combined = compute_divergence(combined_u, combined_v)
    print(f'After naive stitch mean |div|: {div_combined.abs().mean():.4f}')
    
    # 6. Boundary analysis
    dilate_kernel = torch.ones(1, 1, 3, 3)
    mask_expanded = mask_2d.unsqueeze(0).unsqueeze(0)
    dilated = conv2d(mask_expanded.float(), dilate_kernel, padding=1).clamp(0, 1)
    boundary = (dilated - mask_expanded.float()).squeeze()
    boundary_mask = boundary > 0
    
    if boundary_mask.sum() > 0:
        div_at_boundary = div_combined[boundary_mask].abs().mean()
        div_away = div_combined[~boundary_mask].abs().mean()
        
        print()
        print(f'=== BOUNDARY ANALYSIS ===')
        print(f'Boundary pixels: {boundary_mask.sum().item()}')
        print(f'Mean |div| AT boundary: {div_at_boundary:.4f}')
        print(f'Mean |div| AWAY from boundary: {div_away:.4f}')
        print(f'Ratio (boundary/away): {div_at_boundary/div_away:.2f}x')
        
        # Value discontinuity
        grad_u = torch.abs(combined_u[:, :-1] - combined_u[:, 1:])
        grad_v = torch.abs(combined_v[:, :-1] - combined_v[:, 1:])
        
        grad_full = torch.zeros_like(combined_u)
        grad_full[:, :-1] = grad_u + grad_v
        
        grad_at_boundary = grad_full[boundary_mask].mean()
        grad_away = grad_full[~boundary_mask].mean()
        
        print()
        print(f'=== VALUE DISCONTINUITY ===')
        print(f'Mean |gradient| AT boundary: {grad_at_boundary:.4f}')
        print(f'Mean |gradient| AWAY from boundary: {grad_away:.4f}')
        print(f'Ratio (boundary/away): {grad_at_boundary/grad_away:.2f}x')
    
    return {
        'div_at_boundary': div_at_boundary.item() if boundary_mask.sum() > 0 else 0,
        'div_away': div_away.item() if boundary_mask.sum() > 0 else 0,
        'grad_at_boundary': grad_at_boundary.item() if boundary_mask.sum() > 0 else 0,
        'grad_away': grad_away.item() if boundary_mask.sum() > 0 else 0,
    }

if __name__ == '__main__':
    print('='*60)
    div_free_results = analyze_boundary_divergence('div_free')
    
    print()
    print('='*60)
    gaussian_results = analyze_boundary_divergence('gaussian')
    
    print()
    print('='*60)
    print('=== COMPARISON SUMMARY ===')
    print()
    print(f'Metric                    | Div-Free | Gaussian | Ratio')
    print(f'--------------------------|----------|----------|------')
    print(f'Div AT boundary           | {div_free_results["div_at_boundary"]:.4f}   | {gaussian_results["div_at_boundary"]:.4f}   | {div_free_results["div_at_boundary"]/gaussian_results["div_at_boundary"]:.2f}x')
    print(f'Div AWAY from boundary    | {div_free_results["div_away"]:.4f}   | {gaussian_results["div_away"]:.4f}   | {div_free_results["div_away"]/gaussian_results["div_away"]:.2f}x')
    print(f'Gradient AT boundary      | {div_free_results["grad_at_boundary"]:.4f}   | {gaussian_results["grad_at_boundary"]:.4f}   | {div_free_results["grad_at_boundary"]/gaussian_results["grad_at_boundary"]:.2f}x')
    print(f'Gradient AWAY from bndry  | {div_free_results["grad_away"]:.4f}   | {gaussian_results["grad_away"]:.4f}   | {div_free_results["grad_away"]/gaussian_results["grad_away"]:.2f}x')
