#!/usr/bin/env python3
"""
Test script to verify if naive stitching creates high divergence at boundaries.
Tests the theory that hard mask combining breaks div-free spatial correlation.

This script compares Gaussian vs Div-Free noise behavior at boundaries.
"""

import torch
import numpy as np
import sys
sys.path.insert(0, '.')

from ddpm.helper_functions.compute_divergence import compute_divergence
from torch.nn.functional import conv2d

def generate_div_free_noise(shape):
    """Generate divergence-free noise via stream function."""
    H, W = shape[-2:]
    # Random stream function
    psi = torch.randn(1, 1, H, W)
    
    # Compute u = dpsi/dy, v = -dpsi/dx (central differences)
    dy = psi[:, :, 2:, 1:-1] - psi[:, :, :-2, 1:-1]
    dx = psi[:, :, 1:-1, 2:] - psi[:, :, 1:-1, :-2]
    
    u = torch.zeros(1, 1, H, W)
    v = torch.zeros(1, 1, H, W)
    u[:, :, 1:-1, 1:-1] = dy / 2.0
    v[:, :, 1:-1, 1:-1] = -dx / 2.0
    
    # Normalize to unit variance
    noise = torch.cat([u, v], dim=1)
    noise = noise / (noise.std() + 1e-8)
    return noise

def generate_gaussian_noise(shape):
    """Generate standard Gaussian noise."""
    return torch.randn(shape)

def create_boundary_mask(H, W):
    """Create a mask and its boundary."""
    mask = torch.zeros(H, W)
    mask[20:40, :] = 1.0  # Horizontal strip
    
    # Compute boundary (dilation - mask)
    kernel = torch.ones(1, 1, 3, 3)
    mask_4d = mask.unsqueeze(0).unsqueeze(0)
    dilated = conv2d(mask_4d, kernel, padding=1).clamp(0, 1).squeeze()
    boundary = (dilated - mask) > 0
    
    return mask, boundary

def test_naive_stitch(noise_type='gaussian'):
    """Test divergence after naive stitching."""
    
    H, W = 64, 128
    shape = (1, 2, H, W)
    
    # Generate two independent noise samples (simulating known and inpainted regions)
    if noise_type == 'div_free':
        known_noise = generate_div_free_noise(shape)
        inpainted_noise = generate_div_free_noise(shape)
    else:
        known_noise = generate_gaussian_noise(shape)
        inpainted_noise = generate_gaussian_noise(shape)
    
    # Create mask
    mask, boundary = create_boundary_mask(H, W)
    mask_4d = mask.unsqueeze(0).unsqueeze(0)
    
    # NAIVE STITCH
    combined = known_noise * (1 - mask_4d) + inpainted_noise * mask_4d
    
    # Compute divergence
    div_known = compute_divergence(known_noise[0, 0], known_noise[0, 1])
    div_inpainted = compute_divergence(inpainted_noise[0, 0], inpainted_noise[0, 1])
    div_combined = compute_divergence(combined[0, 0], combined[0, 1])
    
    # Compute at boundary vs away from boundary
    div_at_boundary = div_combined[boundary].abs().mean().item()
    div_away = div_combined[~boundary].abs().mean().item()
    
    # Compute gradient discontinuity
    grad_u_x = torch.abs(combined[0, 0, :, :-1] - combined[0, 0, :, 1:])
    grad_v_x = torch.abs(combined[0, 1, :, :-1] - combined[0, 1, :, 1:])
    grad_total = torch.zeros(H, W)
    grad_total[:, :-1] = grad_u_x + grad_v_x
    
    grad_at_boundary = grad_total[boundary].mean().item()
    grad_away = grad_total[~boundary].mean().item()
    
    print(f"\n=== {noise_type.upper()} Noise Analysis ===")
    print(f"Known noise mean |div|: {div_known.abs().mean():.4f}")
    print(f"Inpainted noise mean |div|: {div_inpainted.abs().mean():.4f}")
    print(f"Combined (naive stitch) mean |div|: {div_combined.abs().mean():.4f}")
    print()
    print(f"=== BOUNDARY ANALYSIS ===")
    print(f"Boundary pixels: {boundary.sum().item()}")
    print(f"Mean |div| AT boundary: {div_at_boundary:.4f}")
    print(f"Mean |div| AWAY from boundary: {div_away:.4f}")
    print(f"Ratio (boundary/away): {div_at_boundary/div_away:.2f}x")
    print()
    print(f"=== GRADIENT DISCONTINUITY ===")
    print(f"Mean |grad| AT boundary: {grad_at_boundary:.4f}")
    print(f"Mean |grad| AWAY from boundary: {grad_away:.4f}")
    print(f"Ratio (boundary/away): {grad_at_boundary/grad_away:.2f}x")
    
    return {
        'div_at_boundary': div_at_boundary,
        'div_away': div_away,
        'grad_at_boundary': grad_at_boundary,
        'grad_away': grad_away,
        'div_known': div_known.abs().mean().item(),
        'div_inpainted': div_inpainted.abs().mean().item(),
        'div_combined': div_combined.abs().mean().item(),
    }

if __name__ == '__main__':
    print("="*60)
    print("THEORY: Naive stitching creates boundary discontinuities")
    print("that break div-free spatial correlation but don't")
    print("significantly affect Gaussian (i.i.d.) noise.")
    print("="*60)
    
    # Run multiple times to get stable averages
    n_trials = 10
    
    gaussian_results = []
    divfree_results = []
    
    print(f"\nRunning {n_trials} trials...")
    for i in range(n_trials):
        gaussian_results.append(test_naive_stitch('gaussian'))
        divfree_results.append(test_naive_stitch('div_free'))
    
    # Compute averages
    def avg(results, key):
        return np.mean([r[key] for r in results])
    
    print("\n" + "="*60)
    print("=== SUMMARY (averaged over {} trials) ===".format(n_trials))
    print("="*60)
    print()
    print(f"{'Metric':<30} | {'Gaussian':>10} | {'Div-Free':>10} | {'Ratio':>8}")
    print("-"*70)
    
    # Divergence at boundary
    g_div_bnd = avg(gaussian_results, 'div_at_boundary')
    d_div_bnd = avg(divfree_results, 'div_at_boundary')
    print(f"{'Div AT boundary':<30} | {g_div_bnd:>10.4f} | {d_div_bnd:>10.4f} | {d_div_bnd/g_div_bnd:>7.2f}x")
    
    # Divergence away from boundary
    g_div_away = avg(gaussian_results, 'div_away')
    d_div_away = avg(divfree_results, 'div_away')
    print(f"{'Div AWAY from boundary':<30} | {g_div_away:>10.4f} | {d_div_away:>10.4f} | {d_div_away/g_div_away:>7.2f}x")
    
    # Boundary/Away ratio
    g_ratio = g_div_bnd / g_div_away
    d_ratio = d_div_bnd / d_div_away
    print(f"{'Boundary/Away ratio':<30} | {g_ratio:>10.2f}x | {d_ratio:>10.2f}x | {d_ratio/g_ratio:>7.2f}x")
    
    # Gradient at boundary
    g_grad_bnd = avg(gaussian_results, 'grad_at_boundary')
    d_grad_bnd = avg(divfree_results, 'grad_at_boundary')
    print(f"{'Gradient AT boundary':<30} | {g_grad_bnd:>10.4f} | {d_grad_bnd:>10.4f} | {d_grad_bnd/g_grad_bnd:>7.2f}x")
    
    # Gradient away
    g_grad_away = avg(gaussian_results, 'grad_away')
    d_grad_away = avg(divfree_results, 'grad_away')
    print(f"{'Gradient AWAY from boundary':<30} | {g_grad_away:>10.4f} | {d_grad_away:>10.4f} | {d_grad_away/g_grad_away:>7.2f}x")
    
    # Gradient ratio
    g_grad_ratio = g_grad_bnd / g_grad_away
    d_grad_ratio = d_grad_bnd / d_grad_away
    print(f"{'Gradient bnd/away ratio':<30} | {g_grad_ratio:>10.2f}x | {d_grad_ratio:>10.2f}x | {d_grad_ratio/g_grad_ratio:>7.2f}x")
    
    print()
    print("="*60)
    print("CONCLUSION:")
    if d_ratio > g_ratio * 1.5:
        print("✓ CONFIRMED: Div-free noise shows HIGHER boundary divergence")
        print("  relative to non-boundary areas compared to Gaussian.")
        print("  This supports the theory that naive stitching breaks")
        print("  the spatial correlation structure of div-free noise.")
    else:
        print("✗ NOT CONFIRMED: Boundary divergence ratios are similar.")
    print("="*60)
