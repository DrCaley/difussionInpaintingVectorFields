#!/usr/bin/env python
"""
Side-by-side comparison of Gaussian vs Div-free inpainting using ModelInpainter.
"""
import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from ddpm.helper_functions.compute_divergence import compute_divergence
from ddpm.Testing.model_inpainter import ModelInpainter, StraightLineMaskGenerator
from ddpm.utils.inpainting_utils import inpaint_generate_new_images, calculate_mse, top_left_crop
from ddpm.utils.noise_utils import GaussianNoise


def compute_all_metrics(pred, true, mask, device):
    """Compute all comparison metrics: MSE, percent error, angular error, scaled error, magnitude diff."""
    # Get cropped versions
    pred_crop = top_left_crop(pred, 44, 94).to(device)
    true_crop = top_left_crop(true, 44, 94).to(device)
    mask_crop = top_left_crop(mask, 44, 94).to(device)
    single_mask = mask_crop[:, 0:1, :, :]
    
    # MSE
    mse = calculate_mse(true_crop, pred_crop, mask_crop, normalize=True).item()
    
    # Angular error (degrees)
    u_pred = pred_crop[:, 0, :, :].squeeze()
    v_pred = pred_crop[:, 1, :, :].squeeze()
    u_true = true_crop[:, 0, :, :].squeeze()
    v_true = true_crop[:, 1, :, :].squeeze()
    mask_2d = single_mask.squeeze()
    
    dot = u_pred * u_true + v_pred * v_true
    norm_pred = torch.sqrt(u_pred ** 2 + v_pred ** 2) + 1e-8
    norm_true = torch.sqrt(u_true ** 2 + v_true ** 2) + 1e-8
    cos_angle = torch.clamp(dot / (norm_pred * norm_true), -1.0, 1.0)
    angle_deg = torch.acos(cos_angle) * (180.0 / np.pi)
    angular = angle_deg[mask_2d > 0.5].mean().item()
    
    # Scaled error magnitude (|error| / |true|) - avoiding divide by zero
    error_u = u_pred - u_true
    error_v = v_pred - v_true
    error_mag = torch.sqrt(error_u ** 2 + error_v ** 2)
    true_mag = torch.sqrt(u_true ** 2 + v_true ** 2)
    # Only consider pixels where true magnitude is significant
    valid = (mask_2d > 0.5) & (true_mag > 1e-6)
    scaled_err = (error_mag[valid] / true_mag[valid]).mean().item() if valid.sum() > 0 else float('nan')
    
    # Normalized magnitude difference
    pred_mag = torch.sqrt(u_pred ** 2 + v_pred ** 2)
    mag_diff = torch.abs(pred_mag - true_mag)
    avg_true_mag = true_mag[mask_2d > 0.5].mean()
    norm_mag_diff = (mag_diff / avg_true_mag)[mask_2d > 0.5].mean().item()
    
    # Percent error - handle zeros carefully 
    with torch.no_grad():
        true_crop_safe = true_crop.clone()
        true_crop_safe[true_crop_safe.abs() < 1e-8] = 1e-8  # Avoid div by zero
        percent_error = torch.abs((pred_crop - true_crop) / true_crop_safe)
        percent = percent_error[mask_crop > 0.5].mean().item()
    
    return {
        'mse': mse,
        'percent_error': percent,
        'angular_error': angular,
        'scaled_error_mag': scaled_err,
        'norm_mag_diff': norm_mag_diff,
    }


def run_comparison():
    print("=" * 60)
    print("SIDE-BY-SIDE COMPARISON: Gaussian vs Div-free")
    print("=" * 60)
    print()
    
    # Setup - use div-free config to get data loaders
    mi = ModelInpainter()
    mi.load_models_from_yaml()
    mi._set_up_model(mi.model_paths[0])  # This sets up val_loader
    device = mi.dd.get_device()
    
    # Get test image
    loader = mi.val_loader
    batch = next(iter(loader))
    input_image = batch[0].to(device)
    input_image_original = mi.dd.get_standardizer().unstandardize(
        torch.squeeze(input_image, 0)
    ).to(device)
    input_image_original = torch.unsqueeze(input_image_original, 0)
    
    # Create mask (same for both methods)
    mask_gen = StraightLineMaskGenerator(1, 1)
    land_mask = (input_image_original.abs() > 1e-5).float().to(device)
    raw_mask = mask_gen.generate_mask(input_image.shape)
    mask = raw_mask * land_mask
    
    results = {}
    
    # ==================== GAUSSIAN ====================
    print("Testing GAUSSIAN noise (NO boundary fix)...")
    mi._set_up_model("ddpm/Trained_Models/weekend_ddpm_ocean_model.pt")
    gaussian_noise = GaussianNoise()
    
    # Temporarily disable comb_net for Gaussian
    original_use_comb_net = mi.dd.use_comb_net
    mi.dd.use_comb_net = False
    
    with torch.no_grad():
        result_gaussian = inpaint_generate_new_images(
            mi.best_model, input_image, mask, n_samples=1,
            device=device, resample_steps=1, noise_strategy=gaussian_noise
        )
    
    # Restore setting
    mi.dd.use_comb_net = original_use_comb_net
    
    result_gaussian = mi.dd.get_standardizer().unstandardize(
        torch.squeeze(result_gaussian, 0)
    ).to(device)
    result_gaussian = torch.unsqueeze(result_gaussian, 0)
    
    # Compute all metrics
    metrics_g = compute_all_metrics(result_gaussian, input_image_original, mask, device)
    results['gaussian'] = metrics_g
    print(f"  Gaussian: MSE={metrics_g['mse']:.4f}, Angular={metrics_g['angular_error']:.1f}°")
    print()
    
    # ==================== DIV-FREE ====================
    print("Testing DIV-FREE noise with boundary fix (this will take ~17 min)...")
    mi._set_up_model("ddpm/Trained_Models/div_free_model.pt")
    
    with torch.no_grad():
        result_divfree = inpaint_generate_new_images(
            mi.best_model, input_image, mask, n_samples=1,
            device=device, resample_steps=1, noise_strategy=mi.noise_strategy
        )
    
    result_divfree = mi.dd.get_standardizer().unstandardize(
        torch.squeeze(result_divfree, 0)
    ).to(device)
    result_divfree = torch.unsqueeze(result_divfree, 0)
    
    # Compute all metrics
    metrics_d = compute_all_metrics(result_divfree, input_image_original, mask, device)
    results['divfree'] = metrics_d
    print(f"  Div-free: MSE={metrics_d['mse']:.4f}, Angular={metrics_d['angular_error']:.1f}°")
    print()
    
    # ==================== VISUALIZATION ====================
    print("Generating visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Get numpy arrays
    orig_u = input_image_original[0, 0].cpu().numpy()
    orig_v = input_image_original[0, 1].cpu().numpy()
    mask_np = mask[0, 0].cpu().numpy()
    
    gauss_u = result_gaussian[0, 0].cpu().numpy()
    gauss_v = result_gaussian[0, 1].cpu().numpy()
    
    divfree_u = result_divfree[0, 0].cpu().numpy()
    divfree_v = result_divfree[0, 1].cpu().numpy()
    
    # Crop to focus area (44x94)
    crop_h, crop_w = 44, 94
    orig_u = orig_u[:crop_h, :crop_w]
    orig_v = orig_v[:crop_h, :crop_w]
    gauss_u = gauss_u[:crop_h, :crop_w]
    gauss_v = gauss_v[:crop_h, :crop_w]
    divfree_u = divfree_u[:crop_h, :crop_w]
    divfree_v = divfree_v[:crop_h, :crop_w]
    mask_np = mask_np[:crop_h, :crop_w]
    
    # Create coordinate grid for quiver
    H, W = orig_u.shape
    Y, X = np.mgrid[0:H, 0:W]
    
    # Subsample for cleaner arrows
    skip = 3
    
    def plot_vector_field(ax, u, v, title, mask_overlay=None):
        """Plot vector field with quiver arrows."""
        # Background: magnitude as faint color
        mag = np.sqrt(u**2 + v**2)
        ax.imshow(mag, cmap='Blues', alpha=0.3, origin='upper')
        
        # Quiver plot (subsampled)
        ax.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
                  u[::skip, ::skip], -v[::skip, ::skip],  # negative v for correct orientation
                  color='black', scale=3, width=0.003)
        
        # Mask boundary
        if mask_overlay is not None:
            ax.contour(mask_overlay, levels=[0.5], colors='red', linewidths=2)
        
        ax.set_title(title, fontsize=12)
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)  # Flip y-axis
        ax.set_aspect('equal')
        ax.axis('off')
    
    # Plot all four panels
    plot_vector_field(axes[0, 0], orig_u, orig_v, 'Original (Ground Truth)', mask_np)
    
    # Masked input - zero out masked region
    masked_u = orig_u * (1 - mask_np)
    masked_v = orig_v * (1 - mask_np)
    plot_vector_field(axes[0, 1], masked_u, masked_v, 'Masked Input', mask_np)
    
    plot_vector_field(axes[1, 0], gauss_u, gauss_v, 
                      f'Gaussian Inpainting\nMSE={metrics_g["mse"]:.4f}, Angle={metrics_g["angular_error"]:.1f}°', 
                      mask_np)
    
    plot_vector_field(axes[1, 1], divfree_u, divfree_v, 
                      f'Div-free Inpainting\nMSE={metrics_d["mse"]:.4f}, Angle={metrics_d["angular_error"]:.1f}°', 
                      mask_np)
    
    plt.suptitle('Gaussian vs Div-free Inpainting: Vector Field Comparison', fontsize=14)
    plt.tight_layout()
    
    save_path = BASE_DIR / 'plots' / 'outputs' / 'comparison_gaussian_vs_divfree.png'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to: {save_path}")
    plt.show()
    
    # ==================== SUMMARY ====================
    print("=" * 80)
    print("FINAL COMPARISON - ALL METRICS")
    print("=" * 80)
    print()
    print(f"{'Metric':<25} {'Gaussian':<15} {'Div-free':<15} {'Winner':<10}")
    print("-" * 70)
    
    metric_names = {
        'mse': 'MSE',
        'percent_error': 'Percent Error',
        'angular_error': 'Angular Error (°)',
        'scaled_error_mag': 'Scaled Error Mag',
        'norm_mag_diff': 'Norm Mag Diff',
    }
    
    wins = {'gaussian': 0, 'divfree': 0}
    
    for key, name in metric_names.items():
        g_val = metrics_g[key]
        d_val = metrics_d[key]
        # Lower is better for all metrics
        if g_val < d_val:
            winner = "Gaussian"
            wins['gaussian'] += 1
        elif d_val < g_val:
            winner = "Div-free"
            wins['divfree'] += 1
        else:
            winner = "Tie"
        
        if key == 'angular_error':
            print(f"{name:<25} {g_val:<15.1f} {d_val:<15.1f} {winner}")
        else:
            print(f"{name:<25} {g_val:<15.4f} {d_val:<15.4f} {winner}")
    
    print("-" * 70)
    print()
    print(f"Overall: Gaussian wins {wins['gaussian']}, Div-free wins {wins['divfree']}")
    
    # MSE improvement (primary metric)
    mse_g = metrics_g['mse']
    mse_d = metrics_d['mse']
    if mse_g > mse_d:
        improvement = (mse_g - mse_d) / mse_g * 100
        print(f"✅ Div-free MSE is {improvement:.1f}% BETTER than Gaussian")
    else:
        degradation = (mse_d - mse_g) / mse_g * 100
        print(f"❌ Div-free MSE is {degradation:.1f}% WORSE than Gaussian")
    
    print()
    return results


if __name__ == '__main__':
    run_comparison()
