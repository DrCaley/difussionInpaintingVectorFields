#!/usr/bin/env python
"""
Evaluate eddy reconstruction quality from saved inference .pt files.

Loads a saved inference result (ground truth + one or more predictions),
computes Okubo–Weiss eddy detection on the ground truth, and evaluates
how well each method reconstructs those eddy features.

Usage:
    # Single file
    PYTHONPATH=. python scripts/eval_eddies.py results/some_run.pt

    # Compare multiple methods (multiple .pt files)
    PYTHONPATH=. python scripts/eval_eddies.py results/run_a.pt results/run_b.pt

    # Against a specific experiment's results directory
    PYTHONPATH=. python scripts/eval_eddies.py experiments/02_inpaint_algorithm/repaint_gaussian_attn/results/*.pt

    # With visualization
    PYTHONPATH=. python scripts/eval_eddies.py results/some_run.pt --plot

    # Specify ocean crop size
    PYTHONPATH=. python scripts/eval_eddies.py results/some_run.pt --crop 44 94
"""
import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import argparse
import torch
import numpy as np

from ddpm.utils.eddy_detection import (
    detect_eddies, eddy_metrics, print_eddy_metrics, okubo_weiss,
)


def top_left_crop(tensor, h, w):
    """Crop to top-left (h, w) region. Works on (..., H, W) tensors."""
    return tensor[..., :h, :w]


def load_and_crop(pt_path: str, crop_h: int = 44, crop_w: int = 94):
    """
    Load a saved .pt run file and crop to ocean region.

    Returns dict with:
        'gt'     : (2, H, W)  ground truth velocity
        'methods': {name: (2, H, W)}  prediction(s)
        'mask'   : (H, W) bool, True=missing
        'meta'   : dict of scalar metadata
    """
    data = torch.load(pt_path, map_location='cpu', weights_only=False)

    # --- Ground truth ---
    gt = None
    for key in ('gt', 'ground_truth', 'true'):
        if key in data:
            gt = data[key]
            break
    if gt is None:
        raise KeyError(f"No ground-truth key found in {pt_path}. "
                       f"Keys: {list(data.keys())}")
    if gt.dim() == 4:
        gt = gt.squeeze(0)
    gt = top_left_crop(gt.unsqueeze(0), crop_h, crop_w).squeeze(0)

    # --- Mask ---
    mask = None
    for key in ('missing_mask', 'mask'):
        if key in data:
            mask = data[key]
            break
    if mask is not None:
        if mask.dim() == 4:
            mask = mask.squeeze(0)
        if mask.shape[0] == 2:
            mask = mask[0]
        elif mask.shape[0] == 1:
            mask = mask[0]
        mask = top_left_crop(
            mask.unsqueeze(0).unsqueeze(0), crop_h, crop_w
        ).squeeze(0).squeeze(0)
        mask = mask.bool()

    # --- Methods ---
    method_keys = {
        'ddpm_output': 'DDPM',
        'ddpm_out': 'DDPM',
        'dps_out': 'DPS',
        'gp_out': 'GP',
        'gp_output': 'GP',
        'adaptive_cg_out': 'Adaptive-CG',
        'repaint_out': 'RePaint',
        'inpainted': 'Inpainted',
    }
    methods = {}
    for k, name in method_keys.items():
        if k in data:
            v = data[k]
            if v.dim() == 4:
                v = v.squeeze(0)
            v = top_left_crop(v.unsqueeze(0), crop_h, crop_w).squeeze(0)
            methods[name] = v

    # If nothing matched the known keys, grab any (B,2,H,W) or (2,H,W) tensor
    if not methods:
        skip = {'gt', 'ground_truth', 'true', 'missing_mask', 'mask'}
        for k, v in data.items():
            if k in skip or not isinstance(v, torch.Tensor):
                continue
            if v.dim() in (3, 4) and (v.shape[-3] == 2 if v.dim() == 3
                                       else v.shape[-3] == 2):
                if v.dim() == 4:
                    v = v.squeeze(0)
                v = top_left_crop(v.unsqueeze(0), crop_h, crop_w).squeeze(0)
                methods[k] = v

    # --- Metadata ---
    meta = {}
    for k in ('mask_pct', 'val_index', 'epoch', 'n_steps', 'gp_mse',
              'ddpm_mse', 'dps_mse', 'ratio'):
        if k in data:
            val = data[k]
            meta[k] = val.item() if isinstance(val, torch.Tensor) else val

    return {'gt': gt, 'methods': methods, 'mask': mask, 'meta': meta}


def plot_eddy_comparison(gt, pred, method_name, eddies_true, W_true, W_pred,
                         omega_true, omega_pred, save_path=None):
    """Visualize OW fields, vorticity, and detected eddies."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import TwoSlopeNorm

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Row 0: Ground truth
    # OW field
    vmax_ow = max(abs(W_true.min()), abs(W_true.max())) * 0.8
    norm_ow = TwoSlopeNorm(vcenter=0, vmin=-vmax_ow, vmax=vmax_ow)

    im0 = axes[0, 0].imshow(W_true.numpy(), cmap='RdBu_r', norm=norm_ow)
    axes[0, 0].set_title('Ground Truth: Okubo–Weiss (W)')
    plt.colorbar(im0, ax=axes[0, 0], shrink=0.7)

    # Vorticity
    vmax_v = max(abs(omega_true.min()), abs(omega_true.max())) * 0.8
    norm_v = TwoSlopeNorm(vcenter=0, vmin=-vmax_v, vmax=vmax_v)
    im1 = axes[0, 1].imshow(omega_true.numpy(), cmap='RdBu_r', norm=norm_v)
    axes[0, 1].set_title('Ground Truth: Vorticity (ω)')
    plt.colorbar(im1, ax=axes[0, 1], shrink=0.7)

    # Detected eddies overlaid on speed
    speed_true = (gt[0] ** 2 + gt[1] ** 2).sqrt().numpy()
    axes[0, 2].imshow(speed_true, cmap='viridis')
    for e in eddies_true:
        color = 'red' if e.is_cyclonic else 'blue'
        mask_outline = e.mask.numpy().astype(float)
        axes[0, 2].contour(mask_outline, levels=[0.5], colors=[color],
                           linewidths=1.5)
        axes[0, 2].plot(e.center_x, e.center_y, 'x', color=color, ms=8)
    red_patch = mpatches.Patch(color='red', label='Cyclonic')
    blue_patch = mpatches.Patch(color='blue', label='Anticyclonic')
    axes[0, 2].legend(handles=[red_patch, blue_patch], loc='upper right',
                      fontsize=8)
    axes[0, 2].set_title(f'GT Eddies ({len(eddies_true)} detected)')

    # Row 1: Prediction
    im3 = axes[1, 0].imshow(W_pred.numpy(), cmap='RdBu_r', norm=norm_ow)
    axes[1, 0].set_title(f'{method_name}: Okubo–Weiss (W)')
    plt.colorbar(im3, ax=axes[1, 0], shrink=0.7)

    im4 = axes[1, 1].imshow(omega_pred.numpy(), cmap='RdBu_r', norm=norm_v)
    axes[1, 1].set_title(f'{method_name}: Vorticity (ω)')
    plt.colorbar(im4, ax=axes[1, 1], shrink=0.7)

    # Prediction speed with GT eddy outlines
    speed_pred = (pred[0] ** 2 + pred[1] ** 2).sqrt().numpy()
    axes[1, 2].imshow(speed_pred, cmap='viridis')
    for e in eddies_true:
        color = 'red' if e.is_cyclonic else 'blue'
        mask_outline = e.mask.numpy().astype(float)
        axes[1, 2].contour(mask_outline, levels=[0.5], colors=[color],
                           linewidths=1.5, linestyles='--')
    axes[1, 2].set_title(f'{method_name}: Speed + GT Eddy Outlines')

    plt.suptitle(f'Eddy Evaluation: {method_name}', fontsize=14, y=1.01)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate eddy reconstruction from saved .pt results')
    parser.add_argument('pt_files', nargs='+', help='Saved .pt result file(s)')
    parser.add_argument('--crop', nargs=2, type=int, default=[44, 94],
                        help='Ocean crop (H W), default: 44 94')
    parser.add_argument('--threshold-sigma', type=float, default=0.2,
                        help='OW threshold = -sigma * std(W). Default 0.2')
    parser.add_argument('--min-area', type=int, default=16,
                        help='Minimum eddy area in pixels. Default 16')
    parser.add_argument('--shore-buffer', type=int, default=2,
                        help='Pixels to erode from shore. Default 2')
    parser.add_argument('--plot', action='store_true',
                        help='Generate visualization plots')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Directory to save plots (default: alongside .pt)')
    args = parser.parse_args()

    all_results = {}

    for pt_file in args.pt_files:
        print(f"\n{'#' * 60}")
        print(f"  File: {pt_file}")
        print(f"{'#' * 60}")

        try:
            run = load_and_crop(pt_file, args.crop[0], args.crop[1])
        except Exception as e:
            print(f"  ERROR loading: {e}")
            continue

        gt = run['gt']
        meta = run['meta']
        if meta:
            print(f"  Metadata: {meta}")

        if not run['methods']:
            print("  WARNING: No prediction tensors found. Skipping.")
            continue

        for method_name, pred in run['methods'].items():
            print(f"\n  --- {method_name} ---")
            metrics = eddy_metrics(
                pred, gt,
                threshold_sigma=args.threshold_sigma,
                min_area=args.min_area,
                shore_buffer=args.shore_buffer,
            )
            print_eddy_metrics(metrics, title=f"{method_name} ({Path(pt_file).stem})")

            key = f"{Path(pt_file).stem}/{method_name}"
            all_results[key] = metrics

            if args.plot:
                _, W_true, omega_true = detect_eddies(
                    gt, threshold_sigma=args.threshold_sigma,
                    min_area=args.min_area,
                    shore_buffer=args.shore_buffer,
                )
                W_pred, omega_pred, _, _ = okubo_weiss(pred)
                eddies_true, _, _ = detect_eddies(
                    gt, threshold_sigma=args.threshold_sigma,
                    min_area=args.min_area,
                    shore_buffer=args.shore_buffer,
                )

                save_dir = Path(args.save_dir) if args.save_dir else Path(pt_file).parent
                save_path = save_dir / f"eddy_eval_{Path(pt_file).stem}_{method_name}.png"
                plot_eddy_comparison(
                    gt, pred, method_name, eddies_true,
                    W_true, W_pred, omega_true, omega_pred,
                    save_path=str(save_path),
                )

    # Summary comparison table if multiple results
    if len(all_results) > 1:
        print(f"\n{'=' * 80}")
        print("  COMPARISON SUMMARY")
        print(f"{'=' * 80}")
        key_cols = ['n_eddies_true', 'n_eddies_pred', 'eddy_iou',
                    'eddy_detection_rate', 'eddy_velocity_mse',
                    'eddy_angular_error', 'ow_correlation', 'vort_correlation']
        header = f"{'Method':40s}" + "".join(f"{k:>18s}" for k in key_cols)
        print(header)
        print("-" * len(header))
        for name, m in all_results.items():
            vals = []
            for k in key_cols:
                v = m.get(k, float('nan'))
                if isinstance(v, int) or (isinstance(v, float) and v == int(v)):
                    vals.append(f"{int(v):>18d}")
                else:
                    vals.append(f"{v:>18.4f}")
            print(f"{name:40s}" + "".join(vals))
        print()


if __name__ == '__main__':
    main()
