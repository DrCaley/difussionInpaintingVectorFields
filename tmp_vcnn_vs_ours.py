#!/usr/bin/env python3
"""Compare Voronoi-CNN vs our FiLM+GP diffusion approach."""
import torch, numpy as np, sys
sys.path.insert(0, '.')

# Voronoi-CNN results
vcnn = torch.load('results/voronoi_cnn_eval/voronoi_cnn_eval_results.pt', map_location='cpu', weights_only=False)
results = vcnn['results']
vcnn_mses = [r['mse_missing_only'] for r in results]
print(f"VCNN mean MSE: {np.mean(vcnn_mses):.6f}")
print(f"VCNN median MSE: {np.median(vcnn_mses):.6f}")
print(f"VCNN keys (non-results): {[k for k in vcnn.keys() if k != 'results']}")
if results:
    print(f"VCNN sample keys: {list(results[0].keys())}")

# GP baseline from eddy-balanced eval  
gp_eval = torch.load('results/eddy_balanced_eval/bulk_eval_eddy_balanced_100.pt', map_location='cpu', weights_only=False)
gp_mses = [s['gp_mse'] for s in gp_eval['results']]
gpdiff_mses = [s['ddpm_mse'] for s in gp_eval['results']]
print(f"\nGP mean MSE: {np.mean(gp_mses):.6f}")
print(f"Old GP-Diff (uncond) mean MSE: {np.mean(gpdiff_mses):.6f}")
print(f"VCNN ratio vs GP: {np.mean(vcnn_mses) / np.mean(gp_mses):.4f}")
print(f"Old GP-Diff ratio vs GP: {np.mean(gpdiff_mses) / np.mean(gp_mses):.4f}")

# Win rates
gp_by_vi = {s['val_idx']: s['gp_mse'] for s in gp_eval['results']}
gpdiff_by_vi = {s['val_idx']: s['ddpm_mse'] for s in gp_eval['results']}
vcnn_vs_gp = vcnn_vs_gpdiff = matched = 0
for r in results:
    vi = r['val_idx']
    if vi in gp_by_vi:
        matched += 1
        if r['mse_missing_only'] < gp_by_vi[vi]:
            vcnn_vs_gp += 1
        if r['mse_missing_only'] < gpdiff_by_vi[vi]:
            vcnn_vs_gpdiff += 1

print(f"\nVCNN beats GP: {vcnn_vs_gp}/{matched}")
print(f"VCNN beats old GP-Diff: {vcnn_vs_gpdiff}/{matched}")

# Look at VCNN architecture details
print("\n--- VCNN Architecture ---")
from scripts.voronoi_cnn_model import VoronoiCNN
model = VoronoiCNN()
n_params = sum(p.numel() for p in model.parameters())
print(f"VCNN params: {n_params:,}")

# Check if trained weights exist
import os
vcnn_dir = 'results/voronoi_cnn'
if os.path.isdir(vcnn_dir):
    print(f"\nVCNN trained model dir: {os.listdir(vcnn_dir)}")

# Check what input the VCNN gets
print("\n--- What VCNN sees vs what our model sees ---")
r0 = results[0]
print(f"VCNN result keys: {list(r0.keys())}")
if 'voronoi_pred' in r0:
    print(f"VCNN pred shape: {r0['voronoi_pred'].shape}")
if 'ground_truth' in r0:
    print(f"Ground truth shape: {r0['ground_truth'].shape}")

# Per-sample comparison
print("\n\n=== KEY COMPARISON ===")
print(f"{'Method':<25} {'Mean MSE':>10} {'Ratio vs GP':>12} {'Win Rate vs GP':>15}")
print("-" * 65)
print(f"{'GP baseline':<25} {np.mean(gp_mses):>10.6f} {'1.000x':>12} {'---':>15}")
print(f"{'Voronoi-CNN':<25} {np.mean(vcnn_mses):>10.6f} {np.mean(vcnn_mses)/np.mean(gp_mses):>11.3f}x {vcnn_vs_gp:>3}/{matched:>3}")
print(f"{'Old GP-Diff (uncond)':<25} {np.mean(gpdiff_mses):>10.6f} {np.mean(gpdiff_mses)/np.mean(gp_mses):>11.3f}x {'79/100':>15}")
print(f"{'New FiLM Ens10 (10 samp)':<25} {'~0.00292':>10} {'~0.810x':>12} {'10/10':>15}")

# Eddy vs non-eddy breakdown for VCNN
eddy_mses = [r['mse_missing_only'] for r in results if r['is_eddy']]
noneddy_mses = [r['mse_missing_only'] for r in results if not r['is_eddy']]
print(f"\nVCNN Eddy MSE:     {np.mean(eddy_mses):.6f}  Non-eddy: {np.mean(noneddy_mses):.6f}")
print(f"GP   Eddy MSE:     {np.mean([s['gp_mse'] for s in gp_eval['results'] if s.get('is_eddy', False)]):.6f}")
