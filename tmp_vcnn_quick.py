#!/usr/bin/env python3
"""Quick comparison: VCNN vs our method."""
import torch, sys
sys.path.insert(0, '.')

# Voronoi-CNN results
vcnn = torch.load('results/voronoi_cnn_eval/voronoi_cnn_eval_results.pt', map_location='cpu', weights_only=False)
results = vcnn['results']
vcnn_mses = [float(r['mse_missing_only']) for r in results]

# GP baseline
gp_eval = torch.load('results/eddy_balanced_eval/bulk_eval_eddy_balanced_100.pt', map_location='cpu', weights_only=False)
gp_mses = [float(s['gp_mse']) for s in gp_eval['samples']]
gpdiff_mses = [float(s['ddpm_mse']) for s in gp_eval['samples']]

vcnn_mean = sum(vcnn_mses) / len(vcnn_mses)
gp_mean = sum(gp_mses) / len(gp_mses)
gpdiff_mean = sum(gpdiff_mses) / len(gpdiff_mses)

print(f"VCNN mean MSE:      {vcnn_mean:.6f}  ratio={vcnn_mean/gp_mean:.3f}x GP")
print(f"GP mean MSE:        {gp_mean:.6f}")
print(f"Old GP-Diff (uncond):{gpdiff_mean:.6f}  ratio={gpdiff_mean/gp_mean:.3f}x GP")
print(f"New FiLM Ens10:     ~0.00292  ratio=~0.810x GP")

# Win rates
gp_by_vi = {s['val_idx']: float(s['gp_mse']) for s in gp_eval['samples']}
vcnn_w = sum(1 for r in results if r['val_idx'] in gp_by_vi and float(r['mse_missing_only']) < gp_by_vi[r['val_idx']])
print(f"\nVCNN beats GP: {vcnn_w}/100")

# VCNN eddy vs non-eddy
eddy_m = [float(r['mse_missing_only']) for r in results if r['is_eddy']]
noneddy_m = [float(r['mse_missing_only']) for r in results if not r['is_eddy']]
print(f"\nVCNN eddy mean:     {sum(eddy_m)/len(eddy_m):.6f}")
print(f"VCNN non-eddy mean: {sum(noneddy_m)/len(noneddy_m):.6f}")

# Architecture info
print(f"\nVCNN sample keys: {list(results[0].keys())}")
print(f"VCNN metadata keys: {[k for k in vcnn.keys() if k != 'results']}")

# Check VCNN training info
import os
for path in ['results/voronoi_cnn', 'experiments/05_voronoi_warmstart']:
    if os.path.exists(path):
        print(f"\n{path}: {os.listdir(path)}")
