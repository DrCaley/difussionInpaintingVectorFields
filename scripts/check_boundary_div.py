#!/usr/bin/env python
"""Quick check: does CombNet reduce divergence at the boundary?"""
import torch
import torch.nn.functional as F
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ddpm.vector_combination.combiner_unet import VectorCombinationUNet
from ddpm.helper_functions.compute_divergence import compute_divergence

device = 'mps'

# Load trained model
print('Loading trained CombNet...')
checkpoint = torch.load('ddpm/Trained_Models/pretrained_combnet.pt', weights_only=False)
model = VectorCombinationUNet(n_channels=4, n_classes=2).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"  Loaded from epoch {checkpoint['epoch']+1}, loss={checkpoint['loss']:.6f}")

# Load small dataset
print('Loading test data...')
data = torch.load('results/combnet_training_data_real.pt', weights_only=False)
print(f"  {len(data['known'])} samples")

# Analyze 10 samples
n_samples = 10
print(f"\nAnalyzing boundary divergence on {n_samples} samples...")

results = []
for i in range(n_samples):
    naive = data['naive'][i:i+1].to(device)
    mask = data['mask'][i:i+1].to(device)
    known = data['known'][i:i+1].to(device)
    
    # Get network prediction
    inputs = torch.cat([naive, mask], dim=1)
    with torch.no_grad():
        predicted = model(inputs)
    
    # Create boundary mask (dilate mask, find edge)
    mask_single = mask[:, 0:1, :, :]
    kernel = torch.ones((1, 1, 3, 3), device=device)
    dilated = F.conv2d(mask_single.float(), kernel, padding=1).clamp(0, 1)
    boundary = (dilated - mask_single.float()).squeeze()
    
    # Compute divergence fields
    div_naive = compute_divergence(naive[0,0], naive[0,1])
    div_pred = compute_divergence(predicted[0,0], predicted[0,1])
    div_known = compute_divergence(known[0,0], known[0,1])
    
    # Divergence at boundary only
    boundary_mask = boundary > 0.5
    if boundary_mask.sum() > 0:
        div_naive_boundary = div_naive[boundary_mask].abs().mean().item()
        div_pred_boundary = div_pred[boundary_mask].abs().mean().item()
        div_known_boundary = div_known[boundary_mask].abs().mean().item()
        
        results.append({
            'naive_boundary': div_naive_boundary,
            'pred_boundary': div_pred_boundary,
            'known_boundary': div_known_boundary,
        })

# Print results
print("\n" + "="*60)
print("BOUNDARY DIVERGENCE ANALYSIS")
print("="*60)
print("Sample   Naive_Bnd    Pred_Bnd     Known_Bnd    Reduction")
print("-"*60)
for i, r in enumerate(results):
    reduction = (1 - r['pred_boundary']/r['naive_boundary'])*100 if r['naive_boundary'] > 0 else 0
    print(f"{i+1:<8} {r['naive_boundary']:<12.4f} {r['pred_boundary']:<12.4f} {r['known_boundary']:<12.4f} {reduction:.1f}%")

avg_naive = sum(r['naive_boundary'] for r in results) / len(results)
avg_pred = sum(r['pred_boundary'] for r in results) / len(results)
avg_known = sum(r['known_boundary'] for r in results) / len(results)
avg_reduction = (1 - avg_pred/avg_naive)*100

print("-"*60)
print(f"AVERAGE  {avg_naive:<12.4f} {avg_pred:<12.4f} {avg_known:<12.4f} {avg_reduction:.1f}%")
print()
print(f"Known field boundary div (theoretical floor): {avg_known:.4f}")
print(f"Network reduces boundary divergence by: {avg_reduction:.1f}%")
