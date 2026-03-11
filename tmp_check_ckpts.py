#!/usr/bin/env python3
"""Quick check: compare old topo vs new voronoi checkpoint formats."""
import torch

old = torch.load(
    'experiments/10_topology_metrics/topo_aware_training/results/'
    'inpaint_gaussian_t250_best_ema_weights.pt',
    map_location='cpu', weights_only=False)
print('=== OLD TOPO EMA ===')
print('Type:', type(old))
if isinstance(old, dict):
    keys = list(old.keys())
    print('First 5 keys:', keys[:5])
    print('Last 5 keys:', keys[-5:])
    print('Num keys:', len(keys))

print()

new = torch.load(
    'experiments/10_topology_metrics/voronoi_topo_training/results/'
    'inpaint_gaussian_t250_best_ema_weights.pt',
    map_location='cpu', weights_only=False)
print('=== NEW VORONOI EMA ===')
print('Type:', type(new))
if isinstance(new, dict):
    keys = list(new.keys())
    print('First 5 keys:', keys[:5])
    print('Last 5 keys:', keys[-5:])
    print('Num keys:', len(keys))
