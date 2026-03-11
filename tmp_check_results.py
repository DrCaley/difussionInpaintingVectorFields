#!/usr/bin/env python3
"""Quick check of GP-conditioned eval results."""
import torch, numpy as np

d = torch.load('results/gp_conditioned_eval/gp_conditioned_eval.pt',
               map_location='cpu', weights_only=False)
print('Keys:', list(d.keys()))
print('N samples:', d['n_samples'])
print('Methods tested:', d.get('methods_tested'))
print('Model:', d.get('model'))
print()

samples = d['samples']
print('Sample keys:', list(samples[0].keys()))
print()

gp_mses = [s['gp_mse'] for s in samples]
print(f'GP baseline: mean={np.mean(gp_mses):.6f}, median={np.median(gp_mses):.6f}')

if 'full_reverse_mse' in samples[0]:
    full_mses = [s['full_reverse_mse'] for s in samples]
    ratio = np.mean(full_mses) / np.mean(gp_mses)
    wins = sum(1 for f, g in zip(full_mses, gp_mses) if f < g)
    print(f'Full-reverse: mean={np.mean(full_mses):.6f}, ratio={ratio:.3f}x, wins={wins}/{len(samples)}')

if 'gp_warm_mse' in samples[0]:
    warm_mses = [s['gp_warm_mse'] for s in samples]
    ratio = np.mean(warm_mses) / np.mean(gp_mses)
    wins = sum(1 for f, g in zip(warm_mses, gp_mses) if f < g)
    print(f'GP-warm: mean={np.mean(warm_mses):.6f}, ratio={ratio:.3f}x, wins={wins}/{len(samples)}')

# Per-sample detail
print('\nPer-sample breakdown:')
print(f'{"#":>3} {"ValIdx":>7} {"Type":>6} {"GP_MSE":>10}', end='')
if 'full_reverse_mse' in samples[0]:
    print(f' {"Full-Rev":>10}', end='')
if 'gp_warm_mse' in samples[0]:
    print(f' {"GP-Warm":>10}', end='')
print()

for s in samples:
    line = f'{s["idx"]+1:>3} {s["val_idx"]:>7} {"EDDY" if s["is_eddy_sample"] else "clean":>6} {s["gp_mse"]:>10.6f}'
    if 'full_reverse_mse' in s:
        line += f' {s["full_reverse_mse"]:>10.6f}'
    if 'gp_warm_mse' in s:
        line += f' {s["gp_warm_mse"]:>10.6f}'
    print(line)
