#!/usr/bin/env python3
"""Full MSE audit: check observed-cell leakage for ALL methods at 5%."""
import torch, numpy as np, glob, os

results_dir = 'experiments/07_ddpm_composite/bulk_eval/results/5.0pct_n100_tc75_s6'
sample_dirs = sorted(glob.glob(os.path.join(results_dir, 'sample_*')))

methods = ['gp_mean', 'vcnn', 'composite', 'gpdiff']
accum = {m: {'all': [], 'unobs': [], 'obs': []} for m in methods}

def to_t(x):
    return torch.from_numpy(x).float() if isinstance(x, np.ndarray) else x.float()

for sd in sample_dirs:
    d = torch.load(os.path.join(sd, 'tensors.pt'), map_location='cpu', weights_only=False)
    
    gt = to_t(d['gt'])
    om = to_t(d['ocean_mask']).bool()
    obs = to_t(d['obs_mask']).bool()
    unobs = om & ~obs
    obs_cells = om & obs
    
    for m in methods:
        pred = to_t(d[m])
        mse_all = ((gt - pred)[:, om]**2).mean().item()
        mse_unobs = ((gt - pred)[:, unobs]**2).mean().item()
        mse_obs = ((gt - pred)[:, obs_cells]**2).mean().item()
        accum[m]['all'].append(mse_all)
        accum[m]['unobs'].append(mse_unobs)
        accum[m]['obs'].append(mse_obs)

print("=== MSE Audit (100 samples, 5% coverage) ===\n")
print(f"{'Method':<12} {'All Ocean':>12} {'Unobs Only':>12} {'Obs Only':>12} {'Bias(all-unobs)':>15}")
print("-" * 65)
for m in methods:
    a = np.mean(accum[m]['all'])
    u = np.mean(accum[m]['unobs'])
    o = np.mean(accum[m]['obs'])
    delta = a - u
    print(f"{m:<12} {a:>12.6f} {u:>12.6f} {o:>12.6f} {delta:>+15.6f}")

print("\n=== Fair comparison: Unobserved-only MSE & Wins ===\n")
print(f"{'Method':<12} {'Unobs MSE':>12} {'Wins':>6}")
print("-" * 32)

wins = {m: 0 for m in methods}
for i in range(len(sample_dirs)):
    best_m = min(methods, key=lambda m: accum[m]['unobs'][i])
    wins[best_m] += 1

for m in methods:
    print(f"{m:<12} {np.mean(accum[m]['unobs']):>12.6f} {wins[m]:>6}")

# Also show the summary.pt structure
print("\n--- summary.pt keys ---")
summary = torch.load(os.path.join(results_dir, 'summary.pt'), map_location='cpu', weights_only=False)
for k in sorted(summary.keys()):
    v = summary[k]
    if isinstance(v, (int, float, str)):
        print(f"  {k}: {v}")
    elif isinstance(v, dict):
        print(f"  {k}: {list(v.keys())[:10]}")

# Also do the same check for 1% to see if the pattern holds
print("\n\n=== MSE Audit (100 samples, 1% coverage) ===\n")
results_dir_1 = 'experiments/07_ddpm_composite/bulk_eval/results/1.0pct_n100_tc200_s6'
sample_dirs_1 = sorted(glob.glob(os.path.join(results_dir_1, 'sample_*')))
accum_1 = {m: {'all': [], 'unobs': [], 'obs': []} for m in methods}

for sd in sample_dirs_1:
    d = torch.load(os.path.join(sd, 'tensors.pt'), map_location='cpu', weights_only=False)
    gt = to_t(d['gt'])
    om = to_t(d['ocean_mask']).bool()
    obs = to_t(d['obs_mask']).bool()
    unobs = om & ~obs
    obs_cells = om & obs
    for m in methods:
        pred = to_t(d[m])
        mse_all = ((gt - pred)[:, om]**2).mean().item()
        mse_unobs = ((gt - pred)[:, unobs]**2).mean().item()
        mse_obs = ((gt - pred)[:, obs_cells]**2).mean().item()
        accum_1[m]['all'].append(mse_all)
        accum_1[m]['unobs'].append(mse_unobs)
        accum_1[m]['obs'].append(mse_obs)

print(f"{'Method':<12} {'All Ocean':>12} {'Unobs Only':>12} {'Obs Only':>12} {'Bias(all-unobs)':>15}")
print("-" * 65)
for m in methods:
    a = np.mean(accum_1[m]['all'])
    u = np.mean(accum_1[m]['unobs'])
    o = np.mean(accum_1[m]['obs'])
    delta = a - u
    print(f"{m:<12} {a:>12.6f} {u:>12.6f} {o:>12.6f} {delta:>+15.6f}")

print("\n--- 1% Unobs-only Wins ---")
wins_1 = {m: 0 for m in methods}
for i in range(len(sample_dirs_1)):
    best_m = min(methods, key=lambda m: accum_1[m]['unobs'][i])
    wins_1[best_m] += 1
for m in methods:
    print(f"  {m:<12} {np.mean(accum_1[m]['unobs']):>12.6f} wins={wins_1[m]}")
