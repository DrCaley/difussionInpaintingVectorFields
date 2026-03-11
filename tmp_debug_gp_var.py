#!/usr/bin/env python3
"""Quick debug script for GP variance computation."""
import numpy as np, torch, sys, pickle
sys.path.insert(0, '.')
from ddpm.helper_functions.interpolation_tool import gp_fill

OCEAN_H, OCEAN_W, FULL_H, FULL_W = 44, 94, 64, 128
OBS_ROW = 22

# Ocean mask from data
with open('data.pickle', 'rb') as f:
    train_np, _, _ = pickle.load(f)
t = torch.from_numpy(np.ascontiguousarray(train_np)).float().permute(3,2,1,0)
t = torch.nan_to_num(t, nan=0.0)
speed = (t[0,0]**2 + t[0,1]**2).numpy()
ocean_mask = (speed > 1e-10).astype(np.float32)

# Build mask for gp_fill
obs_mask = np.zeros((OCEAN_H, OCEAN_W), dtype=np.float32)
obs_mask[OBS_ROW, :] = 1.0
obs_mask *= ocean_mask

gp_mask = np.ones((FULL_H, FULL_W), dtype=np.float32)
gp_mask[:OCEAN_H, :OCEAN_W] = 1.0 - obs_mask
gp_mask[:OCEAN_H, :OCEAN_W] *= ocean_mask
gp_mask[OCEAN_H:, :] = 0.0
gp_mask[:, OCEAN_W:] = 0.0

# Use real velocity values from first sample
vel = np.zeros((1, 2, FULL_H, FULL_W), dtype=np.float32)
vel[0, :, :OCEAN_H, :OCEAN_W] = t[0].numpy()

gp_mask_t = torch.from_numpy(gp_mask).unsqueeze(0).unsqueeze(0).expand(1,2,-1,-1).clone()
vel_t = torch.from_numpy(vel)

print(f'gp_mask known count: {(gp_mask_t[0,0]==0).sum().item()}')
print(f'gp_mask unknown ocean count: {(gp_mask_t[0,0]==1).sum().item()}')
print(f'vel at row 22 (sample): {vel[0,0,OBS_ROW,40:45]}')

filled, var_map = gp_fill(
    vel_t, gp_mask_t,
    lengthscale=14.1, variance=0.0103420345, noise=1e-8,
    use_double=True, kernel_type='rbf_legacy', coord_system='pixels',
    return_variance=True,
)

print(f'var_map shape: {var_map.shape}')
v = var_map[0, 0, :OCEAN_H, :OCEAN_W].numpy()
print(f'var u - min: {v.min():.2e}, max: {v.max():.2e}, mean: {v.mean():.2e}, nonzero: {np.count_nonzero(v)}')
v2 = var_map[0, 1, :OCEAN_H, :OCEAN_W].numpy()
print(f'var v - min: {v2.min():.2e}, max: {v2.max():.2e}, mean: {v2.mean():.2e}, nonzero: {np.count_nonzero(v2)}')

# Also try with standard rbf kernel
print('\n--- Try with standard RBF kernel (not legacy) ---')
filled2, var_map2 = gp_fill(
    vel_t, gp_mask_t,
    lengthscale=14.1, variance=0.0103420345, noise=1e-8,
    use_double=True, kernel_type='rbf', coord_system='pixels',
    return_variance=True,
)
v3 = var_map2[0, 0, :OCEAN_H, :OCEAN_W].numpy()
print(f'var u (rbf) - min: {v3.min():.2e}, max: {v3.max():.2e}, mean: {v3.mean():.2e}, nonzero: {np.count_nonzero(v3)}')

# Try with larger variance to see if it's a scale issue
print('\n--- Try with variance=1.0 (larger) ---')
filled3, var_map3 = gp_fill(
    vel_t, gp_mask_t,
    lengthscale=14.1, variance=1.0, noise=1e-8,
    use_double=True, kernel_type='rbf_legacy', coord_system='pixels',
    return_variance=True,
)
v4 = var_map3[0, 0, :OCEAN_H, :OCEAN_W].numpy()
print(f'var u (v=1.0) - min: {v4.min():.2e}, max: {v4.max():.2e}, mean: {v4.mean():.2e}, nonzero: {np.count_nonzero(v4)}')
gp_std = np.sqrt(np.clip(v4, 0, None))
print(f'std u (v=1.0) - min: {gp_std.min():.4f}, max: {gp_std.max():.4f}')
