#!/usr/bin/env python3
"""Diagnose GP-CNN data alignment."""
import torch, numpy as np, pickle, sys
sys.path.insert(0, '.')

# Load raw data
with open('data.pickle', 'rb') as f:
    train_np, val_np, _ = pickle.load(f)

def to_tensor(arr):
    t = torch.from_numpy(np.ascontiguousarray(arr)).float()
    t = t.permute(3, 2, 1, 0)
    return torch.nan_to_num(t, nan=0.0)

train_vel = to_tensor(train_np)
val_vel = to_tensor(val_np)

# Ocean mask
speed = (train_vel[0, 0]**2 + train_vel[0, 1]**2).numpy()
ocean_mask = (speed > 1e-10).astype(np.float32)
mask_t = torch.from_numpy(ocean_mask).bool()

# Normalization
ocean_vals = train_vel[:, :, mask_t]
mean = ocean_vals.mean(dim=(0, 2))
std = ocean_vals.std(dim=(0, 2)) + 1e-8
mean_4d = mean.view(1, 2, 1, 1)
std_4d = std.view(1, 2, 1, 1)
ocean_t = torch.from_numpy(ocean_mask).unsqueeze(0).unsqueeze(0)

val_norm = ((val_vel - mean_4d) / std_4d) * ocean_t

# Load GP
gp = torch.load('data/rams_head/gp_precomputed.pt', map_location='cpu', weights_only=False)
gp_val = gp['gp_test'][:, :, :44, :94]
gp_norm = ((gp_val - mean_4d) / std_4d) * ocean_t

# Check alignment
print('Val  [0] at row 22, col 50:', val_vel[0, :, 22, 50].tolist())
print('GP   [0] at row 22, col 50:', gp_val[0, :, 22, 50].tolist())
print('Diff at obs row 22 col 50:', (gp_val[0, :, 22, 50] - val_vel[0, :, 22, 50]).abs().tolist())

# Avg diff at observed row vs unobserved
obs_diff = (gp_val[:, :, 22, :] - val_vel[:, :, 22, :]).abs().mean()
far_diff = (gp_val[:, :, 0, :] - val_vel[:, :, 0, :]).abs().mean()
print(f'\nAvg abs diff at row 22 (observed): {obs_diff:.6f}')
print(f'Avg abs diff at row 0 (far from obs): {far_diff:.6f}')

# GP baseline loss (identity model)
diff = (gp_norm - val_norm) ** 2
# Use same loss function as training: sum / (ocean_count * channels)
n_ocean = mask_t.sum().item()
gp_loss_train_style = diff[:, :, mask_t].sum() / (n_ocean * 2)
gp_loss_per_sample = diff[:, :, mask_t].sum(dim=1).mean() / (n_ocean * 2)
gp_loss_mean = diff[:, :, mask_t].mean()
print(f'\nGP baseline normalized MSE (per-pixel mean): {gp_loss_mean:.4f}')
print(f'  This is what the model gets if it outputs GP mean unchanged')

# Zeros baseline
zeros_mse = val_norm[:, :, mask_t].pow(2).mean()
print(f'Zeros baseline normalized MSE: {zeros_mse:.4f}')

# Also check: masked_mse_loss exactly as in training
oc = torch.from_numpy(ocean_mask).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
# For one batch
batch_gp = gp_norm[:32]
batch_tgt = val_norm[:32]
diff_b = (batch_gp - batch_tgt) ** 2
masked_b = diff_b * oc
loss_b = masked_b.sum() / (oc.sum() * 2 + 1e-8)
print(f'masked_mse_loss(GP, target) for first 32 val samples: {loss_b:.4f}')

# Check shapes
print(f'\nShapes check:')
print(f'  gp_norm: {gp_norm.shape}, val_norm: {val_norm.shape}')
print(f'  gp_norm range: [{gp_norm.min():.3f}, {gp_norm.max():.3f}]')
print(f'  val_norm range: [{val_norm.min():.3f}, {val_norm.max():.3f}]')
