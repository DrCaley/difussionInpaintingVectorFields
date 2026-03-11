#!/usr/bin/env python3
"""Quick diagnostic: compare V-CNN vs Voronoi using RAW ground truth from data.pickle."""
import pickle
import numpy as np
import torch
from scipy.spatial import cKDTree
from scripts.voronoi_cnn_model import VoronoiCNN, build_voronoi_input

OCEAN_H, OCEAN_W = 44, 94
NM = np.array([-0.06929559429949586, -0.0323937796117541], dtype=np.float32)
NS = np.array([0.1358005549716049, 0.08899177232117582], dtype=np.float32)

# Load raw data
print("Loading data.pickle...")
with open("data.pickle", "rb") as f:
    train_raw, val_raw, test_raw = pickle.load(f)

# Shape: (94, 44, 2, N) -> (N, 2, 44, 94)
test = np.transpose(test_raw, (3, 2, 1, 0)).astype(np.float32)
test = np.nan_to_num(test, nan=0.0)
print(f"test: {test.shape} range=[{test.min():.6f}, {test.max():.6f}]")

# Load GP for comparison
gp = torch.load("data/rams_head/gp_precomputed.pt", map_location="cpu", weights_only=False)
gp_t = gp["gp_test"][:, :, :OCEAN_H, :OCEAN_W].numpy()
print(f"gp_test: {gp_t.shape} range=[{gp_t.min():.6f}, {gp_t.max():.6f}]")

# Compare sample 0
diff = test[0] - gp_t[0]
print(f"\nSample 0: raw vs GP MSE = {(diff**2).mean():.8f}")
print(f"  raw nonzero pixels = {np.count_nonzero(test[0])}")
print(f"  GP  nonzero pixels = {np.count_nonzero(gp_t[0])}")

# Ocean mask from raw data
ocean_mask = (np.abs(test[0]).sum(axis=0) > 1e-7).astype(np.float32)
n_ocean = int(ocean_mask.sum())
print(f"\nOcean cells: {n_ocean}")

# Load V-CNN
ck = torch.load("results/voronoi_cnn/voronoi_cnn_best.pt", map_location="cpu", weights_only=False)
vcnn = VoronoiCNN(**ck["model_config"])
vcnn.load_state_dict(ck["model_state"])
vcnn.eval()
print(f"V-CNN loaded: {sum(p.numel() for p in vcnn.parameters()):,} params")

# Test at multiple coverage levels
for pct in [0.5, 1.0, 2.0, 5.0]:
    vor_mses_raw, vcnn_mses_raw = [], []
    vor_mses_gp, vcnn_mses_gp = [], []
    n_samples = 10

    for si in range(n_samples):
        gt_raw = test[si]
        gt_gp = gp_t[si]
        ocean_b = (np.abs(gt_raw).sum(axis=0) > 1e-7).astype(bool)

        # Random obs mask
        rng = np.random.default_rng(seed=42 + si)
        idx = np.argwhere(ocean_b)
        n_obs = max(1, round(len(idx) * pct / 100.0))
        sel = rng.choice(len(idx), size=n_obs, replace=False)
        obs_mask = np.zeros((OCEAN_H, OCEAN_W), dtype=np.float32)
        for j in sel:
            obs_mask[idx[j][0], idx[j][1]] = 1.0

        vel_obs = gt_raw * obs_mask[None]
        om = ocean_b.astype(np.float32)

        # Voronoi
        ky, kx = np.where(obs_mask > 0.5)
        tree = cKDTree(np.stack([ky, kx], 1))
        gy, gx = np.mgrid[0:OCEAN_H, 0:OCEAN_W]
        _, nn_idx = tree.query(np.stack([gy.ravel(), gx.ravel()], 1))
        nn_idx = nn_idx.reshape(OCEAN_H, OCEAN_W)
        vor = np.stack([vel_obs[0, ky, kx][nn_idx], vel_obs[1, ky, kx][nn_idx]], 0) * om

        # V-CNN
        vel_n = ((vel_obs - NM[:, None, None]) / NS[:, None, None]) * om[None]
        vi = build_voronoi_input(vel_n, obs_mask, om)
        with torch.no_grad():
            p = vcnn(torch.from_numpy(vi).unsqueeze(0))
        vcnn_phys = (p[0].numpy() * NS[:, None, None] + NM[:, None, None]) * om[None]

        # MSE vs raw GT
        vor_mses_raw.append(float(((vor[:, ocean_b] - gt_raw[:, ocean_b])**2).mean()))
        vcnn_mses_raw.append(float(((vcnn_phys[:, ocean_b] - gt_raw[:, ocean_b])**2).mean()))
        # MSE vs GP GT
        vor_mses_gp.append(float(((vor[:, ocean_b] - gt_gp[:, ocean_b])**2).mean()))
        vcnn_mses_gp.append(float(((vcnn_phys[:, ocean_b] - gt_gp[:, ocean_b])**2).mean()))

    n_obs_approx = max(1, round(n_ocean * pct / 100.0))
    vor_raw = np.mean(vor_mses_raw)
    vcnn_raw = np.mean(vcnn_mses_raw)
    vor_gp = np.mean(vor_mses_gp)
    vcnn_gp = np.mean(vcnn_mses_gp)
    print(f"\n--- {pct}% coverage ({n_obs_approx} obs) ---")
    print(f"  vs RAW GT:  Voronoi={vor_raw:.6f}  V-CNN={vcnn_raw:.6f}  ratio={vcnn_raw/vor_raw:.3f}x")
    print(f"  vs GP GT:   Voronoi={vor_gp:.6f}  V-CNN={vcnn_gp:.6f}  ratio={vcnn_gp/vor_gp:.3f}x")

print("\nDone.")
