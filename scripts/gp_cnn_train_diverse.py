#!/usr/bin/env python3
"""
Retrain GP-CNN with diverse random observation masks.

Instead of the fixed row-22 transect, each training sample gets a randomly
selected mask from a precomputed pool.  GP posterior mean and variance are
computed on-the-fly using vectorized linear algebra (the ~38-point GP for
1% masks is a trivial 38×38 system).

Key design:
  - Pool of M masks precomputed at init (default M=50)
  - For each mask: precompute GP weight matrix W and variance vector
  - At __getitem__: pick random mask, compute GP mean as W @ y (fast matmul)
  - GP std is normalized per-mask and passed as input channels
  - Sensor mask and ocean mask vary per sample (different masks)

Usage:
    PYTHONPATH=. python scripts/gp_cnn_train_diverse.py [--epochs 200] [--smoke]
    PYTHONPATH=. python scripts/gp_cnn_train_diverse.py --epochs 3 --smoke
"""

import argparse
import os
import pickle
import sys
import time
import random as python_random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from scripts.voronoi_cnn_model import VoronoiCNN

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OCEAN_H, OCEAN_W = 44, 94
FULL_H, FULL_W = 64, 128
OUT_DIR = Path("results/gp_cnn_diverse")

GP_LENGTHSCALE = 14.1
GP_VARIANCE = 0.0103420345
GP_NOISE = 1e-8


# ---------------------------------------------------------------------------
# Fast vectorized GP for small observation sets
# ---------------------------------------------------------------------------

class FastGP:
    """
    Vectorized GP posterior for a fixed mask geometry.
    
    For a mask with n_obs observation points on an n_ocean-pixel grid:
      - W matrix: (n_ocean, n_obs) — maps observations to posterior means
      - var: (n_ocean,) — posterior variance at each ocean pixel
    
    At query time: gp_mean = W @ y  (fast matmul, same for both u,v)
    """
    
    def __init__(self, obs_rows, obs_cols, all_rows, all_cols,
                 lengthscale=GP_LENGTHSCALE, variance=GP_VARIANCE,
                 noise=GP_NOISE):
        """
        Parameters
        ----------
        obs_rows, obs_cols : 1D arrays of observed pixel coordinates
        all_rows, all_cols : 1D arrays of ALL ocean pixel coordinates
        """
        self.obs_rows = obs_rows
        self.obs_cols = obs_cols
        n_obs = len(obs_rows)
        n_all = len(all_rows)
        
        # Build kernel matrices using RBF kernel
        # K(x, x') = variance * exp(-||x - x'||^2 / (2 * ls^2))
        obs_coords = np.stack([obs_rows, obs_cols], axis=1).astype(np.float64)  # (n_obs, 2)
        all_coords = np.stack([all_rows, all_cols], axis=1).astype(np.float64)  # (n_all, 2)
        
        # K(obs, obs) — n_obs × n_obs
        diff_obs = obs_coords[:, None, :] - obs_coords[None, :, :]  # (n_obs, n_obs, 2)
        K_obs = variance * np.exp(-0.5 * np.sum(diff_obs**2, axis=-1) / lengthscale**2)
        K_obs += noise * np.eye(n_obs)
        
        # K(all, obs) — n_all × n_obs
        diff_all_obs = all_coords[:, None, :] - obs_coords[None, :, :]  # (n_all, n_obs, 2)
        K_all_obs = variance * np.exp(-0.5 * np.sum(diff_all_obs**2, axis=-1) / lengthscale**2)
        
        # Solve: W = K(all, obs) @ inv(K(obs, obs) + noise*I)
        # Use Cholesky for numerical stability
        L = np.linalg.cholesky(K_obs)  # K_obs already has noise on diagonal
        alpha = np.linalg.solve(L, K_all_obs.T)  # L^{-1} K(obs, all).T
        self.W = np.linalg.solve(L.T, alpha).T.astype(np.float32)  # (n_all, n_obs)
        
        # Posterior variance: var_i = K(i,i) - K(i,obs) @ inv(K_obs) @ K(obs,i)
        # = variance - sum(W_i * K_all_obs_i)
        self.var = (variance - np.sum(self.W * K_all_obs.astype(np.float32), axis=1))
        self.var = np.clip(self.var, 0, None).astype(np.float32)  # (n_all,)
        
        # Precompute GP std, normalized to [0, 1]
        gp_std = np.sqrt(self.var)
        std_max = gp_std.max() + 1e-8
        self.gp_std_norm = (gp_std / std_max).astype(np.float32)  # (n_all,)
    
    def predict_mean(self, y_obs):
        """
        Compute GP posterior mean.
        
        Parameters
        ----------
        y_obs : (n_obs,) array — observed values at obs locations
        
        Returns
        -------
        mu : (n_all,) array — posterior mean at all ocean locations
        """
        return self.W @ y_obs


class MaskPool:
    """
    Pool of M random observation masks with precomputed GP solvers.
    """
    
    def __init__(self, ocean_mask, reveal_pct=1.0, n_masks=50, seed=42):
        self.ocean_mask = ocean_mask
        ocean_idx = np.argwhere(ocean_mask > 0.5)  # (n_ocean, 2)
        self.all_rows = ocean_idx[:, 0]
        self.all_cols = ocean_idx[:, 1]
        self.n_ocean = len(ocean_idx)
        
        # Build row,col→flat_index mapping for fast lookup
        self.rc_to_flat = {}
        for i, (r, c) in enumerate(ocean_idx):
            self.rc_to_flat[(r, c)] = i
        
        rng = np.random.default_rng(seed)
        n_reveal = max(1, round(self.n_ocean * reveal_pct / 100.0))
        
        print(f"  Building mask pool: {n_masks} masks × {n_reveal} obs ({reveal_pct}% of {self.n_ocean} ocean px)")
        
        self.masks = []  # list of dicts with GP solver + mask arrays
        t0 = time.time()
        
        for m in range(n_masks):
            chosen = rng.choice(self.n_ocean, size=n_reveal, replace=False)
            obs_rows = self.all_rows[chosen]
            obs_cols = self.all_cols[chosen]
            
            # Build observation mask (H, W)
            obs_mask = np.zeros((OCEAN_H, OCEAN_W), dtype=np.float32)
            obs_mask[obs_rows, obs_cols] = 1.0
            
            # Flat indices for obs in the all-ocean array
            obs_flat = chosen  # indices into all_rows/all_cols
            
            # Build fast GP solver
            gp = FastGP(obs_rows, obs_cols, self.all_rows, self.all_cols)
            
            # Build GP std map as (2, H, W) — same for both components
            gp_std_2d = np.zeros((OCEAN_H, OCEAN_W), dtype=np.float32)
            gp_std_2d[self.all_rows, self.all_cols] = gp.gp_std_norm
            gp_std_map = np.stack([gp_std_2d, gp_std_2d], axis=0)  # (2, H, W)
            
            self.masks.append({
                "obs_mask": obs_mask,       # (H, W)
                "obs_flat": obs_flat,       # indices into ocean array
                "gp": gp,                   # FastGP solver
                "gp_std_map": gp_std_map,   # (2, H, W) normalized std
            })
        
        elapsed = time.time() - t0
        print(f"  Mask pool built in {elapsed:.1f}s")
    
    def __len__(self):
        return len(self.masks)
    
    def get_gp_prediction(self, mask_idx, vel_ocean):
        """
        Compute GP posterior mean for a given mask and velocity field.
        
        Parameters
        ----------
        mask_idx : int — which mask to use
        vel_ocean : (2, n_ocean) — velocity values at all ocean pixels
        
        Returns
        -------
        gp_mean : (2, n_ocean) — GP posterior mean at all ocean pixels
        """
        md = self.masks[mask_idx]
        gp = md["gp"]
        obs_flat = md["obs_flat"]
        
        # Extract observed values
        y_u = vel_ocean[0, obs_flat]
        y_v = vel_ocean[1, obs_flat]
        
        # GP mean via fast matmul
        mu_u = gp.predict_mean(y_u)
        mu_v = gp.predict_mean(y_v)
        
        return np.stack([mu_u, mu_v], axis=0)  # (2, n_ocean)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DiverseGPDataset(Dataset):
    """
    GP-CNN training dataset with diverse random masks.
    
    Each sample randomly selects a mask from the pool and computes
    GP posterior mean on-the-fly using the precomputed GP solver.
    
    Returns: (gp_input, target_vel)
        gp_input   : (6, H, W) — [gp_mean_u, gp_mean_v, gp_std_u, gp_std_v, sensor_mask, ocean_mask]
        target_vel : (2, H, W) — ground truth velocity (normalized)
    """
    
    def __init__(self, vel_raw, vel_norm, ocean_mask, mask_pool,
                 norm_mean, norm_std, augment=False):
        """
        Parameters
        ----------
        vel_raw  : (N, 2, H, W) raw physical velocity
        vel_norm : (N, 2, H, W) normalized velocity (targets)
        ocean_mask : (H, W) float
        mask_pool : MaskPool
        norm_mean, norm_std : (2,) arrays for normalization
        augment : bool — apply random flips
        """
        self.N = vel_raw.shape[0]
        self.vel_raw = vel_raw.numpy() if isinstance(vel_raw, torch.Tensor) else vel_raw
        self.vel_norm = vel_norm
        self.ocean_mask = ocean_mask
        self.mask_pool = mask_pool
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.augment = augment
        
        # Pre-extract ocean pixel values for fast GP (avoid repeated indexing)
        ocean_bool = ocean_mask > 0.5
        self.vel_ocean = np.zeros((self.N, 2, mask_pool.n_ocean), dtype=np.float32)
        for i in range(self.N):
            self.vel_ocean[i, 0] = self.vel_raw[i, 0][ocean_bool]
            self.vel_ocean[i, 1] = self.vel_raw[i, 1][ocean_bool]
        
        # Constant channel
        self.ocean_ch = torch.from_numpy(ocean_mask.astype(np.float32)).unsqueeze(0)  # (1, H, W)
    
    def __len__(self):
        return self.N
    
    def __getitem__(self, idx):
        # Pick random mask
        mask_idx = python_random.randint(0, len(self.mask_pool) - 1)
        md = self.mask_pool.masks[mask_idx]
        
        # GP posterior mean for this sample's velocity under this mask
        gp_mean_ocean = self.mask_pool.get_gp_prediction(mask_idx, self.vel_ocean[idx])
        
        # Reshape to (2, H, W)
        gp_mean_2d = np.zeros((2, OCEAN_H, OCEAN_W), dtype=np.float32)
        gp_mean_2d[0][self.ocean_mask > 0.5] = gp_mean_ocean[0]
        gp_mean_2d[1][self.ocean_mask > 0.5] = gp_mean_ocean[1]
        
        # Normalize GP mean
        gp_mean_norm = (gp_mean_2d - self.norm_mean[:, None, None]) / self.norm_std[:, None, None]
        gp_mean_norm *= self.ocean_mask[None, :, :]
        
        target = self.vel_norm[idx]  # (2, H, W) tensor, already normalized
        
        gp_mean_t = torch.from_numpy(gp_mean_norm.astype(np.float32))
        gp_std_t = torch.from_numpy(md["gp_std_map"])
        sensor_t = torch.from_numpy(md["obs_mask"]).unsqueeze(0)  # (1, H, W)
        
        # Augmentation: random flips that preserve ∇·v = 0
        if self.augment and python_random.random() < 0.5:
            # Horizontal flip: negate u, flip cols
            gp_mean_t = torch.flip(gp_mean_t, [-1])
            gp_mean_t[0] = -gp_mean_t[0]
            gp_std_t = torch.from_numpy(np.flip(md["gp_std_map"], axis=-1).copy())
            sensor_t = torch.flip(sensor_t, [-1])
            target = torch.flip(target, [-1]).clone()
            target[0] = -target[0]
        
        if self.augment and python_random.random() < 0.5:
            # Vertical flip: negate v, flip rows
            gp_mean_t = torch.flip(gp_mean_t, [-2])
            gp_mean_t[1] = -gp_mean_t[1]
            gp_std_t = torch.flip(gp_std_t, [-2])
            sensor_t = torch.flip(sensor_t, [-2])
            target = torch.flip(target, [-2]).clone()
            target[1] = -target[1]
        
        # Stack: [gp_u, gp_v, std_u, std_v, sensor_mask, ocean_mask]
        gp_input = torch.cat([gp_mean_t, gp_std_t, sensor_t, self.ocean_ch], dim=0)
        
        return gp_input, target


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def masked_mse_loss(pred, target, ocean_mask):
    diff = (pred - target) ** 2
    if ocean_mask.dim() == 2:
        ocean_mask = ocean_mask.unsqueeze(0).unsqueeze(0)
    elif ocean_mask.dim() == 3:
        ocean_mask = ocean_mask.unsqueeze(0)
    masked = diff * ocean_mask
    return masked.sum() / (ocean_mask.sum() * pred.shape[1] + 1e-8)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_pickle_data(path="data.pickle"):
    print(f"Loading raw data from {path}...")
    with open(path, "rb") as f:
        train_np, val_np, test_np = pickle.load(f)
    
    def to_tensor(arr):
        t = torch.from_numpy(np.ascontiguousarray(arr)).float()
        t = t.permute(3, 2, 1, 0)
        t = torch.nan_to_num(t, nan=0.0)
        return t
    
    train_t = to_tensor(train_np)
    val_t = to_tensor(val_np)
    print(f"  Train: {train_t.shape}, Val: {val_t.shape}")
    return train_t, val_t


def compute_normalization(vel, ocean_mask):
    mask_t = torch.from_numpy(ocean_mask).bool()
    ocean_vals = vel[:, :, mask_t]
    mean = ocean_vals.mean(dim=(0, 2))
    std = ocean_vals.std(dim=(0, 2)) + 1e-8
    return mean, std


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")
    
    # Load data
    train_vel, val_vel = load_pickle_data()
    
    # Ocean mask
    speed = (train_vel[0, 0] ** 2 + train_vel[0, 1] ** 2).numpy()
    ocean_mask = (speed > 1e-10).astype(np.float32)
    print(f"  Ocean pixels: {ocean_mask.sum():.0f} / {OCEAN_H * OCEAN_W}")
    
    # Normalization
    mean, std = compute_normalization(train_vel, ocean_mask)
    norm_mean = mean.numpy()
    norm_std = std.numpy()
    print(f"  Normalization — mean: [{norm_mean[0]:.4f}, {norm_mean[1]:.4f}], "
          f"std: [{norm_std[0]:.4f}, {norm_std[1]:.4f}]")
    
    mean_4d = mean.view(1, 2, 1, 1)
    std_4d = std.view(1, 2, 1, 1)
    ocean_t = torch.from_numpy(ocean_mask).unsqueeze(0).unsqueeze(0)
    
    train_norm = ((train_vel - mean_4d) / std_4d) * ocean_t
    val_norm = ((val_vel - mean_4d) / std_4d) * ocean_t
    
    # Build mask pools
    print("\nBuilding training mask pool...")
    train_pool = MaskPool(ocean_mask, reveal_pct=args.reveal_pct,
                          n_masks=args.n_masks, seed=42)
    print("Building validation mask pool...")
    val_pool = MaskPool(ocean_mask, reveal_pct=args.reveal_pct,
                        n_masks=args.n_val_masks, seed=9999)
    
    # Datasets
    train_ds = DiverseGPDataset(
        train_vel, train_norm, ocean_mask, train_pool,
        norm_mean, norm_std, augment=args.augment,
    )
    val_ds = DiverseGPDataset(
        val_vel, val_norm, ocean_mask, val_pool,
        norm_mean, norm_std, augment=False,
    )
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=False)
    print(f"\nTrain: {len(train_ds)}, Val: {len(val_ds)}")
    
    # Model — same architecture as original GP-CNN
    model = VoronoiCNN(in_channels=6, out_channels=2,
                       base_ch=args.base_ch, depth=args.depth).to(device)
    n_params = model.count_parameters()
    print(f"Model: GP-CNN-Diverse (U-Net-lite), {n_params:,} parameters")
    print(f"  in_channels=6, base_ch={args.base_ch}, depth={args.depth}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    
    ocean_mask_t = torch.from_numpy(ocean_mask).to(device)
    
    start_epoch = 1
    best_val_loss = float("inf")
    patience_counter = 0
    log_lines = ["epoch,train_loss,val_loss,lr,epoch_time"]
    
    if args.resume:
        ckpt_path = OUT_DIR / "gp_cnn_diverse_best.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model_state"])
            model.to(device)
            best_val_loss = ckpt["val_loss"]
            start_epoch = ckpt["epoch"] + 1
            for _ in range(ckpt["epoch"]):
                scheduler.step()
            print(f"Resumed from epoch {ckpt['epoch']}, best_val={best_val_loss:.6f}")
        else:
            print("No checkpoint found — starting fresh")
    
    print(f"\n{'='*60}")
    print(f"Starting training: {args.epochs} epochs, batch_size={args.batch_size}")
    print(f"Mask: {args.reveal_pct}% random, pool size={args.n_masks}")
    print(f"{'='*60}\n")
    
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        
        for gp_in, target in train_loader:
            gp_in = gp_in.to(device)
            target = target.to(device)
            
            pred = model(gp_in)
            loss = masked_mse_loss(pred, target, ocean_mask_t)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        train_avg = epoch_loss / max(n_batches, 1)
        
        # Validation
        model.eval()
        val_loss = 0.0
        vn = 0
        with torch.no_grad():
            for gp_in, target in val_loader:
                gp_in = gp_in.to(device)
                target = target.to(device)
                pred = model(gp_in)
                loss = masked_mse_loss(pred, target, ocean_mask_t)
                val_loss += loss.item()
                vn += 1
        val_avg = val_loss / max(vn, 1)
        
        lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0
        line = (f"Epoch {epoch:3d}/{args.epochs}  "
                f"train={train_avg:.6f}  val={val_avg:.6f}  "
                f"lr={lr:.2e}  ({elapsed:.1f}s)")
        print(line)
        log_lines.append(f"{epoch},{train_avg:.6f},{val_avg:.6f},{lr:.2e},{elapsed:.1f}")
        
        if val_avg < best_val_loss:
            best_val_loss = val_avg
            patience_counter = 0
            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_loss": val_avg,
                "train_loss": train_avg,
                "norm_mean": mean,
                "norm_std": std,
                "ocean_mask": ocean_mask,
                "gp_params": {
                    "lengthscale": GP_LENGTHSCALE,
                    "variance": GP_VARIANCE,
                    "noise": GP_NOISE,
                },
                "model_config": {
                    "in_channels": 6,
                    "out_channels": 2,
                    "base_ch": args.base_ch,
                    "depth": args.depth,
                    "model_type": "gp_cnn_diverse",
                    "reveal_pct": args.reveal_pct,
                    "n_masks": args.n_masks,
                },
            }
            torch.save(ckpt, OUT_DIR / "gp_cnn_diverse_best.pt")
            print(f"  → saved best (val={val_avg:.6f})")
        else:
            patience_counter += 1
        
        if patience_counter >= args.patience:
            print(f"  Early stopping at epoch {epoch} (patience={args.patience})")
            break
        
        if epoch % 10 == 0 or epoch == args.epochs:
            with open(OUT_DIR / "training_log.csv", "w") as f:
                f.write("\n".join(log_lines) + "\n")
    
    with open(OUT_DIR / "training_log.csv", "w") as f:
        f.write("\n".join(log_lines) + "\n")
    
    print(f"\nDone. Best val loss: {best_val_loss:.6f}")
    print(f"Outputs in {OUT_DIR}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train GP-CNN with diverse random masks"
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base-ch", type=int, default=32)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--reveal-pct", type=float, default=1.0,
                        help="Percentage of ocean pixels to reveal per mask (default: 1%%)")
    parser.add_argument("--n-masks", type=int, default=50,
                        help="Number of random masks in the training pool")
    parser.add_argument("--n-val-masks", type=int, default=20,
                        help="Number of random masks for validation pool")
    parser.add_argument("--augment", action="store_true",
                        help="Enable divergence-preserving flip augmentation")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    
    if args.smoke:
        args.epochs = 3
        args.batch_size = 16
        args.n_masks = 5
        args.n_val_masks = 3
        print("=== SMOKE TEST (3 epochs, 5 masks) ===")
    
    train(args)


if __name__ == "__main__":
    main()
