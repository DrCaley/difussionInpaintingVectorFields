#!/usr/bin/env python3
"""
Train a Voronoi-CNN baseline for sparse velocity field reconstruction.

Based on: Fukami et al. (2021), "Global field reconstruction from sparse
sensors with Voronoi tessellation-assisted deep learning", Nature Machine
Intelligence.

The model learns to reconstruct full 2D ocean velocity fields from sparse
observations.  During training, random mask patterns are used for
augmentation.  At inference time the fixed row-22 mask is applied.

Usage:
    PYTHONPATH=. python scripts/voronoi_cnn_train.py [--epochs 200] [--smoke]

Outputs saved to results/voronoi_cnn/:
    voronoi_cnn_best.pt    — best model checkpoint
    training_log.csv       — per-epoch metrics
"""

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from scripts.voronoi_cnn_model import VoronoiCNN, build_voronoi_input

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OCEAN_H, OCEAN_W = 44, 94
OBS_ROW = 22
OUT_DIR = Path("results/voronoi_cnn")


# ---------------------------------------------------------------------------
# Mask generation for training augmentation
# ---------------------------------------------------------------------------

def make_row22_mask() -> np.ndarray:
    """Fixed single-row-22 mask. Returns (H, W), 1=known, 0=missing."""
    mask = np.zeros((OCEAN_H, OCEAN_W), dtype=np.float32)
    mask[OBS_ROW, :] = 1.0
    return mask


def make_random_rows_mask(n_rows: int = None) -> np.ndarray:
    """Random horizontal rows. Returns (H, W), 1=known, 0=missing."""
    if n_rows is None:
        n_rows = np.random.randint(1, 5)  # 1-4 rows
    rows = np.random.choice(OCEAN_H, size=n_rows, replace=False)
    mask = np.zeros((OCEAN_H, OCEAN_W), dtype=np.float32)
    mask[rows, :] = 1.0
    return mask


def make_sparse_points_mask(frac: float = None) -> np.ndarray:
    """Random sparse points. Returns (H, W), 1=known, 0=missing."""
    if frac is None:
        frac = np.random.uniform(0.01, 0.10)  # 1-10% coverage
    mask = np.zeros((OCEAN_H, OCEAN_W), dtype=np.float32)
    n_pts = max(1, int(frac * OCEAN_H * OCEAN_W))
    idx = np.random.choice(OCEAN_H * OCEAN_W, size=n_pts, replace=False)
    mask.flat[idx] = 1.0
    return mask


def make_random_walk_mask(ocean_mask: np.ndarray,
                          n_steps: int = None) -> np.ndarray:
    """Random walk path through the ocean domain (robot-style exploration).
    Returns (H, W), 1=known, 0=missing."""
    if n_steps is None:
        # Walk covers ~1-8% of ocean pixels
        n_ocean = int(ocean_mask.sum())
        n_steps = np.random.randint(max(10, n_ocean // 100),
                                     max(20, n_ocean // 12))
    mask = np.zeros((OCEAN_H, OCEAN_W), dtype=np.float32)
    # Start at random ocean pixel
    ocean_ys, ocean_xs = np.where(ocean_mask > 0.5)
    start_idx = np.random.randint(len(ocean_ys))
    y, x = int(ocean_ys[start_idx]), int(ocean_xs[start_idx])
    mask[y, x] = 1.0

    for _ in range(n_steps):
        # Random step in 8-connected neighbourhood
        dy, dx = np.random.randint(-1, 2), np.random.randint(-1, 2)
        ny, nx = np.clip(y + dy, 0, OCEAN_H - 1), np.clip(x + dx, 0, OCEAN_W - 1)
        if ocean_mask[ny, nx] > 0.5:
            y, x = ny, nx
        mask[y, x] = 1.0
    return mask


def make_random_transect_mask(ocean_mask: np.ndarray) -> np.ndarray:
    """Random straight-line transect(s) at random angles through the domain.
    Returns (H, W), 1=known, 0=missing."""
    from skimage.draw import line as sk_line
    mask = np.zeros((OCEAN_H, OCEAN_W), dtype=np.float32)
    n_lines = np.random.randint(1, 4)  # 1-3 transects
    for _ in range(n_lines):
        # Random start and end on domain boundary
        side1 = np.random.randint(4)
        if side1 == 0:    # top
            r0, c0 = 0, np.random.randint(OCEAN_W)
        elif side1 == 1:  # bottom
            r0, c0 = OCEAN_H - 1, np.random.randint(OCEAN_W)
        elif side1 == 2:  # left
            r0, c0 = np.random.randint(OCEAN_H), 0
        else:             # right
            r0, c0 = np.random.randint(OCEAN_H), OCEAN_W - 1
        side2 = np.random.randint(4)
        if side2 == 0:
            r1, c1 = 0, np.random.randint(OCEAN_W)
        elif side2 == 1:
            r1, c1 = OCEAN_H - 1, np.random.randint(OCEAN_W)
        elif side2 == 2:
            r1, c1 = np.random.randint(OCEAN_H), 0
        else:
            r1, c1 = np.random.randint(OCEAN_H), OCEAN_W - 1
        rr, cc = sk_line(r0, c0, r1, c1)
        valid = (rr >= 0) & (rr < OCEAN_H) & (cc >= 0) & (cc < OCEAN_W)
        mask[rr[valid], cc[valid]] = 1.0
    return mask * ocean_mask


def make_training_mask(ocean_mask: np.ndarray) -> np.ndarray:
    """
    Sample a random observation mask for training augmentation.
    NO row-22 bias — the robot path varies every deployment.
    Returns (H, W) with 1=known, 0=missing, intersected with ocean_mask.

    Distribution (all random, no fixed patterns):
      30%  random walk  (robot exploration path)
      25%  random transect lines (1-3 straight-line passes)
      25%  random 1-4 horizontal rows
      20%  sparse random points (1-10% coverage)
    """
    r = np.random.random()
    if r < 0.30:
        # Random walk (robot-style exploration)
        mask = make_random_walk_mask(ocean_mask)
    elif r < 0.55:
        # Random straight-line transects
        mask = make_random_transect_mask(ocean_mask)
    elif r < 0.80:
        # Random 1-4 horizontal rows
        mask = make_random_rows_mask()
    else:
        # Sparse random points (1-10%)
        mask = make_sparse_points_mask()

    # Intersect with ocean
    mask = mask * ocean_mask
    return mask


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class VoronoiReconDataset(Dataset):
    """
    Each sample: (voronoi_input, target_vel)
        voronoi_input : (5, H, W) — Voronoi tessellation + metadata
        target_vel    : (2, H, W) — ground truth velocity field
    """

    def __init__(self, vel_data: torch.Tensor, ocean_mask: np.ndarray,
                 augment: bool = True, fixed_mask: np.ndarray = None):
        """
        vel_data:    (N, 2, H, W) float tensor, NaN→0 applied.
        ocean_mask:  (H, W) numpy, 1=ocean, 0=land.
        augment:     If True, sample random masks each epoch.
        fixed_mask:  If provided and augment=False, use this fixed mask.
        """
        super().__init__()
        self.vel = vel_data.numpy()  # keep as numpy for Voronoi
        self.vel_t = vel_data        # keep tensor for targets
        self.N = vel_data.shape[0]
        self.ocean_mask = ocean_mask
        self.augment = augment
        self.fixed_mask = fixed_mask if fixed_mask is not None else make_row22_mask()

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        vel = self.vel[idx]  # (2, H, W) numpy
        target = self.vel_t[idx]  # (2, H, W) tensor

        if self.augment:
            mask = make_training_mask(self.ocean_mask)
        else:
            mask = self.fixed_mask

        voronoi_in = build_voronoi_input(vel, mask, self.ocean_mask)
        voronoi_in = torch.from_numpy(voronoi_in)

        return voronoi_in, target


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor,
                    ocean_mask: torch.Tensor) -> torch.Tensor:
    """MSE computed only over ocean pixels."""
    # ocean_mask: (1, H, W) or (H, W) — broadcast over batch & channels
    diff = (pred - target) ** 2  # (B, 2, H, W)
    if ocean_mask.dim() == 2:
        ocean_mask = ocean_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    elif ocean_mask.dim() == 3:
        ocean_mask = ocean_mask.unsqueeze(0)  # (1, 1, H, W)
    masked = diff * ocean_mask
    return masked.sum() / (ocean_mask.sum() * pred.shape[1] + 1e-8)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_pickle_data(path: str = "data.pickle"):
    """Load raw velocity data → (train, val) as (N, 2, H, W) tensors."""
    print(f"Loading data from {path}...")
    with open(path, "rb") as f:
        train_np, val_np, _test_np = pickle.load(f)

    def to_tensor(arr):
        t = torch.from_numpy(np.ascontiguousarray(arr)).float()
        t = t.permute(3, 2, 1, 0)  # (W,H,C,N) → (N,C,H,W) = (N,2,44,94)
        t = torch.nan_to_num(t, nan=0.0)
        return t

    train_t = to_tensor(train_np)
    val_t = to_tensor(val_np)
    print(f"  Train: {train_t.shape}, Val: {val_t.shape}")
    return train_t, val_t


def compute_normalization(train_vel: torch.Tensor, ocean_mask: np.ndarray):
    """Compute per-channel mean/std over ocean pixels of training set."""
    mask_t = torch.from_numpy(ocean_mask).bool()  # (H, W)
    # train_vel: (N, 2, H, W)
    ocean_vals = train_vel[:, :, mask_t]  # (N, 2, n_ocean)
    mean = ocean_vals.mean(dim=(0, 2))  # (2,)
    std = ocean_vals.std(dim=(0, 2)) + 1e-8  # (2,)
    return mean, std


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Data
    train_vel, val_vel = load_pickle_data()

    # Ocean mask from first sample
    speed = (train_vel[0, 0] ** 2 + train_vel[0, 1] ** 2).numpy()
    ocean_mask = (speed > 1e-10).astype(np.float32)
    print(f"  Ocean pixels: {ocean_mask.sum():.0f} / {OCEAN_H * OCEAN_W}")

    # Normalisation (per-channel z-score)
    mean, std = compute_normalization(train_vel, ocean_mask)
    print(f"  Normalization — mean: [{mean[0]:.4f}, {mean[1]:.4f}], "
          f"std: [{std[0]:.4f}, {std[1]:.4f}]")

    # Apply normalisation to velocity data
    mean_4d = mean.view(1, 2, 1, 1)
    std_4d = std.view(1, 2, 1, 1)
    train_vel_n = (train_vel - mean_4d) / std_4d
    val_vel_n = (val_vel - mean_4d) / std_4d

    # Zero out land in normalised data
    ocean_t = torch.from_numpy(ocean_mask).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    train_vel_n = train_vel_n * ocean_t
    val_vel_n = val_vel_n * ocean_t

    # Fixed row-22 mask for validation
    row22_mask = make_row22_mask() * ocean_mask

    # Datasets
    train_ds = VoronoiReconDataset(train_vel_n, ocean_mask, augment=True)
    val_ds = VoronoiReconDataset(val_vel_n, ocean_mask, augment=False,
                                  fixed_mask=row22_mask)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=False)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    # Model
    model = VoronoiCNN(in_channels=5, out_channels=2,
                       base_ch=args.base_ch, depth=args.depth).to(device)
    print(f"Model parameters: {model.count_parameters():,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    ocean_mask_t = torch.from_numpy(ocean_mask).to(device)  # (H, W)

    # Resume from checkpoint if requested
    start_epoch = 1
    best_val_loss = float("inf")
    patience_counter = 0
    log_lines = ["epoch,train_loss,val_loss,lr"]

    if args.resume:
        ckpt_path = OUT_DIR / "voronoi_cnn_best.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model_state"])
            model.to(device)
            best_val_loss = ckpt["val_loss"]
            start_epoch = ckpt["epoch"] + 1
            # Advance scheduler to the right position
            for _ in range(ckpt["epoch"]):
                scheduler.step()
            print(f"Resumed from epoch {ckpt['epoch']}, "
                  f"best_val={best_val_loss:.6f}")
        else:
            print("No checkpoint found — starting fresh")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for voronoi_in, target in train_loader:
            voronoi_in = voronoi_in.to(device)
            target = target.to(device)

            pred = model(voronoi_in)
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
            for voronoi_in, target in val_loader:
                voronoi_in = voronoi_in.to(device)
                target = target.to(device)
                pred = model(voronoi_in)
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
        log_lines.append(f"{epoch},{train_avg:.6f},{val_avg:.6f},{lr:.2e}")

        # Save best
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
                "model_config": {
                    "in_channels": 5,
                    "out_channels": 2,
                    "base_ch": args.base_ch,
                    "depth": args.depth,
                },
            }
            torch.save(ckpt, OUT_DIR / "voronoi_cnn_best.pt")
            print(f"  → saved best (val={val_avg:.6f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= args.patience:
            print(f"  Early stopping at epoch {epoch} (patience={args.patience})")
            break

    # Save log
    with open(OUT_DIR / "training_log.csv", "w") as f:
        f.write("\n".join(log_lines) + "\n")

    print(f"\nDone. Best val loss: {best_val_loss:.6f}")
    print(f"Outputs in {OUT_DIR}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train Voronoi-CNN velocity reconstruction baseline"
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base-ch", type=int, default=32,
                        help="Base channel width for U-Net encoder")
    parser.add_argument("--depth", type=int, default=3,
                        help="Number of encoder/decoder levels")
    parser.add_argument("--patience", type=int, default=30,
                        help="Early stopping patience")
    parser.add_argument("--smoke", action="store_true",
                        help="3-epoch smoke test")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from best checkpoint")
    args = parser.parse_args()

    if args.smoke:
        args.epochs = 3
        args.batch_size = 16
        print("=== SMOKE TEST (3 epochs) ===")

    train(args)


if __name__ == "__main__":
    main()
