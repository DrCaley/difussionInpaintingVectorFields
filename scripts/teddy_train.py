#!/usr/bin/env python3
"""
Train a Teddy-style network that predicts eddy segmentation maps
directly from sparse velocity observations (row 22 of the ocean domain).

Usage:
    PYTHONPATH=. python scripts/teddy_train.py [--epochs 100] [--batch-size 64] [--smoke]

Outputs saved to results/teddy_baseline/:
    teddy_best.pt          — best model checkpoint
    teddy_labels_train.pt  — cached GT eddy labels (training set)
    teddy_labels_val.pt    — cached GT eddy labels (validation set)
    training_log.txt       — per-epoch loss log
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

# Project imports
from ddpm.utils.eddy_detection import detect_eddies_gamma
from scripts.teddy_model import TeddyNet

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OCEAN_H, OCEAN_W = 44, 94
OBS_ROW = 22

# Eddy detection — same params as bulk_eval
EDDY_PARAMS = dict(
    radius=8,
    gamma_threshold=0.65,
    min_area=25,
    shore_buffer=2,
    smooth_sigma=2.0,
    min_mean_speed_ratio=0.3,
    min_vorticity=0.03,
)

OUT_DIR = Path("results/teddy_baseline")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class TeddyEddyDataset(Dataset):
    """
    Each sample: (obs, label, ocean_mask)
        obs         (W, 2)   — velocity at row 22
        label       (H, W)   — binary eddy map (1=eddy, 0=no-eddy)
        ocean_mask  (H, W)   — 1=ocean, 0=land
    """

    def __init__(self, vel_data: torch.Tensor, label_cache_path: str | None = None):
        """
        vel_data: (N, 2, H, W) float tensor in physical units, NaN→0 already applied.
        """
        super().__init__()
        self.vel = vel_data  # (N, 2, 44, 94)
        self.N = vel_data.shape[0]

        # Ocean mask (same for all samples)
        speed = (vel_data[0, 0] ** 2 + vel_data[0, 1] ** 2)
        self.ocean_mask = (speed > 1e-10).float()  # (H, W)

        # Precompute or load GT labels
        if label_cache_path and os.path.exists(label_cache_path):
            print(f"  Loading cached labels from {label_cache_path}")
            self.labels = torch.load(label_cache_path, map_location="cpu", weights_only=True)
        else:
            self.labels = self._generate_labels()
            if label_cache_path:
                os.makedirs(os.path.dirname(label_cache_path), exist_ok=True)
                torch.save(self.labels, label_cache_path)
                print(f"  Saved labels to {label_cache_path}")

        # Compute per-sample obs normalization stats (across the training set)
        obs_all = self.vel[:, :, OBS_ROW, :]  # (N, 2, W)
        self.obs_mean = obs_all.mean()
        self.obs_std = obs_all.std() + 1e-8

    def _generate_labels(self) -> torch.Tensor:
        """Run Gamma1 eddy detection on every sample to create GT label maps."""
        print(f"  Generating GT eddy labels for {self.N} samples...")
        labels = torch.zeros(self.N, OCEAN_H, OCEAN_W)
        ocean = self.ocean_mask.bool()
        n_with_eddy = 0
        t0 = time.time()

        for i in range(self.N):
            vel = torch.nan_to_num(self.vel[i], nan=0.0)  # (2, H, W)
            eddies, _, _ = detect_eddies_gamma(vel, ocean_mask=ocean, **EDDY_PARAMS)
            for e in eddies:
                if e.mask is not None:
                    labels[i][e.mask] = 1.0
            if len(eddies) > 0:
                n_with_eddy += 1

            if (i + 1) % 500 == 0 or i == self.N - 1:
                elapsed = time.time() - t0
                print(f"    [{i+1}/{self.N}] {elapsed:.0f}s "
                      f"({n_with_eddy} samples w/ eddy so far)")

        print(f"  Done. {n_with_eddy}/{self.N} samples have ≥1 eddy "
              f"({n_with_eddy/self.N*100:.1f}%)")
        eddy_pix = labels.sum() / self.N
        print(f"  Avg eddy pixels per sample: {eddy_pix:.1f} / {OCEAN_H*OCEAN_W}")
        return labels

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        vel = self.vel[idx]                      # (2, H, W)
        obs = vel[:, OBS_ROW, :].T               # (W, 2) — sequence for Transformer
        obs = (obs - self.obs_mean) / self.obs_std  # normalize
        label = self.labels[idx]                   # (H, W)
        return obs, label, self.ocean_mask


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def focal_loss(logits, targets, alpha=0.75, gamma=2.0):
    """Binary focal loss computed only on ocean pixels.
    alpha weights the positive (eddy) class."""
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    pt = torch.where(targets > 0.5, p, 1 - p)
    alpha_t = torch.where(targets > 0.5, alpha, 1 - alpha)
    loss = alpha_t * ((1 - pt) ** gamma) * bce
    return loss


def dice_loss(logits, targets, smooth=1.0):
    """Differentiable Dice loss."""
    p = torch.sigmoid(logits)
    intersection = (p * targets).sum()
    union = p.sum() + targets.sum()
    return 1 - (2 * intersection + smooth) / (union + smooth)


def combined_loss(logits, targets, ocean_mask):
    """Focal + Dice, masked to ocean pixels only."""
    # Flatten to ocean-only pixels
    mask = ocean_mask.bool()  # (B, H, W)
    logits_flat = logits.squeeze(1)[mask]
    targets_flat = targets[mask]

    fl = focal_loss(logits_flat, targets_flat).mean()
    dl = dice_loss(logits_flat, targets_flat)
    return fl + dl


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_pickle_data(path: str = "data.pickle"):
    """Load raw velocity data from pickle, return (train, val) as (N,2,H,W) tensors."""
    print(f"Loading data from {path}...")
    with open(path, "rb") as f:
        train_np, val_np, _test_np = pickle.load(f)

    def to_tensor(arr):
        # arr shape: (W=94, H=44, C=2, N) → (N, C, H, W)
        t = torch.from_numpy(np.ascontiguousarray(arr)).float()
        t = t.permute(3, 2, 1, 0)  # (N, 2, 44, 94)
        t = torch.nan_to_num(t, nan=0.0)
        return t

    train_t = to_tensor(train_np)
    val_t = to_tensor(val_np)
    print(f"  Train: {train_t.shape}, Val: {val_t.shape}")
    return train_t, val_t


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ----- device -----
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # ----- data -----
    train_vel, val_vel = load_pickle_data()

    train_ds = TeddyEddyDataset(
        train_vel,
        label_cache_path=str(OUT_DIR / "teddy_labels_train.pt"),
    )
    val_ds = TeddyEddyDataset(
        val_vel,
        label_cache_path=str(OUT_DIR / "teddy_labels_val.pt"),
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=False,
    )

    print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

    # ----- model -----
    model = TeddyNet(
        obs_dim=2, d_model=64, n_heads=4, n_enc_layers=3,
        n_cnn_layers=8, ocean_h=OCEAN_H, ocean_w=OCEAN_W, obs_row=OBS_ROW,
    ).to(device)
    print(f"Model parameters: {model.count_parameters():,}")

    # ----- optimizer -----
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # ----- training -----
    best_val_loss = float("inf")
    log_lines = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for obs, label, ocean in train_loader:
            obs = obs.to(device)
            label = label.to(device)
            ocean = ocean.to(device)

            logits = model(obs)  # (B, 1, H, W)
            loss = combined_loss(logits, label, ocean)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        train_avg = epoch_loss / max(n_batches, 1)

        # ----- validation -----
        model.eval()
        val_loss = 0.0
        vn = 0
        with torch.no_grad():
            for obs, label, ocean in val_loader:
                obs = obs.to(device)
                label = label.to(device)
                ocean = ocean.to(device)
                logits = model(obs)
                loss = combined_loss(logits, label, ocean)
                val_loss += loss.item()
                vn += 1
        val_avg = val_loss / max(vn, 1)

        line = (f"Epoch {epoch:3d}/{args.epochs}  "
                f"train_loss={train_avg:.4f}  val_loss={val_avg:.4f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}")
        print(line)
        log_lines.append(line)

        # save best
        if val_avg < best_val_loss:
            best_val_loss = val_avg
            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_loss": val_avg,
                "obs_mean": train_ds.obs_mean,
                "obs_std": train_ds.obs_std,
                "eddy_params": EDDY_PARAMS,
            }
            torch.save(ckpt, OUT_DIR / "teddy_best.pt")
            print(f"  ⟶ saved best (val_loss={val_avg:.4f})")

    # save log
    with open(OUT_DIR / "training_log.txt", "w") as f:
        f.write("\n".join(log_lines) + "\n")
    print(f"\nDone. Best val loss: {best_val_loss:.4f}")
    print(f"Outputs in {OUT_DIR}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train Teddy eddy-prediction baseline")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--smoke", action="store_true", help="3-epoch smoke test")
    args = parser.parse_args()

    if args.smoke:
        args.epochs = 3
        args.batch_size = 16
        print("=== SMOKE TEST (3 epochs) ===")

    train(args)


if __name__ == "__main__":
    main()
