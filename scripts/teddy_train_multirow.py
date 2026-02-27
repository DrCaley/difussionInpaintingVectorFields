#!/usr/bin/env python3
"""
Train a multi-row Teddy network that predicts eddy segmentation maps
from sparse velocity observations at multiple rows.

Coverage variants:
  --obs-rows 22                → 1 row  (original, ~2.4% known)
  --obs-rows 4,13,22,31,40    → 5 rows (~11% known, like multi-track)
  --obs-rows all               → all 44 rows (100%, upper-bound sanity check)

Usage:
    PYTHONPATH=. python scripts/teddy_train_multirow.py --obs-rows all --tag full --epochs 200
    PYTHONPATH=. python scripts/teddy_train_multirow.py --obs-rows 4,13,22,31,40 --tag 5row --epochs 200
    PYTHONPATH=. python scripts/teddy_train_multirow.py --obs-rows 22 --tag 1row --epochs 200

Outputs saved to results/teddy_<tag>/
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

from ddpm.utils.eddy_detection import detect_eddies_gamma
from scripts.teddy_model_multirow import TeddyNetMultiRow

# ---------------------------------------------------------------------------
OCEAN_H, OCEAN_W = 44, 94

EDDY_PARAMS = dict(
    radius=8, gamma_threshold=0.65, min_area=25, shore_buffer=2,
    smooth_sigma=2.0, min_mean_speed_ratio=0.3, min_vorticity=0.03,
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class TeddyMultiRowDataset(Dataset):
    """
    Each sample:
        obs       (N_obs, 2)  — velocities at observed pixels
        rows      (N_obs,)    — row indices
        cols      (N_obs,)    — col indices
        label     (H, W)      — binary eddy map
        ocean     (H, W)      — ocean mask
    """

    def __init__(self, vel_data: torch.Tensor, obs_rows: list[int],
                 label_cache_path: str | None = None):
        super().__init__()
        self.vel = vel_data          # (N, 2, H, W)
        self.N = vel_data.shape[0]
        self.obs_rows = sorted(obs_rows)

        # Ocean mask
        speed = vel_data[0, 0] ** 2 + vel_data[0, 1] ** 2
        self.ocean_mask = (speed > 1e-10).float()  # (H, W)

        # Build observation indices (same for all samples)
        row_list, col_list = [], []
        for r in self.obs_rows:
            for c in range(OCEAN_W):
                if self.ocean_mask[r, c] > 0.5:
                    row_list.append(r)
                    col_list.append(c)
        self.obs_row_idx = torch.tensor(row_list, dtype=torch.float32)
        self.obs_col_idx = torch.tensor(col_list, dtype=torch.float32)
        self.n_obs = len(row_list)
        coverage = self.n_obs / self.ocean_mask.sum().item() * 100
        print(f"  Observation rows: {self.obs_rows}")
        print(f"  Obs pixels/sample: {self.n_obs}  "
              f"({coverage:.1f}% of ocean)")

        # Precompute or load GT labels (shared with single-row version)
        if label_cache_path and os.path.exists(label_cache_path):
            print(f"  Loading cached labels from {label_cache_path}")
            self.labels = torch.load(label_cache_path, map_location="cpu",
                                     weights_only=True)
        else:
            self.labels = self._generate_labels()
            if label_cache_path:
                os.makedirs(os.path.dirname(label_cache_path), exist_ok=True)
                torch.save(self.labels, label_cache_path)
                print(f"  Saved labels to {label_cache_path}")

        # Normalisation stats over all obs pixels in training set
        obs_vals = []
        for r in self.obs_rows:
            obs_vals.append(self.vel[:, :, r, :])  # (N, 2, W)
        obs_all = torch.cat(obs_vals, dim=2)  # (N, 2, total_cols)
        self.obs_mean = obs_all.mean()
        self.obs_std = obs_all.std() + 1e-8

    def _generate_labels(self) -> torch.Tensor:
        print(f"  Generating GT eddy labels for {self.N} samples...")
        labels = torch.zeros(self.N, OCEAN_H, OCEAN_W)
        ocean = self.ocean_mask.bool()
        n_eddy = 0
        t0 = time.time()
        for i in range(self.N):
            vel = torch.nan_to_num(self.vel[i], nan=0.0)
            eddies, _, _ = detect_eddies_gamma(vel, ocean_mask=ocean,
                                                **EDDY_PARAMS)
            for e in eddies:
                if e.mask is not None:
                    labels[i][e.mask] = 1.0
            if eddies:
                n_eddy += 1
            if (i + 1) % 500 == 0 or i == self.N - 1:
                print(f"    [{i+1}/{self.N}] {time.time()-t0:.0f}s "
                      f"({n_eddy} w/ eddy)")
        pix = labels.sum() / self.N
        print(f"  Done. {n_eddy}/{self.N} have eddy "
              f"({n_eddy/self.N*100:.1f}%). "
              f"Avg eddy px: {pix:.1f}/{OCEAN_H*OCEAN_W}")
        return labels

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        vel = self.vel[idx]                              # (2, H, W)
        rows_i = self.obs_row_idx.long()
        cols_i = self.obs_col_idx.long()
        obs = vel[:, rows_i, cols_i].T                   # (N_obs, 2)
        obs = (obs - self.obs_mean) / self.obs_std
        return (obs, self.obs_row_idx, self.obs_col_idx,
                self.labels[idx], self.ocean_mask)


# ---------------------------------------------------------------------------
# Loss functions (identical to single-row version)
# ---------------------------------------------------------------------------
def focal_loss(logits, targets, alpha=0.75, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    pt = torch.where(targets > 0.5, p, 1 - p)
    at = torch.where(targets > 0.5, alpha, 1 - alpha)
    return at * ((1 - pt) ** gamma) * bce


def dice_loss(logits, targets, smooth=1.0):
    p = torch.sigmoid(logits)
    inter = (p * targets).sum()
    union = p.sum() + targets.sum()
    return 1 - (2 * inter + smooth) / (union + smooth)


def combined_loss(logits, targets, ocean_mask):
    mask = ocean_mask.bool()
    logits_flat = logits.squeeze(1)[mask]
    targets_flat = targets[mask]
    fl = focal_loss(logits_flat, targets_flat).mean()
    dl = dice_loss(logits_flat, targets_flat)
    return fl + dl


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_pickle_data(path="data.pickle"):
    print(f"Loading data from {path}...")
    with open(path, "rb") as f:
        train_np, val_np, _test_np = pickle.load(f)

    def to_tensor(arr):
        t = torch.from_numpy(np.ascontiguousarray(arr)).float()
        t = t.permute(3, 2, 1, 0)   # (N, 2, 44, 94)
        return torch.nan_to_num(t, nan=0.0)

    return to_tensor(train_np), to_tensor(val_np)


def parse_obs_rows(s: str) -> list[int]:
    """Parse --obs-rows argument."""
    if s.strip().lower() == "all":
        return list(range(OCEAN_H))
    return [int(x.strip()) for x in s.split(",")]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(args):
    obs_rows = parse_obs_rows(args.obs_rows)
    out_dir = Path(f"results/teddy_{args.tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Data — reuse cached labels from single-row run if available
    train_vel, val_vel = load_pickle_data()
    label_cache_train = "results/teddy_baseline/teddy_labels_train.pt"
    label_cache_val = "results/teddy_baseline/teddy_labels_val.pt"

    train_ds = TeddyMultiRowDataset(train_vel, obs_rows,
                                     label_cache_path=label_cache_train)
    val_ds = TeddyMultiRowDataset(val_vel, obs_rows,
                                   label_cache_path=label_cache_val)
    # Use train stats for val too
    val_ds.obs_mean = train_ds.obs_mean
    val_ds.obs_std = train_ds.obs_std

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                               shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=0)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    # Model
    model = TeddyNetMultiRow(
        obs_dim=2, d_model=64, n_heads=4, n_enc_layers=3,
        n_cnn_layers=8, ocean_h=OCEAN_H, ocean_w=OCEAN_W,
    ).to(device)
    print(f"Parameters: {model.count_parameters():,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val = float("inf")
    log_lines = []
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        eloss, nb = 0.0, 0
        for obs, rows, cols, label, ocean in train_loader:
            obs = obs.to(device)
            rows = rows.to(device)
            cols = cols.to(device)
            label = label.to(device)
            ocean = ocean.to(device)

            logits = model(obs, rows, cols)
            loss = combined_loss(logits, label, ocean)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            eloss += loss.item()
            nb += 1

        scheduler.step()
        tavg = eloss / max(nb, 1)

        # Validation
        model.eval()
        vloss, vn = 0.0, 0
        with torch.no_grad():
            for obs, rows, cols, label, ocean in val_loader:
                obs = obs.to(device)
                rows = rows.to(device)
                cols = cols.to(device)
                label = label.to(device)
                ocean = ocean.to(device)
                logits = model(obs, rows, cols)
                vloss += combined_loss(logits, label, ocean).item()
                vn += 1
        vavg = vloss / max(vn, 1)

        line = (f"Epoch {epoch:3d}/{args.epochs}  "
                f"train={tavg:.4f}  val={vavg:.4f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}")
        print(line)
        log_lines.append(line)

        if vavg < best_val:
            best_val = vavg
            patience_counter = 0
            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_loss": vavg,
                "obs_mean": train_ds.obs_mean,
                "obs_std": train_ds.obs_std,
                "obs_rows": obs_rows,
                "n_obs": train_ds.n_obs,
                "eddy_params": EDDY_PARAMS,
                "tag": args.tag,
            }
            torch.save(ckpt, out_dir / "teddy_best.pt")
            print(f"  → saved best (val={vavg:.4f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= args.patience:
            print(f"  Early stopping at epoch {epoch} "
                  f"(no improvement for {args.patience} epochs)")
            break

    # Save log
    with open(out_dir / "training_log.txt", "w") as f:
        f.write("\n".join(log_lines) + "\n")
    print(f"\nDone. Best val: {best_val:.4f}  →  {out_dir}/")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--obs-rows", type=str, required=True,
                   help="Comma-separated row indices or 'all'")
    p.add_argument("--tag", type=str, required=True,
                   help="Name tag for output dir (results/teddy_<tag>/)")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=30,
                   help="Early stopping patience")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.epochs = 3
        args.batch_size = 8
    train(args)


if __name__ == "__main__":
    main()
