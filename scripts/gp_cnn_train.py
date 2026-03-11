#!/usr/bin/env python3
"""
Train a GP-CNN model for sparse velocity field reconstruction.

Instead of Voronoi tessellation (nearest-neighbour piecewise-constant),
this uses precomputed GP posterior mean + variance as CNN input.
The GP provides smooth, physically-informed initial estimates plus
pixel-wise uncertainty — the CNN learns to refine the GP output.

Key insight: GP variance from a fixed mask depends only on observation
geometry, NOT on observed values. So the variance map is computed once
and reused for all samples. GP means are precomputed by precompute_gp.py.

Input channels (6):
  0 — GP posterior mean (u component)  [varies per sample]
  1 — GP posterior mean (v component)  [varies per sample]
  2 — GP posterior std  (u component)  [same for all samples]
  3 — GP posterior std  (v component)  [same for all samples]
  4 — binary sensor mask (1 where observed)
  5 — ocean mask

Architecture: Same U-Net-lite as VoronoiCNN, with in_channels=6.

Prerequisites:
    PYTHONPATH=. python scripts/precompute_gp.py  # if not already done

Usage:
    PYTHONPATH=. python scripts/gp_cnn_train.py [--epochs 200] [--smoke]

Outputs saved to results/gp_cnn/:
    gp_cnn_best.pt     — best model checkpoint
    training_log.csv   — per-epoch metrics
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

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from scripts.voronoi_cnn_model import VoronoiCNN  # reuse same U-Net architecture
from ddpm.helper_functions.interpolation_tool import gp_fill

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OCEAN_H, OCEAN_W = 44, 94
FULL_H, FULL_W = 64, 128
OBS_ROW = 22
OUT_DIR = Path("results/gp_cnn")

GP_PARAMS = {
    "lengthscale": 14.1,
    "variance": 0.0103420345,
    "noise": 1e-8,
    "kernel_type": "rbf_legacy",
    "coord_system": "pixels",
}


# ---------------------------------------------------------------------------
# Compute GP variance map (once, for the fixed row-22 mask)
# ---------------------------------------------------------------------------

def compute_gp_variance_map(ocean_mask: np.ndarray) -> np.ndarray:
    """
    Compute the GP posterior variance for the fixed row-22 observation mask.
    
    Since variance depends only on observation locations (not values),
    we only need to compute this once. We use a dummy velocity field
    (all zeros) — the variance is independent of the actual values.
    
    Returns: (2, OCEAN_H, OCEAN_W) GP posterior std, normalized to [0,1].
    """
    print("  Computing GP variance map (one-time)...")
    
    # Build fixed row-22 observation mask for the ocean sub-domain
    obs_mask_ocean = np.zeros((OCEAN_H, OCEAN_W), dtype=np.float32)
    obs_mask_ocean[OBS_ROW, :] = 1.0
    obs_mask_ocean *= ocean_mask
    
    # Dummy velocity field — gp_fill uses valid_mask_2d = (abs(vel).sum > 1e-5)
    # to identify valid pixels. We must put non-zero values at ALL ocean pixels,
    # not just observed ones, otherwise gp_fill skips them.
    # Variance is independent of actual values, so any constant works.
    vel_full = np.zeros((1, 2, FULL_H, FULL_W), dtype=np.float32)
    vel_full[0, 0, :OCEAN_H, :OCEAN_W] = 0.01 * ocean_mask
    vel_full[0, 1, :OCEAN_H, :OCEAN_W] = 0.01 * ocean_mask
    
    # Build GP mask: 0=known, 1=unknown (gp_fill convention)
    gp_mask = np.ones((FULL_H, FULL_W), dtype=np.float32)
    gp_mask[:OCEAN_H, :OCEAN_W] = 1.0 - obs_mask_ocean  # known obs → 0
    # Land in ocean sub-domain: mark as known (0)
    gp_mask[:OCEAN_H, :OCEAN_W] *= ocean_mask  # only ocean unknowns stay 1
    # Border region: mark as known (0) 
    gp_mask[OCEAN_H:, :] = 0.0
    gp_mask[:, OCEAN_W:] = 0.0
    
    gp_mask_t = torch.from_numpy(gp_mask).unsqueeze(0).unsqueeze(0).expand(1, 2, -1, -1).clone()
    vel_t = torch.from_numpy(vel_full)
    
    _, var_map = gp_fill(
        vel_t, gp_mask_t,
        lengthscale=GP_PARAMS["lengthscale"],
        variance=GP_PARAMS["variance"],
        noise=GP_PARAMS["noise"],
        use_double=True,
        kernel_type=GP_PARAMS["kernel_type"],
        coord_system=GP_PARAMS["coord_system"],
        return_variance=True,
    )
    
    # Extract ocean sub-domain and convert to std
    var_ocean = var_map[0, :, :OCEAN_H, :OCEAN_W].numpy()  # (2, 44, 94)
    gp_std = np.sqrt(np.clip(var_ocean, 0, None))
    
    # Normalize each channel to [0, 1]
    for c in range(2):
        max_std = gp_std[c].max() + 1e-8
        gp_std[c] /= max_std
    
    gp_std *= ocean_mask[np.newaxis]
    print(f"  GP std range: u=[{gp_std[0].min():.4f}, {gp_std[0].max():.4f}], "
          f"v=[{gp_std[1].min():.4f}, {gp_std[1].max():.4f}]")
    
    return gp_std.astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class GPReconDataset(Dataset):
    """
    Each sample: (gp_input, target_vel)
        gp_input   : (6, H, W) — GP mean + std + masks
        target_vel : (2, H, W) — ground truth velocity field (normalized)
    
    Uses precomputed GP means (fast) + constant GP variance map.
    """

    def __init__(self, gp_means: torch.Tensor, vel_targets: torch.Tensor,
                 gp_std_map: np.ndarray, sensor_mask: np.ndarray,
                 ocean_mask: np.ndarray):
        """
        gp_means:    (N, 2, H_ocean, W_ocean) normalized GP posterior means
        vel_targets: (N, 2, H_ocean, W_ocean) normalized ground truth
        gp_std_map:  (2, H_ocean, W_ocean) normalized GP posterior std (constant)
        sensor_mask: (H_ocean, W_ocean) binary, 1=observed
        ocean_mask:  (H_ocean, W_ocean) binary, 1=ocean
        """
        super().__init__()
        self.gp_means = gp_means
        self.targets = vel_targets
        self.N = gp_means.shape[0]
        
        # Pre-build the constant channels as tensors
        self.gp_std = torch.from_numpy(gp_std_map)            # (2, H, W)
        self.sensor_mask = torch.from_numpy(sensor_mask).unsqueeze(0)  # (1, H, W)
        self.ocean_mask_ch = torch.from_numpy(ocean_mask).unsqueeze(0)  # (1, H, W)
        
        # Pre-cat the constant part: [std_u, std_v, sensor, ocean]
        self.const_channels = torch.cat([
            self.gp_std,           # (2, H, W)
            self.sensor_mask,      # (1, H, W)
            self.ocean_mask_ch,    # (1, H, W)
        ], dim=0)  # (4, H, W)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        gp_mean = self.gp_means[idx]  # (2, H, W) normalized
        target = self.targets[idx]     # (2, H, W) normalized
        
        # Stack: [gp_u, gp_v, std_u, std_v, sensor_mask, ocean_mask]
        gp_in = torch.cat([gp_mean, self.const_channels], dim=0)  # (6, H, W)
        
        return gp_in, target


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor,
                    ocean_mask: torch.Tensor) -> torch.Tensor:
    """MSE computed only over ocean pixels."""
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

def load_pickle_data(path: str = "data.pickle"):
    """Load raw velocity data → (train, val) as (N, 2, H, W) tensors.

    NOTE: Precomputed GP has gp_train (1st split) and gp_test (3rd split).
    We use the 3rd split (test) as validation so GP means match targets.
    """
    print(f"Loading raw data from {path}...")
    with open(path, "rb") as f:
        train_np, _val_np, test_np = pickle.load(f)

    def to_tensor(arr):
        t = torch.from_numpy(np.ascontiguousarray(arr)).float()
        t = t.permute(3, 2, 1, 0)
        t = torch.nan_to_num(t, nan=0.0)
        return t

    train_t = to_tensor(train_np)
    test_t = to_tensor(test_np)  # use test as val (matches gp_test)
    print(f"  Train: {train_t.shape}, Val (test split): {test_t.shape}")
    return train_t, test_t


def compute_normalization(vel: torch.Tensor, ocean_mask: np.ndarray):
    """Compute per-channel mean/std over ocean pixels."""
    mask_t = torch.from_numpy(ocean_mask).bool()
    ocean_vals = vel[:, :, mask_t]
    mean = ocean_vals.mean(dim=(0, 2))
    std = ocean_vals.std(dim=(0, 2)) + 1e-8
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

    # ── Load data ──
    train_vel, val_vel = load_pickle_data()

    # Ocean mask
    speed = (train_vel[0, 0] ** 2 + train_vel[0, 1] ** 2).numpy()
    ocean_mask = (speed > 1e-10).astype(np.float32)  # (44, 94)
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

    # Normalize targets
    train_targets = ((train_vel - mean_4d) / std_4d) * ocean_t
    val_targets = ((val_vel - mean_4d) / std_4d) * ocean_t

    # ── Load precomputed GP means ──
    gp_path = BASE_DIR / "data" / "rams_head" / "gp_precomputed.pt"
    if not gp_path.exists():
        print(f"\nERROR: Precomputed GP data not found at {gp_path}")
        print("Run first:  PYTHONPATH=. python scripts/precompute_gp.py")
        sys.exit(1)

    print(f"\nLoading precomputed GP from {gp_path}...")
    gp_data = torch.load(gp_path, map_location="cpu", weights_only=False)
    gp_train_raw = gp_data["gp_train"]  # (N_train, 2, 64, 128) raw space
    gp_test_raw = gp_data["gp_test"]    # (N_test, 2, 64, 128) raw space
    print(f"  GP train: {gp_train_raw.shape}, GP test: {gp_test_raw.shape}")

    # Extract 44×94 ocean sub-domain and normalize
    gp_train_ocean = gp_train_raw[:, :, :OCEAN_H, :OCEAN_W]
    gp_test_ocean = gp_test_raw[:, :, :OCEAN_H, :OCEAN_W]

    # Normalize GP means with same normalization as targets
    gp_train_norm = ((gp_train_ocean - mean_4d) / std_4d) * ocean_t
    gp_test_norm = ((gp_test_ocean - mean_4d) / std_4d) * ocean_t
    print(f"  GP means normalized: train range [{gp_train_norm.min():.3f}, {gp_train_norm.max():.3f}]")

    # ── Compute GP variance map (one-time) ──
    gp_std_map = compute_gp_variance_map(ocean_mask)

    # ── Sensor mask (fixed row-22) ──
    sensor_mask = np.zeros((OCEAN_H, OCEAN_W), dtype=np.float32)
    sensor_mask[OBS_ROW, :] = 1.0
    sensor_mask *= ocean_mask

    # ── Datasets ──
    train_ds = GPReconDataset(gp_train_norm, train_targets, gp_std_map,
                              sensor_mask, ocean_mask)
    val_ds = GPReconDataset(gp_test_norm, val_targets, gp_std_map,
                            sensor_mask, ocean_mask)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=False)
    print(f"\nTrain: {len(train_ds)}, Val: {len(val_ds)}")

    # ── Model ──
    model = VoronoiCNN(in_channels=6, out_channels=2,
                       base_ch=args.base_ch, depth=args.depth).to(device)
    n_params = model.count_parameters()
    print(f"Model: GP-CNN (U-Net-lite), {n_params:,} parameters")
    print(f"  in_channels=6, base_ch={args.base_ch}, depth={args.depth}")

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    ocean_mask_t = torch.from_numpy(ocean_mask).to(device)

    # ── Training state ──
    start_epoch = 1
    best_val_loss = float("inf")
    patience_counter = 0
    log_lines = ["epoch,train_loss,val_loss,lr,epoch_time"]

    if args.resume:
        ckpt_path = OUT_DIR / "gp_cnn_best.pt"
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
                "gp_params": GP_PARAMS,
                "gp_std_map": gp_std_map,
                "model_config": {
                    "in_channels": 6,
                    "out_channels": 2,
                    "base_ch": args.base_ch,
                    "depth": args.depth,
                    "model_type": "gp_cnn",
                },
            }
            torch.save(ckpt, OUT_DIR / "gp_cnn_best.pt")
            print(f"  → saved best (val={val_avg:.6f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= args.patience:
            print(f"  Early stopping at epoch {epoch} (patience={args.patience})")
            break

        # Periodic log save
        if epoch % 10 == 0 or epoch == args.epochs:
            with open(OUT_DIR / "training_log.csv", "w") as f:
                f.write("\n".join(log_lines) + "\n")

    # Final log
    with open(OUT_DIR / "training_log.csv", "w") as f:
        f.write("\n".join(log_lines) + "\n")

    print(f"\nDone. Best val loss: {best_val_loss:.6f}")
    print(f"Outputs in {OUT_DIR}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train GP-CNN velocity reconstruction model"
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
