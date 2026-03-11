#!/usr/bin/env python3
"""
Voronoi-CNN baseline for sparse velocity field reconstruction.

Based on: Fukami, Maulik, Ramachandra, Fukagata & Taira (2021),
"Global field reconstruction from sparse sensors with Voronoi
tessellation-assisted deep learning", Nature Machine Intelligence.

The idea:
  1. Given sparse velocity observations (e.g. row 22 of a 44×94 ocean domain),
     build a Voronoi tessellation that assigns each grid cell the value of
     the nearest observed pixel.  This creates a piecewise-constant "image"
     that covers the whole domain.
  2. Feed [voronoi_u, voronoi_v, distance_field] through a CNN encoder–decoder
     to reconstruct the full velocity field.

The CNN uses a U-Net-lite architecture with skip connections, suitable for
the small 44×94 domain.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# Voronoi pre-processing (runs on CPU/NumPy, fast for our small grids)
# ---------------------------------------------------------------------------

def build_voronoi_input(
    vel: np.ndarray,           # (2, H, W) velocity field — only known pixels are used
    mask: np.ndarray,          # (H, W)  1=known, 0=missing  (opposite of inpaint convention!)
    ocean_mask: np.ndarray,    # (H, W)  1=ocean, 0=land
) -> np.ndarray:
    """
    Build a Voronoi input tensor from sparse observations.

    Returns
    -------
    voronoi_input : np.ndarray, shape (5, H, W)
        Channels:
          0 — Voronoi-filled u  (nearest-neighbour value everywhere)
          1 — Voronoi-filled v
          2 — normalised distance to nearest sensor (0 at sensors, 1 at max)
          3 — binary sensor mask (1 where observed)
          4 — ocean mask
    """
    H, W = mask.shape
    # Observed pixel coordinates
    ky, kx = np.where(mask > 0.5)
    if len(ky) == 0:
        # Degenerate: no observations → return zeros
        return np.zeros((5, H, W), dtype=np.float32)

    # Build KD-tree of observed locations
    obs_coords = np.stack([ky, kx], axis=1).astype(np.float64)
    tree = cKDTree(obs_coords)

    # Query all grid points
    gy, gx = np.mgrid[0:H, 0:W]
    grid_coords = np.stack([gy.ravel(), gx.ravel()], axis=1).astype(np.float64)
    dist, idx = tree.query(grid_coords, k=1)

    dist = dist.reshape(H, W)
    idx = idx.reshape(H, W)

    # Voronoi fill: nearest sensor value at every pixel
    obs_u = vel[0, ky, kx]  # values at sensor locations
    obs_v = vel[1, ky, kx]
    voronoi_u = obs_u[idx]
    voronoi_v = obs_v[idx]

    # Normalise distance to [0, 1]
    max_dist = dist.max() + 1e-8
    dist_norm = dist / max_dist

    # Apply ocean mask: zero out land
    voronoi_u *= ocean_mask
    voronoi_v *= ocean_mask
    dist_norm *= ocean_mask

    sensor_mask = mask.astype(np.float32)

    out = np.stack([voronoi_u, voronoi_v, dist_norm, sensor_mask, ocean_mask],
                   axis=0).astype(np.float32)
    return out


def build_voronoi_input_batch(
    vel_batch: torch.Tensor,    # (B, 2, H, W)
    mask: np.ndarray,           # (H, W) 1=known, 0=missing — same for all samples
    ocean_mask: np.ndarray,     # (H, W)
) -> torch.Tensor:
    """Batch version — returns (B, 5, H, W) tensor."""
    B = vel_batch.shape[0]
    out = []
    for i in range(B):
        v = vel_batch[i].numpy()
        out.append(build_voronoi_input(v, mask, ocean_mask))
    return torch.from_numpy(np.stack(out, axis=0))


# ---------------------------------------------------------------------------
# CNN architecture — lightweight U-Net
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Conv → BN → ReLU → Conv → BN → ReLU"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class VoronoiCNN(nn.Module):
    """
    U-Net-lite encoder–decoder for Voronoi → velocity field reconstruction.

    Input:  (B, 5, H, W)  — voronoi_u, voronoi_v, dist, sensor_mask, ocean_mask
    Output: (B, 2, H, W)  — reconstructed u, v
    """

    def __init__(self, in_channels: int = 5, out_channels: int = 2,
                 base_ch: int = 64, depth: int = 4):
        super().__init__()
        self.depth = depth

        # Encoder
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch = in_channels
        for i in range(depth):
            out_ch = base_ch * (2 ** i)
            self.encoders.append(ConvBlock(ch, out_ch))
            self.pools.append(nn.MaxPool2d(2))
            ch = out_ch

        # Bottleneck
        self.bottleneck = ConvBlock(ch, ch * 2)

        # Decoder
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        ch = ch * 2
        for i in range(depth - 1, -1, -1):
            out_ch = base_ch * (2 ** i)
            self.upconvs.append(
                nn.ConvTranspose2d(ch, out_ch, kernel_size=2, stride=2)
            )
            self.decoders.append(ConvBlock(out_ch * 2, out_ch))  # *2 for skip
            ch = out_ch

        # Head
        self.head = nn.Conv2d(base_ch, out_channels, 1)

    def forward(self, x):
        # Pad to make dimensions divisible by 2^depth
        orig_h, orig_w = x.shape[2], x.shape[3]
        factor = 2 ** self.depth
        pad_h = (factor - orig_h % factor) % factor
        pad_w = (factor - orig_w % factor) % factor
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        # Encoder
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for upconv, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = upconv(x)
            # Handle size mismatch from non-power-of-2 dims
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear",
                                  align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        x = self.head(x)

        # Crop back to original size
        x = x[:, :, :orig_h, :orig_w]
        return x

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
