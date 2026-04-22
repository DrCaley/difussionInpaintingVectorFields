"""Palette-style inpainting dataset.

Wraps OceanImageDataset to produce (x0, t, noise, mask, known_values)
tuples for mask-aware training.

Each __getitem__ call:
  1. Gets (x0, t, noise) from the base dataset
  2. Optionally applies velocity-field-aware augmentation
  3. Generates a random training mask
  4. Computes known_values = x0 * (1 - mask)  (standardised space)
  5. Returns (x0, t, noise, mask_single, known_values)

The training loop concatenates [x_t, mask, known_values] → 5 channels
before passing to the inpainting UNet.

If ``use_distance_field=True``, a normalized distance-to-nearest-sensor
field (inspired by V-CNN) is appended to known_values, making it
(3, H, W) instead of (2, H, W).  The training loop then concatenates
[x_t, mask, known_values] → 6 channels.

Note: after resize_transform + standardize, x0 is (2, 64, 128).
The original ocean mask is dropped by resize (3→2 channels).
Land pixels are zero-padded regions (rows 44-63, cols 94-127)
which become the standardizer's negative mean after standardization.
We detect land by unstandardizing and checking magnitude.

Augmentation (controlled by ``augment`` flag):
  - Horizontal flip: flip spatial cols, negate u (x-velocity)
  - Vertical flip: flip spatial rows, negate v (y-velocity)
  Both preserve divergence-free property: ∇·v = ∂u/∂x + ∂v/∂y.
  Applied with 50% probability each (4 possible combinations).
  Noise is re-generated after augmentation to match the flipped field.
"""

import random
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from torch.utils.data import Dataset

from ddpm.helper_functions.masks.training_masks import generate_training_mask

# Ocean domain within padded (64, 128) grid
OCEAN_H, OCEAN_W = 44, 94


class OceanInpaintDataset(Dataset):
    """Wraps an OceanImageDataset to add mask conditioning for training."""

    def __init__(self, base_dataset, standardizer=None, augment=False,
                 use_distance_field=False, use_bathymetry=False,
                 use_voronoi_fill=False):
        """
        Args:
            base_dataset: OceanImageDataset instance (returns (x0, t, noise))
            standardizer: used to unstandardize for land-mask detection
            augment: if True, apply random velocity-field flips
            use_distance_field: if True, append normalized distance-to-nearest-
                sensor field to known_values (2ch → 3ch)
            use_bathymetry: if True, append normalized bathymetry field
            use_voronoi_fill: if True, replace sparse obs with Voronoi
                (nearest-neighbour) fill so conditioning channels are dense
        """
        self.base = base_dataset
        self.standardizer = standardizer
        self.augment = augment
        self.use_distance_field = use_distance_field
        self.use_bathymetry = use_bathymetry
        self.use_voronoi_fill = use_voronoi_fill
        # Pre-compute a static land mask from the first sample
        # (land pixels are the same across all time steps)
        self._land_mask = None

    def _get_land_mask(self, x0, idx=None):
        """Compute land mask (1=ocean, 0=land) from standardized x0.

        When the base dataset has per-sample ocean masks (subframe data),
        use those directly. Otherwise fall back to magnitude thresholding.
        """
        # Per-sample ocean mask from extended pickle
        if idx is not None and hasattr(self.base, 'ocean_masks') and self.base.ocean_masks is not None:
            # The pre-resize ocean mask is stored in tensor_arr as channel 2
            # but after resize_transform it's cropped. Use the raw mask instead.
            mask_slice = self.base.ocean_masks[..., idx]
            if isinstance(mask_slice, np.ndarray):
                raw_mask = torch.from_numpy(mask_slice.T.copy()).float()
            else:
                raw_mask = mask_slice.T.float()  # (44, 94)
            # Pad to match the resized (64, 128) shape
            _, h, w = x0.shape
            padded = torch.zeros(1, h, w)
            mh, mw = raw_mask.shape
            padded[0, :mh, :mw] = raw_mask
            return padded

        # Legacy path: static mask derived from magnitude
        if self._land_mask is not None:
            return self._land_mask

        if self.standardizer is not None:
            x0_raw = self.standardizer.unstandardize(x0)
        else:
            x0_raw = x0
        # Ocean pixels have non-zero magnitude; padded land is exactly zero
        mag = (x0_raw[:2] ** 2).sum(dim=0).sqrt()  # (H, W)
        land_mask = (mag > 1e-5).float().unsqueeze(0)  # (1, H, W)
        self._land_mask = land_mask
        return land_mask

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x0, t, noise, _data_sample_num = self.base[idx]
        # x0 is (2, 64, 128) — standardised [u, v]
        c, h, w = x0.shape

        # ── Velocity-field augmentation ──────────────────────────
        # Flips that preserve ∇·v = 0:
        #   H-flip: x → -x, so u → -u (negate channel 0, flip dim=-1)
        #   V-flip: y → -y, so v → -v (negate channel 1, flip dim=-2)
        # Noise must match the augmented field, so we re-generate it
        # after flipping (the noise strategy may depend on x0 shape).
        if self.augment:
            flip_h = random.random() < 0.5
            flip_v = random.random() < 0.5
            if flip_h or flip_v:
                if flip_h:
                    x0 = x0.flip(-1)         # flip cols
                    x0[0] = -x0[0]           # negate u
                    noise = noise.flip(-1)
                    noise[0] = -noise[0]
                if flip_v:
                    x0 = x0.flip(-2)         # flip rows
                    x0[1] = -x0[1]           # negate v
                    noise = noise.flip(-2)
                    noise[1] = -noise[1]
                # Invalidate cached land mask since spatial layout changed
                self._land_mask = None

        # Detect land mask from the data
        land_mask = self._get_land_mask(x0, idx=idx)  # (1, H, W)

        # Generate random training mask: 1=missing, 0=known
        mask_single = generate_training_mask(h, w, land_mask)  # (1, H, W)

        # Known values: zero out the missing region (in standardised space)
        known_mask = 1.0 - mask_single  # 1=known, 0=missing

        if self.use_voronoi_fill:
            # Dense Voronoi (nearest-neighbour) fill in standardised space.
            # Every pixel gets the value of its nearest observed pixel,
            # giving the conditioning encoder dense spatial input instead
            # of 95%+ zeros.
            obs_yx = np.argwhere(known_mask[0, :OCEAN_H, :OCEAN_W].numpy() > 0.5)
            if len(obs_yx) >= 1:
                tree = cKDTree(obs_yx.astype(np.float64))
                gy, gx = np.mgrid[0:OCEAN_H, 0:OCEAN_W]
                grid_coords = np.stack([gy.ravel(), gx.ravel()], axis=1).astype(np.float64)
                _, nn_idx = tree.query(grid_coords, k=1)
                nn_idx = nn_idx.reshape(OCEAN_H, OCEAN_W)
                # Observed values in standardised space
                obs_u = x0[0, :OCEAN_H, :OCEAN_W].numpy()
                obs_v = x0[1, :OCEAN_H, :OCEAN_W].numpy()
                obs_u_vals = obs_u[obs_yx[:, 0], obs_yx[:, 1]]
                obs_v_vals = obs_v[obs_yx[:, 0], obs_yx[:, 1]]
                ocean_np = land_mask[0, :OCEAN_H, :OCEAN_W].numpy()
                vor_u = obs_u_vals[nn_idx] * ocean_np
                vor_v = obs_v_vals[nn_idx] * ocean_np
                known_values = torch.zeros(2, h, w)
                known_values[0, :OCEAN_H, :OCEAN_W] = torch.from_numpy(vor_u.astype(np.float32))
                known_values[1, :OCEAN_H, :OCEAN_W] = torch.from_numpy(vor_v.astype(np.float32))
            else:
                known_values = torch.zeros(2, h, w)
        else:
            known_values = x0[:2] * known_mask  # (2, H, W), zeros where missing

        if self.use_distance_field:
            # Observation mask: 1 where we have a real sensor, 0 elsewhere.
            # Exclude land pixels (which also have mask=0 but are not sensors).
            obs_mask = known_mask * land_mask  # (1, H, W)
            # EDT from nearest observation; zero at sensor locations.
            dist = distance_transform_edt(
                (1.0 - obs_mask[0]).numpy()
            )  # (H, W)
            # Normalize to [0, 1]
            dmax = dist.max()
            if dmax > 0:
                dist = dist / dmax
            dist_tensor = torch.from_numpy(dist).float().unsqueeze(0)  # (1, H, W)
            known_values = torch.cat([known_values, dist_tensor], dim=0)  # (3, H, W)

        if self.use_bathymetry:
            bathy = self.base.load_bathymetry(idx)  # (H_ocean, W_ocean) or None
            if bathy is not None:
                _, bh, bw = known_values.shape
                bathy_padded = torch.zeros(1, bh, bw)
                bh_src, bw_src = bathy.shape
                bathy_padded[0, :bh_src, :bw_src] = bathy
                known_values = torch.cat([known_values, bathy_padded], dim=0)
            else:
                # Fallback: zero bathymetry channel
                known_values = torch.cat([known_values, torch.zeros_like(known_values[:1])], dim=0)

        return x0, t, noise, mask_single, known_values
