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
import torch
from torch.utils.data import Dataset

from ddpm.helper_functions.masks.training_masks import generate_training_mask


class OceanInpaintDataset(Dataset):
    """Wraps an OceanImageDataset to add mask conditioning for training."""

    def __init__(self, base_dataset, standardizer=None, augment=False):
        """
        Args:
            base_dataset: OceanImageDataset instance (returns (x0, t, noise))
            standardizer: used to unstandardize for land-mask detection
            augment: if True, apply random velocity-field flips
        """
        self.base = base_dataset
        self.standardizer = standardizer
        self.augment = augment
        # Pre-compute a static land mask from the first sample
        # (land pixels are the same across all time steps)
        self._land_mask = None

    def _get_land_mask(self, x0):
        """Compute land mask (1=ocean, 0=land) from standardized x0."""
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
        x0, t, noise = self.base[idx]
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
        land_mask = self._get_land_mask(x0)  # (1, H, W)

        # Generate random training mask: 1=missing, 0=known
        mask_single = generate_training_mask(h, w, land_mask)  # (1, H, W)

        # Known values: zero out the missing region (in standardised space)
        known_mask = 1.0 - mask_single  # 1=known, 0=missing
        known_values = x0[:2] * known_mask  # (2, H, W), zeros where missing

        return x0, t, noise, mask_single, known_values
