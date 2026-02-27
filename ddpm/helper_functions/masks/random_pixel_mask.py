"""
RandomPixelMaskGenerator – reveals a given *percentage* of ocean pixels
and marks the rest as missing (to be inpainted).

Mask convention (same as every other mask in the project):
    1.0 = missing / to inpaint
    0.0 = known  / revealed

Usage:
    mask_gen = RandomPixelMaskGenerator(reveal_percent=1.0)  # reveal 1%
    mask = mask_gen.generate_mask(image_shape)                # (1,1,H,W)
"""

import torch

from data_prep.data_initializer import DDInitializer
from ddpm.helper_functions.masks.abstract_mask import MaskGenerator

dd = DDInitializer()


class RandomPixelMaskGenerator(MaskGenerator):
    """Randomly reveal *reveal_percent* % of pixels; inpaint the rest."""

    def __init__(self, reveal_percent: float = 1.0):
        """
        Parameters
        ----------
        reveal_percent : float
            Percentage of total pixels to reveal (mark as known).
            E.g. 1.0 → 1 % known, 99 % missing.
        """
        if not 0.0 < reveal_percent < 100.0:
            raise ValueError("reveal_percent must be in (0, 100)")
        self.reveal_percent = reveal_percent

    def generate_mask(self, image_shape=None):
        if image_shape is None:
            raise ValueError("image_shape is required")

        _, _, h, w = image_shape
        total_pixels = h * w
        n_reveal = max(1, round(total_pixels * self.reveal_percent / 100.0))

        # Start with everything missing
        mask = torch.ones(1, 1, h, w, dtype=torch.float32)

        # Pick n_reveal random pixel indices to reveal (set to 0)
        perm = torch.randperm(total_pixels)[:n_reveal]
        rows = perm // w
        cols = perm % w
        mask[0, 0, rows, cols] = 0.0

        return mask.to(dd.get_device())

    def __str__(self):
        return f"RandomPixelMask({self.reveal_percent}%)"

    def get_num_lines(self):
        return 0
