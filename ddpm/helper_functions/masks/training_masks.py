"""Gaussian mask generation for inpainting training.

Generates random sparse-pixel masks at fixed known-coverage fractions.
Mask convention: 1 = missing, 0 = known.

Each call uniformly picks one of the target known fractions
(0.1%, 0.2%, 0.5%, 1%, 2%, 5%, 10%) and samples that fraction
of pixels as known observations.
"""

import random
import torch
import numpy as np


# Known fractions to sample from (uniform random each call)
_KNOWN_FRACS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10]


def generate_training_mask(h: int, w: int, land_mask: torch.Tensor = None) -> torch.Tensor:
    """Generate a random gaussian (sparse pixel) mask for training.

    Uniformly picks a known fraction from _KNOWN_FRACS and marks
    that fraction of pixels as known (0), rest as missing (1).

    Args:
        h, w: spatial dimensions (64, 128 after resize)
        land_mask: (1, H, W) or (H, W) binary tensor, 1 = valid ocean pixel.
                   If provided, mask is intersected with land_mask so we never
                   ask the model to inpaint land pixels.

    Returns:
        mask: (1, H, W) float tensor.  1 = missing, 0 = known.
    """
    frac = random.choice(_KNOWN_FRACS)
    known = (torch.rand(1, h, w) < frac).float()
    mask = 1.0 - known

    # Intersect with land mask if provided
    if land_mask is not None:
        lm = land_mask.view(1, h, w) if land_mask.dim() == 2 else land_mask[:1]
        mask = mask * lm  # only mark ocean pixels as missing

    return mask
