"""GP-forward / GP-conditioned inpainting dataset.

Supports two modes controlled by the ``gp_conditioned`` flag:

**GP-forward** (``gp_conditioned=False``, legacy):
    Forward diffusion noises the GP posterior instead of GT.
    known_values = x0 * (1-mask)  (sparse GT observations at known pixels).

**GP-conditioned** (``gp_conditioned=True``, recommended):
    Forward diffusion noises GT normally (standard DDPM!).
    known_values = gp_std  (full GP field everywhere as conditioning).
    This aligns training and inference distributions perfectly:
      - Training:  model input = [noised_GT, mask, GP], predict GT
      - Inference: model input = [x_t_from_reverse, mask, GP], predict GT
    The GP conditioning is computed identically in both cases.

The dataset loads pre-computed GP fields (raw space) from a .pt file produced
by ``scripts/precompute_gp.py``, standardizes them once at startup, and
returns 6-tuples:

    (x0, gp_std, t, noise, mask, known_values)

where:
    x0          – standardized ground truth  (2, H, W)   [prediction target]
    gp_std      – standardized GP posterior  (2, H, W)   [noised in gp_forward;
                                                           passed through in gp_conditioned]
    t           – random diffusion timestep  (scalar)
    noise       – noise sample               (2, H, W)
    mask        – fixed observation mask     (1, H, W)   [1=missing, 0=known]
    known_values– conditioning channels      (2, H, W)   [GP field or sparse GT, by mode]

Augmentation (controlled by ``augment`` flag, GP-conditioned mode only):
    - Horizontal flip: flip cols and negate u (channel 0)
    - Vertical flip: flip rows and negate v (channel 1)
    These preserve divergence-free structure (∇·v = 0).
    Both x0 AND gp are flipped consistently so conditioning stays aligned.
    Noise is re-generated after flipping to match the augmented field.
"""

import random
import torch
from torch.utils.data import Dataset


class OceanGPForwardDataset(Dataset):
    """Wraps an OceanImageDataset + pre-computed GP fields.

    Supports two modes:
      - gp_conditioned=False (legacy "gp_forward"):
            gp_std is noised by the training loop; known = sparse GT obs
      - gp_conditioned=True:
            gp_std is returned but NOT noised (training loop noises GT);
            known = full GP field (used as conditioning input)
    """

    def __init__(self, base_dataset, gp_fields_raw, standardizer, mask_1ch,
                 gp_conditioned=False, augment=False):
        """
        Args:
            base_dataset: OceanImageDataset (returns (x0, t, noise))
            gp_fields_raw: (N, 2, H, W) tensor — GP posteriors in raw space.
                           Must be aligned with base_dataset by index.
            standardizer: callable, e.g. UnifiedZScoreStandardizer
            mask_1ch: (1, 1, H, W) or (1, H, W) observation mask (1=missing, 0=known)
            gp_conditioned: if True, use GP-conditioned mode (noise GT,
                           condition on full GP field)
            augment: if True, apply random velocity-field-aware flips
                     (only applies to gp_conditioned mode)
        """
        assert len(base_dataset) == len(gp_fields_raw), (
            f"Dataset has {len(base_dataset)} samples but GP cache has "
            f"{len(gp_fields_raw)}"
        )
        self.base = base_dataset
        self.standardizer = standardizer
        self.gp_conditioned = gp_conditioned
        self.augment = augment and gp_conditioned  # only for gp_conditioned

        # Pre-standardize all GP fields at init (one-time cost)
        self.gp_std = standardizer(gp_fields_raw)  # (N, 2, H, W)

        # Store mask as (1, H, W)
        if mask_1ch.dim() == 4:
            mask_1ch = mask_1ch.squeeze(0)  # (1, H, W)
        self.mask = mask_1ch  # (1, H, W)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x0, t, noise = self.base[idx]          # x0: standardized GT (2, H, W)
        gp = self.gp_std[idx].clone()           # standardized GP (2, H, W)
        mask = self.mask                         # (1, H, W)

        # ── Velocity-field-aware augmentation ─────────────────────
        # Flips that preserve ∇·v = 0:
        #   H-flip: x → -x, so u → -u (negate channel 0, flip dim=-1)
        #   V-flip: y → -y, so v → -v (negate channel 1, flip dim=-2)
        # Both x0 AND gp must be flipped consistently.
        # Mask is symmetric so no flip needed (same fixed observation pattern,
        # but we flip it too for generality).
        if self.augment:
            flip_h = random.random() < 0.5
            flip_v = random.random() < 0.5
            if flip_h or flip_v:
                x0 = x0.clone()
                if flip_h:
                    x0 = x0.flip(-1);  x0[0] = -x0[0]
                    gp = gp.flip(-1);  gp[0] = -gp[0]
                    noise = noise.flip(-1); noise[0] = -noise[0]
                    mask = mask.flip(-1)
                if flip_v:
                    x0 = x0.flip(-2);  x0[1] = -x0[1]
                    gp = gp.flip(-2);  gp[1] = -gp[1]
                    noise = noise.flip(-2); noise[1] = -noise[1]
                    mask = mask.flip(-2)

        if self.gp_conditioned:
            # GP-conditioned: condition on full GP field, noise GT in training loop
            known = gp                           # full GP field (2, H, W)
        else:
            # GP-forward (legacy): sparse GT observations, noise GP in training loop
            known = x0[:2] * (1.0 - mask)        # GT observations (2, H, W)

        return x0, gp, t, noise, mask, known
