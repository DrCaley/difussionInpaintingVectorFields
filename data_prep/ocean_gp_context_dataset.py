"""GP-context conditioned inpainting dataset.

Extends the GP-conditioned approach with dense spatial context channels
inspired by V-CNN's 5-channel input design.  Instead of conditioning on
sparse observations, this dataset provides the denoiser with:

  - GP posterior mean (2ch) — dense initial estimate everywhere
  - GP posterior variance (1ch, max over u/v) — uncertainty map
  - Distance to nearest sensor (1ch) — spatial proximity
  - Ocean mask (1ch) — domain boundary

Training works as standard conditional DDPM:
  1. Forward-diffuse GT: x_t = √ᾱ x₀ + √(1-ᾱ)ε
  2. Feed [x_t(2), mask(1), gp_mean(2), gp_var(1), distance(1), ocean_mask(1)] → UNet
  3. UNet predicts x₀ (or ε)
  4. Loss on missing region only (with mask_xt)

The GP context channels are STATIC (same for all timesteps) — only x_t evolves.
This is the Palette (Saharia et al. 2022) approach but with richer conditioning.

Returns 7-tuples:
    (x0, gp_std, t, noise, mask, known_context)

where:
    x0            – standardized ground truth          (2, H, W)
    gp_std        – standardized GP posterior mean     (2, H, W)
    t             – random diffusion timestep          (scalar)
    noise         – noise sample                       (2, H, W)
    mask          – observation mask                   (1, H, W)  [1=missing, 0=known]
    known_context – dense conditioning channels        (5, H, W)
                    [gp_mean_u, gp_mean_v, gp_var_max, distance, ocean_mask]
"""

import random
import torch
from torch.utils.data import Dataset


class OceanGPContextDataset(Dataset):
    """GP-context conditioned dataset for training with dense spatial priors.

    Provides the denoiser with V-CNN-inspired dense conditioning channels
    alongside the noisy image, enabling it to leverage spatial context
    at every pixel during iterative denoising.
    """

    def __init__(self, base_dataset, gp_fields_raw, var_fields_raw,
                 standardizer, mask_1ch, dist_map, ocean_mask=None,
                 augment=False):
        """
        Args:
            base_dataset: OceanImageDataset (returns (x0, t, noise))
            gp_fields_raw: (N, 2, H, W) tensor — GP posterior means (raw space)
            var_fields_raw: (N, 2, H, W) tensor — GP posterior variances (raw space)
            standardizer: callable, e.g. UnifiedZScoreStandardizer
            mask_1ch: (1, 1, H, W) or (1, H, W) observation mask (1=missing, 0=known)
            dist_map: (1, 1, H, W) or (1, H, W) normalized distance to nearest sensor
            ocean_mask: (1, 1, H, W) or (1, H, W) optional; 1=ocean, 0=land/padding.
                       If None, derived from the data (non-zero = ocean).
            augment: if True, apply random velocity-field-aware flips
        """
        assert len(base_dataset) == len(gp_fields_raw), (
            f"Dataset has {len(base_dataset)} samples but GP cache has "
            f"{len(gp_fields_raw)}"
        )
        assert len(base_dataset) == len(var_fields_raw), (
            f"Dataset has {len(base_dataset)} samples but variance cache has "
            f"{len(var_fields_raw)}"
        )
        self.base = base_dataset
        self.standardizer = standardizer
        self.augment = augment

        # Pre-standardize GP means at init (one-time cost)
        self.gp_std = standardizer(gp_fields_raw)  # (N, 2, H, W)

        # GP variance: take max over u,v channels to get single uncertainty map
        # Normalize to [0, 1] range for stable training
        var_max = var_fields_raw.max(dim=1, keepdim=True).values  # (N, 1, H, W)
        var_global_max = var_max.max()
        if var_global_max > 0:
            self.gp_var = (var_max / var_global_max).float()  # (N, 1, H, W)
        else:
            self.gp_var = var_max.float()
        self.var_global_max = var_global_max  # store for potential denormalization

        # Store mask as (1, H, W)
        if mask_1ch.dim() == 4:
            mask_1ch = mask_1ch.squeeze(0)
        self.mask = mask_1ch  # (1, H, W)

        # Distance map: (1, H, W) already normalized to [0, 1]
        if dist_map.dim() == 4:
            dist_map = dist_map.squeeze(0)
        self.dist_map = dist_map.float()  # (1, H, W)

        # Ocean mask: (1, H, W)
        if ocean_mask is not None:
            if ocean_mask.dim() == 4:
                ocean_mask = ocean_mask.squeeze(0)
            self.ocean_mask = ocean_mask.float()  # (1, H, W)
        else:
            # Derive from data: ocean cells are within the 44×94 area
            # Everything outside is land/padding (zeros)
            self.ocean_mask = self._derive_ocean_mask()

    def _derive_ocean_mask(self):
        """Create ocean mask from the known domain geometry (44×94 area)."""
        H, W = self.mask.shape[-2:]
        ocean = torch.zeros(1, H, W)
        ocean[0, :44, :94] = 1.0  # OCEAN_H=44, OCEAN_W=94
        return ocean

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x0, t, noise = self.base[idx]       # x0: standardized GT (2, H, W)
        gp = self.gp_std[idx].clone()        # standardized GP mean (2, H, W)
        gp_var = self.gp_var[idx]           # GP variance (1, H, W)
        mask = self.mask                     # (1, H, W)
        dist = self.dist_map                 # (1, H, W)
        ocean = self.ocean_mask              # (1, H, W)

        # ── Velocity-field-aware augmentation ─────────────────────
        if self.augment:
            flip_h = random.random() < 0.5
            flip_v = random.random() < 0.5
            if flip_h or flip_v:
                x0 = x0.clone()
                gp_var = gp_var.clone()
                dist = dist.clone()
                ocean = ocean.clone()
                mask = mask.clone()
                if flip_h:
                    x0 = x0.flip(-1);  x0[0] = -x0[0]
                    gp = gp.flip(-1);  gp[0] = -gp[0]
                    noise = noise.flip(-1); noise[0] = -noise[0]
                    gp_var = gp_var.flip(-1)
                    dist = dist.flip(-1)
                    ocean = ocean.flip(-1)
                    mask = mask.flip(-1)
                if flip_v:
                    x0 = x0.flip(-2);  x0[1] = -x0[1]
                    gp = gp.flip(-2);  gp[1] = -gp[1]
                    noise = noise.flip(-2); noise[1] = -noise[1]
                    gp_var = gp_var.flip(-2)
                    dist = dist.flip(-2)
                    ocean = ocean.flip(-2)
                    mask = mask.flip(-2)

        # Build dense conditioning context: (5, H, W)
        # [gp_mean_u, gp_mean_v, gp_var_max, distance, ocean_mask]
        known_context = torch.cat([gp, gp_var, dist, ocean], dim=0)  # (5, H, W)

        # Return 6-tuple matching the GP-conditioned interface
        # (the training loop unpacks 6-tuples as: x0, gp_source, t, noise, mask, known)
        return x0, gp, t, noise, mask, known_context
