"""Multi-mask GP-context dataset for training with diverse observation patterns.

Extends OceanGPContextDataset to support multiple pre-computed mask
configurations. Each __getitem__ call randomly selects one of the available
masks (and its corresponding GP posterior / distance map), so the model
sees varied observation densities (e.g. 0.1%, 1%, 5%, 10% known) during
training.

Returns the same 6-tuple interface as OceanGPContextDataset:
    (x0, gp_std, t, noise, mask, known_context)

where known_context = [gp_mean_u, gp_mean_v, gp_var_max, distance, ocean_mask]
"""

import random
import torch
from torch.utils.data import Dataset


class OceanGPMultiMaskDataset(Dataset):
    """GP-context dataset with multiple observation masks for diversity.

    During each sample access, a mask configuration is chosen uniformly
    at random.  The corresponding GP posterior mean, variance, distance
    map, and mask are used to build the conditioning channels.
    """

    def __init__(
        self,
        base_dataset,
        gp_fields_list,      # list of (N, 2, H, W) tensors — one per mask
        var_fields_list,      # list of (N, 2, H, W) tensors — one per mask
        standardizer,
        masks_1ch,            # list of (1, 1, H, W) or (1, H, W) masks
        dist_maps,            # list of (1, 1, H, W) or (1, H, W) distance maps
        ocean_mask=None,
        augment=False,
    ):
        """
        Args:
            base_dataset: OceanImageDataset (returns (x0, t, noise))
            gp_fields_list: list of (N, 2, H, W) — GP posterior means per mask (raw)
            var_fields_list: list of (N, 2, H, W) — GP posterior variances per mask (raw)
            standardizer: callable, e.g. UnifiedZScoreStandardizer
            masks_1ch: list of (1, 1, H, W) or (1, H, W) observation masks (1=missing)
            dist_maps: list of (1, 1, H, W) or (1, H, W) distance maps
            ocean_mask: optional (1, 1, H, W) shared ocean mask
            augment: if True, apply random velocity-field-aware flips
        """
        self.n_masks = len(gp_fields_list)
        assert self.n_masks == len(var_fields_list) == len(masks_1ch) == len(dist_maps)

        for i in range(self.n_masks):
            assert len(base_dataset) == len(gp_fields_list[i]), (
                f"Dataset has {len(base_dataset)} samples but GP cache {i} "
                f"has {len(gp_fields_list[i])}"
            )

        self.base = base_dataset
        self.standardizer = standardizer
        self.augment = augment

        # Pre-standardize GP means for each mask (one-time cost)
        self.gp_std_list = [standardizer(gp) for gp in gp_fields_list]

        # GP variance: max over u,v channels, normalize to [0,1]
        self.gp_var_list = []
        self.var_global_maxes = []
        for var_raw in var_fields_list:
            var_max = var_raw.max(dim=1, keepdim=True).values  # (N, 1, H, W)
            vgm = var_max.max()
            if vgm > 0:
                self.gp_var_list.append((var_max / vgm).float())
            else:
                self.gp_var_list.append(var_max.float())
            self.var_global_maxes.append(vgm)

        # Store masks as (1, H, W) each
        self.masks = []
        for m in masks_1ch:
            if m.dim() == 4:
                m = m.squeeze(0)
            self.masks.append(m)

        # Distance maps as (1, H, W)
        self.dist_maps = []
        for d in dist_maps:
            if d.dim() == 4:
                d = d.squeeze(0)
            self.dist_maps.append(d.float())

        # Ocean mask (shared across all masks)
        if ocean_mask is not None:
            if ocean_mask.dim() == 4:
                ocean_mask = ocean_mask.squeeze(0)
            self.ocean_mask = ocean_mask.float()
        else:
            self.ocean_mask = self._derive_ocean_mask()

    def _derive_ocean_mask(self):
        H, W = self.masks[0].shape[-2:]
        ocean = torch.zeros(1, H, W)
        ocean[0, :44, :94] = 1.0
        return ocean

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x0, t, noise = self.base[idx]

        # Randomly select a mask configuration
        m = random.randint(0, self.n_masks - 1)

        gp = self.gp_std_list[m][idx].clone()
        gp_var = self.gp_var_list[m][idx]
        mask = self.masks[m]
        dist = self.dist_maps[m]
        ocean = self.ocean_mask

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
        known_context = torch.cat([gp, gp_var, dist, ocean], dim=0)

        return x0, gp, t, noise, mask, known_context
