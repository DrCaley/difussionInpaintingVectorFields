"""On-the-fly GP-context dataset with truly random masks.

Every __getitem__ call generates a FRESH random observation mask and computes
the corresponding GP posterior on the fly.  This gives the model infinite
mask diversity — a different random mask for every sample on every epoch.

The mask density is sampled uniformly from a configurable list of known
fractions (e.g. 0.1%, 1%, 5%, 10%), and a completely new set of random
pixel positions is drawn each time.

GP computation cost per sample:
  - 0.1% known ≈ 4 pixels  → ~0.005s (4×4 Cholesky)
  - 0.2% known ≈ 8 pixels  → ~0.005s (8×8 Cholesky)
  - 0.5% known ≈ 21 pixels → ~0.01s  (21×21 Cholesky)
  - 1%   known ≈ 41 pixels → ~0.01s  (41×41 Cholesky)
With num_workers=16 in DataLoader, this parallelizes well.

Returns the same 6-tuple interface as OceanGPContextDataset:
    (x0, gp_std, t, noise, mask, known_context)

where known_context = [gp_mean_u, gp_mean_v, gp_var_max, distance, ocean_mask]
"""

import random
import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.ndimage import distance_transform_edt

# Import gp_fill — each DataLoader worker gets its own process so no GIL issue
from ddpm.helper_functions.interpolation_tool import gp_fill

# Domain constants
OCEAN_H, OCEAN_W = 44, 94
GRID_H, GRID_W = 64, 128
N_OCEAN = OCEAN_H * OCEAN_W  # 4136


class OceanGPOnTheFlyDataset(Dataset):
    """GP-context dataset with truly random masks computed on the fly.

    Each sample access:
      1. Picks a random known fraction from the configured list
      2. Generates a completely new random sparse mask
      3. Computes GP posterior mean & variance for that specific mask
      4. Computes distance-to-nearest-sensor map
      5. Returns the standard 6-tuple conditioning interface

    This eliminates the precompute bottleneck and gives infinite mask diversity.
    """

    def __init__(
        self,
        base_dataset,
        raw_velocity_data,
        standardizer,
        gp_params,
        known_fracs=(0.001, 0.01, 0.05, 0.10),
        ocean_mask=None,
        augment=False,
        border_mask=None,
    ):
        """
        Args:
            base_dataset: OceanImageDataset (returns (x0, t, noise))
            raw_velocity_data: (N, 2, H, W) tensor — UNstandardized velocity fields
            standardizer: callable, e.g. standardizer(tensor) → standardized tensor
            gp_params: dict with keys: lengthscale, variance, noise, kernel_type,
                       coord_system
            known_fracs: tuple/list of known fractions to sample from uniformly
            ocean_mask: (1, H, W) or (1, 1, H, W) — 1=ocean, 0=land. If None, derived.
            augment: if True, apply random velocity-field-aware flips
            border_mask: (1, 1, H, W) — border mask for padding. If None, computed.
        """
        assert len(base_dataset) == len(raw_velocity_data), (
            f"Dataset has {len(base_dataset)} but raw data has {len(raw_velocity_data)}"
        )
        self.base = base_dataset
        self.raw_data = raw_velocity_data  # (N, 2, H, W) — raw (unstandardized)
        self.standardizer = standardizer
        self.gp_params = gp_params
        self.known_fracs = list(known_fracs)
        self.augment = augment

        # Pre-compute ocean pixel coordinates (row, col) in the 44×94 ocean area
        # These are used for random mask generation
        ocean_rows, ocean_cols = [], []
        for r in range(OCEAN_H):
            for c in range(OCEAN_W):
                ocean_rows.append(r)
                ocean_cols.append(c)
        self.ocean_rows = np.array(ocean_rows)  # (4136,)
        self.ocean_cols = np.array(ocean_cols)  # (4136,)

        # Border mask for ensuring padding stays masked
        if border_mask is not None:
            if border_mask.dim() == 4:
                border_mask = border_mask.squeeze(0)
            self.border_mask_np = border_mask.squeeze().numpy()
        else:
            self.border_mask_np = self._default_border_mask()

        # Ocean mask: (1, H, W)
        if ocean_mask is not None:
            if ocean_mask.dim() == 4:
                ocean_mask = ocean_mask.squeeze(0)
            self.ocean_mask = ocean_mask.float()
        else:
            self.ocean_mask = self._derive_ocean_mask()

        # Pre-compute valid mask for GP (pixels with non-zero velocity)
        # This avoids GP trying to interpolate the land/padding zeros
        self.valid_mask_2d = (raw_velocity_data.abs().sum(dim=1) > 1e-5)  # (N, H, W)

    def _default_border_mask(self):
        """Default border: ocean area (44×94) is interior, rest is border."""
        mask = np.ones((GRID_H, GRID_W), dtype=np.float32)
        mask[:OCEAN_H, :OCEAN_W] = 1.0  # All ones — will be multiplied with generated mask
        return mask

    def _derive_ocean_mask(self):
        H, W = GRID_H, GRID_W
        ocean = torch.zeros(1, H, W)
        ocean[0, :OCEAN_H, :OCEAN_W] = 1.0
        return ocean

    def _generate_random_mask(self, known_frac):
        """Generate a random sparse observation mask.

        Args:
            known_frac: fraction of ocean pixels to mark as known

        Returns:
            mask_1ch: (1, H, W) float tensor. 1=missing, 0=known.
        """
        n_known = max(1, int(round(known_frac * N_OCEAN)))

        # Random subset of ocean pixels
        chosen = np.random.choice(N_OCEAN, size=n_known, replace=False)
        rows = self.ocean_rows[chosen]
        cols = self.ocean_cols[chosen]

        # Start all-missing, set chosen to known
        mask = np.ones((GRID_H, GRID_W), dtype=np.float32)
        mask[rows, cols] = 0.0

        # Apply border mask (land/padding stays missing=1)
        mask = mask * self.border_mask_np

        return torch.from_numpy(mask).unsqueeze(0)  # (1, H, W)

    def _compute_distance_map(self, mask_1ch):
        """Compute normalized distance to nearest known pixel."""
        known_binary = (1.0 - mask_1ch.squeeze().numpy())  # 1=known, 0=missing
        if known_binary.max() == 0:
            return torch.zeros(1, GRID_H, GRID_W)
        dist_np = distance_transform_edt(1.0 - known_binary)
        dist_max = dist_np.max()
        if dist_max > 0:
            dist_np = dist_np / dist_max
        return torch.from_numpy(dist_np.astype(np.float32)).unsqueeze(0)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x0, t, noise = self.base[idx]  # x0: standardized GT (2, H, W)

        # 1. Random density
        known_frac = random.choice(self.known_fracs)

        # 2. Random mask (different every call!)
        mask_1ch = self._generate_random_mask(known_frac)  # (1, H, W)

        # 3. Compute GP posterior for this specific mask
        x0_raw = self.raw_data[idx].unsqueeze(0)  # (1, 2, H, W)
        mask_2ch = mask_1ch.unsqueeze(0).expand(-1, 2, -1, -1)  # (1, 2, H, W)

        gp_mean_raw, gp_var_raw = gp_fill(
            x0_raw,
            mask_2ch,
            lengthscale=self.gp_params["lengthscale"],
            variance=self.gp_params["variance"],
            noise=self.gp_params["noise"],
            use_double=False,  # float32 OK for small matrices (≤41×41)
            kernel_type=self.gp_params["kernel_type"],
            coord_system=self.gp_params["coord_system"],
            return_variance=True,
        )
        # gp_mean_raw: (1, 2, H, W), gp_var_raw: (1, 2, H, W)

        # 4. Standardize GP mean (same normalization as training targets)
        gp_std = self.standardizer(gp_mean_raw.squeeze(0))  # (2, H, W)

        # 5. GP variance: max over u,v channels → single uncertainty channel
        gp_var_max = gp_var_raw.squeeze(0).max(dim=0, keepdim=True).values  # (1, H, W)
        var_max_val = gp_var_max.max()
        if var_max_val > 0:
            gp_var_max = gp_var_max / var_max_val
        gp_var_max = gp_var_max.float()

        # 6. Distance transform
        dist = self._compute_distance_map(mask_1ch)  # (1, H, W)

        ocean = self.ocean_mask  # (1, H, W)

        # ── Velocity-field-aware augmentation ─────────────────────
        if self.augment:
            flip_h = random.random() < 0.5
            flip_v = random.random() < 0.5
            if flip_h or flip_v:
                x0 = x0.clone()
                gp_std = gp_std.clone()
                gp_var_max = gp_var_max.clone()
                dist = dist.clone()
                ocean = ocean.clone()
                mask_1ch = mask_1ch.clone()
                noise = noise.clone()
                if flip_h:
                    x0 = x0.flip(-1);  x0[0] = -x0[0]
                    gp_std = gp_std.flip(-1);  gp_std[0] = -gp_std[0]
                    noise = noise.flip(-1); noise[0] = -noise[0]
                    gp_var_max = gp_var_max.flip(-1)
                    dist = dist.flip(-1)
                    ocean = ocean.flip(-1)
                    mask_1ch = mask_1ch.flip(-1)
                if flip_v:
                    x0 = x0.flip(-2);  x0[1] = -x0[1]
                    gp_std = gp_std.flip(-2);  gp_std[1] = -gp_std[1]
                    noise = noise.flip(-2); noise[1] = -noise[1]
                    gp_var_max = gp_var_max.flip(-2)
                    dist = dist.flip(-2)
                    ocean = ocean.flip(-2)
                    mask_1ch = mask_1ch.flip(-2)

        # Build dense conditioning context: (5, H, W)
        known_context = torch.cat([gp_std, gp_var_max, dist, ocean], dim=0)

        return x0, gp_std, t, noise, mask_1ch, known_context
