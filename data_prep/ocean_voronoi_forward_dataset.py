"""Voronoi-forward inpainting dataset.

Computes Voronoi (nearest-neighbour) interpolation **on-the-fly** for each
sample, with a fresh random sparse mask every time.  This is the Voronoi
analogue of ``OceanGPForwardDataset`` with GP-forward mode, but much cheaper
since Voronoi is O(N log N) cKDTree lookups instead of O(N^3) GP solves.

Training contract (voronoi_forward):
    Forward diffusion noises the **Voronoi fill** instead of GT:
        x_t = √ᾱ_t · voronoi + √(1-ᾱ_t) · ε
    Prediction target is clean x₀  →  requires ``prediction_target: x0``.
    The model learns to map noised-Voronoi → clean-GT, so at inference time
    it is specifically adapted to refine Voronoi initializations.

Returns 6-tuples (same shape as OceanGPForwardDataset for compatibility):

    (x0, voronoi_std, t, noise, mask, known_values)

where:
    x0           – standardized ground truth   (2, 64, 128)  [prediction target]
    voronoi_std  – standardized Voronoi fill   (2, 64, 128)  [noised by training loop]
    t            – random diffusion timestep   (scalar)
    noise        – noise sample                (2, 64, 128)
    mask         – observation mask            (1, 64, 128)  [1=missing, 0=known]
    known_values – sparse GT observations      (2, 64, 128)  [x0 * (1-mask)]
"""

import random
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.spatial import cKDTree

# Ocean domain: physical pixels within the (64, 128) padded grid
OCEAN_H, OCEAN_W = 44, 94


class OceanVoronoiForwardDataset(Dataset):
    """On-the-fly Voronoi-forward dataset for training.

    Each ``__getitem__`` call:
      1. Gets (x0, t, noise) from the base dataset (standardized, 2×64×128)
      2. Unstandardizes to raw (u, v) in physical space
      3. Generates a random sparse observation mask within the ocean domain
      4. Computes Voronoi (nearest-neighbour) fill from observed pixels
      5. Re-standardizes the Voronoi field
      6. Returns 6-tuple compatible with GP-forward training loop
    """

    def __init__(
        self,
        base_dataset,
        raw_data,
        standardizer,
        known_fracs=(0.001, 0.005, 0.01, 0.02, 0.05),
        augment=False,
    ):
        """
        Args:
            base_dataset: OceanImageDataset (returns (x0, t, noise), x0 is 2×64×128 std)
            raw_data: (N, 2, 64, 128) tensor — raw (unstandardized) velocity fields,
                      zero-padded outside ocean region (44×94).
            standardizer: callable — e.g. ZScoreStandardizer or UnifiedZScore
            known_fracs: tuple/list of coverage fractions to sample from.
                         Each call randomly picks one and places that fraction
                         of ocean pixels as observations.
            augment: if True, apply velocity-field-aware flips
        """
        assert len(base_dataset) == len(raw_data), (
            f"Dataset has {len(base_dataset)} samples but raw_data has "
            f"{len(raw_data)}"
        )
        self.base = base_dataset
        self.raw_data = raw_data  # (N, 2, 64, 128) in physical units
        self.standardizer = standardizer
        self.known_fracs = list(known_fracs)
        self.augment = augment

        # Pre-compute ocean mask from the first sample
        # Ocean cells have non-zero velocity; land padding is zero
        sample_raw = raw_data[0]  # (2, 64, 128)
        mag = (sample_raw[:2, :OCEAN_H, :OCEAN_W] ** 2).sum(dim=0).sqrt()
        self.ocean_mask_np = (mag > 1e-6).numpy()  # (44, 94) bool
        self.ocean_indices = np.argwhere(self.ocean_mask_np)  # (M, 2) — [row, col]
        self.n_ocean = len(self.ocean_indices)
        assert self.n_ocean > 0, "No ocean pixels found in raw data"

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x0, t, noise = self.base[idx]  # x0: standardized (2, 64, 128)

        # ── Get raw (physical-space) velocity for this sample ────
        raw = self.raw_data[idx]  # (2, 64, 128)

        # ── Random sparse observation mask ────────────────────────
        frac = random.choice(self.known_fracs)
        n_obs = max(3, int(self.n_ocean * frac))  # at least 3 observations
        n_obs = min(n_obs, self.n_ocean)

        chosen = np.random.choice(self.n_ocean, size=n_obs, replace=False)
        obs_yx = self.ocean_indices[chosen]  # (n_obs, 2)

        # Build binary observation mask in padded grid: 1=observed, 0=missing
        obs_mask_hw = np.zeros((OCEAN_H, OCEAN_W), dtype=np.float32)
        obs_mask_hw[obs_yx[:, 0], obs_yx[:, 1]] = 1.0

        # ── Voronoi (nearest-neighbour) fill ──────────────────────
        tree = cKDTree(obs_yx.astype(np.float64))
        gy, gx = np.mgrid[0:OCEAN_H, 0:OCEAN_W]
        grid_coords = np.stack([gy.ravel(), gx.ravel()], axis=1).astype(np.float64)
        _, nn_idx = tree.query(grid_coords, k=1)
        nn_idx = nn_idx.reshape(OCEAN_H, OCEAN_W)

        # Extract observed values in raw space
        raw_ocean = raw[:, :OCEAN_H, :OCEAN_W].numpy()  # (2, 44, 94)
        obs_u = raw_ocean[0, obs_yx[:, 0], obs_yx[:, 1]]
        obs_v = raw_ocean[1, obs_yx[:, 0], obs_yx[:, 1]]

        # Fill with nearest-neighbour values (only in ocean cells)
        vor_u = obs_u[nn_idx] * self.ocean_mask_np
        vor_v = obs_v[nn_idx] * self.ocean_mask_np

        # Build padded Voronoi field (2, 64, 128)
        vor_raw = np.zeros((2, 64, 128), dtype=np.float32)
        vor_raw[0, :OCEAN_H, :OCEAN_W] = vor_u
        vor_raw[1, :OCEAN_H, :OCEAN_W] = vor_v
        vor_raw_t = torch.from_numpy(vor_raw)

        # Standardize the Voronoi field
        vor_std = self.standardizer(vor_raw_t.unsqueeze(0)).squeeze(0)  # (2, 64, 128)

        # ── Build inpainting mask (1=missing, 0=known) ───────────
        # In the padded (64, 128) grid
        mask_64 = np.ones((1, 64, 128), dtype=np.float32)
        mask_64[0, :OCEAN_H, :OCEAN_W] = 1.0 - obs_mask_hw  # 0 where observed
        mask = torch.from_numpy(mask_64)

        # Known values: sparse GT observations in standardized space
        known = x0[:2] * (1.0 - mask)  # (2, 64, 128), zeros where missing

        # ── Velocity-field-aware augmentation ─────────────────────
        if self.augment:
            flip_h = random.random() < 0.5
            flip_v = random.random() < 0.5
            if flip_h or flip_v:
                x0 = x0.clone()
                vor_std = vor_std.clone()
                if flip_h:
                    x0 = x0.flip(-1);  x0[0] = -x0[0]
                    vor_std = vor_std.flip(-1);  vor_std[0] = -vor_std[0]
                    noise = noise.flip(-1);  noise[0] = -noise[0]
                    mask = mask.flip(-1)
                    known = known.flip(-1);  known[0] = -known[0]
                if flip_v:
                    x0 = x0.flip(-2);  x0[1] = -x0[1]
                    vor_std = vor_std.flip(-2);  vor_std[1] = -vor_std[1]
                    noise = noise.flip(-2);  noise[1] = -noise[1]
                    mask = mask.flip(-2)
                    known = known.flip(-2);  known[1] = -known[1]

        return x0, vor_std, t, noise, mask, known
