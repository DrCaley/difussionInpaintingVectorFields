"""Multi-resolution FiLM-conditioned Helmholtz dual-head UNet.

Same backbone as MyUNet_Helmholtz_Split_FiLM but replaces the CNN conditioning
encoder with a Feature Pyramid Network (FPN) encoder that directly processes
sparse observations at multiple scales — no Voronoi fill needed.

Key idea: at 0.5% coverage (~20 observations on a 44×94 grid), the Voronoi
nearest-neighbour fill creates large blocky cells with sharp discontinuities
at cell boundaries.  A CNN encoder faithfully encodes these artifacts.

Instead, we:
  1. Pool sparse observations to each UNet resolution level
  2. Normalize pooled values by observation density → honest local mean
  3. Process coarsest-to-finest (top-down FPN) so fine levels inherit
     global context from coarser scales that have better coverage

At 4×8 (coarsest): ~20 obs over 32 cells → ~60% cells have data
At 8×16:           ~20 obs over 128 cells → ~15% cells have data
At 64×128 (finest): 0.5% pixel coverage — but context flows down from coarser

Conditioning input should be SPARSE observations [mask, obs_u*mask, obs_v*mask],
NOT dense Voronoi fill.

Architecture (identical backbone to FiLM, only encoder differs):
  Input:  (N, 5, H, W) = [x_t(2ch), mask(1ch), sparse_u(1ch), sparse_v(1ch)]
  FPN encoder: sparse → multi-scale features at 5 resolution levels (top-down)
  Shared: encoder → bottleneck → dec4 (8×16) → dec3 (16×32)   [all FiLM'd]
  ψ branch: dec2_psi (32×64) → dec1_psi (64×128) → psi_conv → curl → v_sol
  φ branch: dec2_phi (32×64) → dec1_phi (64×128) → phi_conv → grad → v_irr
  Output: (N, 2, H, W) — v = curl(ψ) + grad(φ)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ddpm.neural_networks.unets.unet_helmholtz_split_film import (
    MyUNet_Helmholtz_Split_FiLM,
)


class MultiResCondEncoder(nn.Module):
    """Feature Pyramid conditioning encoder for sparse observations.

    Processes sparse observations at multiple scales via a top-down pathway:
    coarsest level (best coverage) processes first, then context propagates
    to finer levels via upsampling + concatenation.

    At each resolution, sparse observations are pooled to compute:
    - Normalized local mean (sum_obs / count_obs) — unbiased estimate
    - Observation density (fraction of cell observed) — uncertainty signal

    Optionally accepts a distance-to-nearest-sensor field (V-CNN style),
    which is pooled with simple avg_pool2d (it's dense, not sparse).

    Produces features matching the Helmholtz Split UNet's 5 resolution levels:
        c1: (N,  64,  64, 128)   ← enc1 level
        c2: (N, 128,  32,  64)   ← enc2 level
        c3: (N, 256,  16,  32)   ← enc3 level
        c4: (N, 256,   8,  16)   ← enc4 level
        c5: (N, 256,   4,   8)   ← bottleneck level
    """

    def __init__(self, ch=(64, 128, 256, 256), use_distance_field=False,
                 use_bathymetry=False):
        super().__init__()
        self.use_distance_field = use_distance_field
        self.use_bathymetry = use_bathymetry
        # Pooled channels: 3 base (norm_u, norm_v, density) + optional dense fields
        pool_ch = 3
        if use_distance_field:
            pool_ch += 1
        if use_bathymetry:
            pool_ch += 1

        # Top-down processing: coarsest first
        # Level 5 (bottleneck, 4×8): just pooled sparse features
        self.proc5 = nn.Sequential(
            nn.Conv2d(pool_ch, ch[3], 3, 1, 1), nn.SiLU(),
            nn.Conv2d(ch[3], ch[3], 3, 1, 1), nn.SiLU(),
        )
        # Level 4 (8×16): pooled + upsampled coarser features
        self.proc4 = nn.Sequential(
            nn.Conv2d(pool_ch + ch[3], ch[3], 3, 1, 1), nn.SiLU(),
            nn.Conv2d(ch[3], ch[3], 3, 1, 1), nn.SiLU(),
        )
        # Level 3 (16×32)
        self.proc3 = nn.Sequential(
            nn.Conv2d(pool_ch + ch[3], ch[2], 3, 1, 1), nn.SiLU(),
            nn.Conv2d(ch[2], ch[2], 3, 1, 1), nn.SiLU(),
        )
        # Level 2 (32×64)
        self.proc2 = nn.Sequential(
            nn.Conv2d(pool_ch + ch[2], ch[1], 3, 1, 1), nn.SiLU(),
            nn.Conv2d(ch[1], ch[1], 3, 1, 1), nn.SiLU(),
        )
        # Level 1 (64×128, full res)
        self.proc1 = nn.Sequential(
            nn.Conv2d(pool_ch + ch[1], ch[0], 3, 1, 1), nn.SiLU(),
            nn.Conv2d(ch[0], ch[0], 3, 1, 1), nn.SiLU(),
        )

    @staticmethod
    def _pool_sparse(obs_uv, known_mask, target_h, target_w):
        """Pool sparse observations to target resolution, normalize by density.

        Args:
            obs_uv: (B, 2, H, W) — sparse observed values (zero where unobserved)
            known_mask: (B, 1, H, W) — 1 where observed, 0 where missing
            target_h, target_w: target spatial dimensions

        Returns:
            (B, 3, target_h, target_w) — [normalized_u, normalized_v, density]
        """
        H, W = obs_uv.shape[-2:]
        kh, kw = H // target_h, W // target_w

        if kh == 1 and kw == 1:
            # Full resolution — no pooling
            density = known_mask
            normalized = obs_uv
        else:
            # Pool density and observation sums
            density = F.avg_pool2d(known_mask, (kh, kw))
            pooled = F.avg_pool2d(obs_uv, (kh, kw))
            # pooled / density = (sum_obs/area) / (count/area) = sum_obs/count
            safe_density = density.clamp(min=1e-8)
            normalized = pooled / safe_density
            # Zero out cells with no observations (avoid dividing noise by eps)
            has_obs = (density > 1e-7).float()
            normalized = normalized * has_obs

        return torch.cat([normalized, density], dim=1)  # (B, 3, tH, tW)

    @staticmethod
    def _pool_dense(field, target_h, target_w):
        """Pool a dense field to target resolution with simple avg_pool2d."""
        H, W = field.shape[-2:]
        kh, kw = H // target_h, W // target_w
        if kh == 1 and kw == 1:
            return field
        return F.avg_pool2d(field, (kh, kw))

    def forward(self, cond):
        """
        Args:
            cond: (B, C, H, W) — C=3: [missing_mask, sparse_u, sparse_v]
                   Optional dense fields appended: distance_field, bathymetry
        Returns:
            c1..c5 matching HelmholtzCondEncoder output shapes
        """
        missing_mask = cond[:, :1]
        known_mask = 1.0 - missing_mask
        obs_uv = cond[:, 1:3]  # (B, 2, H, W) — values only at known locations

        # Pool sparse observations to each resolution
        p5 = self._pool_sparse(obs_uv, known_mask, 4, 8)
        p4 = self._pool_sparse(obs_uv, known_mask, 8, 16)
        p3 = self._pool_sparse(obs_uv, known_mask, 16, 32)
        p2 = self._pool_sparse(obs_uv, known_mask, 32, 64)
        p1 = self._pool_sparse(obs_uv, known_mask, 64, 128)

        # Dense contextual fields follow sparse channels
        dense_idx = 3
        if self.use_distance_field:
            dist = cond[:, dense_idx:dense_idx + 1]
            dense_idx += 1
            p5 = torch.cat([p5, self._pool_dense(dist, 4, 8)], dim=1)
            p4 = torch.cat([p4, self._pool_dense(dist, 8, 16)], dim=1)
            p3 = torch.cat([p3, self._pool_dense(dist, 16, 32)], dim=1)
            p2 = torch.cat([p2, self._pool_dense(dist, 32, 64)], dim=1)
            p1 = torch.cat([p1, self._pool_dense(dist, 64, 128)], dim=1)

        if self.use_bathymetry:
            bathy = cond[:, dense_idx:dense_idx + 1]
            dense_idx += 1
            p5 = torch.cat([p5, self._pool_dense(bathy, 4, 8)], dim=1)
            p4 = torch.cat([p4, self._pool_dense(bathy, 8, 16)], dim=1)
            p3 = torch.cat([p3, self._pool_dense(bathy, 16, 32)], dim=1)
            p2 = torch.cat([p2, self._pool_dense(bathy, 32, 64)], dim=1)
            p1 = torch.cat([p1, self._pool_dense(bathy, 64, 128)], dim=1)

        # Top-down: coarsest first, propagate context to finer levels
        c5 = self.proc5(p5)

        c5_up = F.interpolate(c5, size=(8, 16), mode='bilinear', align_corners=False)
        c4 = self.proc4(torch.cat([p4, c5_up], dim=1))

        c4_up = F.interpolate(c4, size=(16, 32), mode='bilinear', align_corners=False)
        c3 = self.proc3(torch.cat([p3, c4_up], dim=1))

        c3_up = F.interpolate(c3, size=(32, 64), mode='bilinear', align_corners=False)
        c2 = self.proc2(torch.cat([p2, c3_up], dim=1))

        c2_up = F.interpolate(c2, size=(64, 128), mode='bilinear', align_corners=False)
        c1 = self.proc1(torch.cat([p1, c2_up], dim=1))

        return c1, c2, c3, c4, c5


class MyUNet_Helmholtz_Split_FiLM_MultiRes(MyUNet_Helmholtz_Split_FiLM):
    """FiLM-conditioned Helmholtz UNet with multi-resolution sparse conditioning.

    Identical backbone to MyUNet_Helmholtz_Split_FiLM — same encoder, decoder,
    FiLM layers, ψ/φ branches, and Helmholtz physics operators. The only
    difference is the conditioning encoder: a top-down FPN that processes raw
    sparse observations instead of a CNN that processes Voronoi fill.

    The conditioning input should be SPARSE observations
    [mask, obs_u * mask, obs_v * mask] rather than dense Voronoi fill.
    """

    def __init__(self, n_steps=1000, time_emb_dim=256, in_channels=5,
                 detach_heads: bool = False, use_distance_field: bool = False,
                 use_bathymetry: bool = False):
        super().__init__(
            n_steps=n_steps, time_emb_dim=time_emb_dim, in_channels=in_channels,
            detach_heads=detach_heads,
        )
        # Replace CNN encoder with multi-resolution FPN encoder
        self.cond_encoder = MultiResCondEncoder(
            ch=[64, 128, 256, 256], use_distance_field=use_distance_field,
            use_bathymetry=use_bathymetry,
        )
