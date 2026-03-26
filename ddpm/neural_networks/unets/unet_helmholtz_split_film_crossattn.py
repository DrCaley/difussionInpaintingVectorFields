"""Cross-attention FiLM-conditioned Helmholtz dual-head UNet.

Same backbone as MyUNet_Helmholtz_Split_FiLM but replaces the CNN conditioning
encoder with a cross-attention (Neural Process-style) encoder that processes
sparse observations as unordered point sets.

Key idea: grid-based encoders (CNN, FPN) quantize observation positions to
pixel grids. At 0.5% coverage (~20 obs on 44×94), this loses sub-pixel position
information and creates artifacts (Voronoi cells, pooling aliasing). Instead:

  1. Each observation becomes a token: MLP([r/H, c/W, u, v]) → d_model
  2. At each UNet resolution, a grid of learned query embeddings attends
     to all observation tokens via multi-head cross-attention
  3. Output: 5 feature maps matching the Helmholtz FiLM interface

Advantages:
  - Exact continuous positions preserved (no grid quantization)
  - Every query pixel sees every observation (global receptive field)
  - Naturally handles variable observation counts via attention masking
  - Acts as a universal interpolator (Neural Process / Set Function framework)

Conditioning input: same 5ch as FiLM — [x_t(2ch), mask(1ch), sparse_u(1ch), sparse_v(1ch)]
The forward() method extracts point coordinates from the mask channel on-the-fly.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ddpm.neural_networks.unets.unet_helmholtz_split_film import (
    MyUNet_Helmholtz_Split_FiLM,
    FiLMLayer,
)
from ddpm.neural_networks.unets.unet_xl_attn import sinusoidal_embedding


class CrossAttnCondEncoder(nn.Module):
    """Cross-attention conditioning encoder for sparse point observations.

    Processes observations as an unordered set of tokens, then generates
    dense feature maps at 5 UNet resolution levels via cross-attention.

    Architecture:
        Observation MLP: [r_norm, c_norm, u, v] → d_model token
        Per-level: learned grid queries (H_l × W_l × d_l) attend to obs tokens
                   via multi-head cross-attention, then project to output channels

    Produces features matching the Helmholtz Split UNet's 5 resolution levels:
        c1: (N,  64,  64, 128)   ← enc1 level
        c2: (N, 128,  32,  64)   ← enc2 level
        c3: (N, 256,  16,  32)   ← enc3 level
        c4: (N, 256,   8,  16)   ← enc4 level
        c5: (N, 256,   4,   8)   ← bottleneck level
    """

    def __init__(self, d_model=128, nhead=4, ch=(64, 128, 256, 256)):
        super().__init__()
        self.d_model = d_model

        # Per-observation MLP: [r/H, c/W, u, v] → d_model
        self.obs_mlp = nn.Sequential(
            nn.Linear(4, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Resolution levels: (H, W, out_channels)
        self.levels = [(64, 128, ch[0]), (32, 64, ch[1]),
                       (16, 32, ch[2]), (8, 16, ch[3]), (4, 8, ch[3])]

        # Per-level components
        self.query_embeds = nn.ParameterList()
        self.cross_attns = nn.ModuleList()
        self.out_projs = nn.ModuleList()

        for h, w, out_ch in self.levels:
            # Learned query embeddings for this grid resolution
            self.query_embeds.append(nn.Parameter(torch.randn(h * w, d_model) * 0.02))

            # Multi-head cross-attention: queries=grid, keys/values=observations
            self.cross_attns.append(
                nn.MultiheadAttention(d_model, nhead, batch_first=True)
            )

            # Project from d_model to output channel count + refine
            self.out_projs.append(nn.Sequential(
                nn.Linear(d_model, out_ch),
                nn.SiLU(),
                nn.Linear(out_ch, out_ch),
            ))

    def forward(self, coords, values, padding_mask, batch_size):
        """
        Args:
            coords:  (B, N_max, 2) — normalized [r/H, c/W] in [0, 1]
            values:  (B, N_max, 2) — [u, v] observation values (standardized)
            padding_mask: (B, N_max) — True for padding positions to ignore
            batch_size: int

        Returns:
            c1..c5 matching HelmholtzCondEncoder output shapes
        """
        # Build observation tokens: (B, N_max, d_model)
        obs_input = torch.cat([coords, values], dim=-1)  # (B, N_max, 4)
        obs_tokens = self.obs_mlp(obs_input)  # (B, N_max, d_model)

        outputs = []
        for i, (h, w, out_ch) in enumerate(self.levels):
            # Query embeddings: (1, H*W, d_model) → (B, H*W, d_model)
            queries = self.query_embeds[i].unsqueeze(0).expand(batch_size, -1, -1)

            # Cross-attention: grid queries attend to observation tokens
            # key_padding_mask: True = ignore that key position
            attn_out, _ = self.cross_attns[i](
                queries, obs_tokens, obs_tokens,
                key_padding_mask=padding_mask,
            )  # (B, H*W, d_model)

            # Project to output channels and reshape to spatial
            feat = self.out_projs[i](attn_out)  # (B, H*W, out_ch)
            feat = feat.permute(0, 2, 1).reshape(batch_size, out_ch, h, w)
            outputs.append(feat)

        return tuple(outputs)  # c1, c2, c3, c4, c5


class MyUNet_Helmholtz_Split_FiLM_CrossAttn(MyUNet_Helmholtz_Split_FiLM):
    """Cross-attention conditioned Helmholtz UNet.

    Inherits the full UNet backbone + FiLM layers from the parent class.
    Replaces the CNN cond_encoder with CrossAttnCondEncoder.

    Input:  (N, 5, H, W) = [x_t(2ch), mask(1ch), sparse_u(1ch), sparse_v(1ch)]
    The mask and sparse channels are used to extract point observations on-the-fly.
    Output: (N, 2, H, W) — v = curl(ψ) + grad(φ)
    """

    def __init__(self, n_steps: int = 1000, time_emb_dim: int = 256,
                 in_channels: int = 5, d_model: int = 128, nhead: int = 4,
                 detach_heads: bool = False):
        super().__init__(n_steps=n_steps, time_emb_dim=time_emb_dim,
                         in_channels=in_channels, detach_heads=detach_heads)

        # Replace the CNN conditioning encoder with cross-attention
        ch = [64, 128, 256, 256]
        self.cond_encoder = CrossAttnCondEncoder(
            d_model=d_model, nhead=nhead, ch=ch,
        )

    def _extract_points(self, cond):
        """Extract point observations from grid-format conditioning.

        Args:
            cond: (B, 3, H, W) — [miss_mask(1ch), sparse_u(1ch), sparse_v(1ch)]

        Returns:
            coords: (B, N_max, 2) — normalized [r/H, c/W]
            values: (B, N_max, 2) — [u, v] at observed locations
            padding_mask: (B, N_max) — True for padding positions
        """
        B, _, H, W = cond.shape
        miss_mask = cond[:, 0]        # (B, H, W) — 1=missing, 0=known
        known_mask = 1.0 - miss_mask  # 1=known, 0=missing
        sparse_u = cond[:, 1]         # (B, H, W)
        sparse_v = cond[:, 2]         # (B, H, W)

        # Find max number of observations across batch for padding
        counts = []
        all_coords = []
        all_values = []
        for b in range(B):
            obs_idx = known_mask[b].nonzero(as_tuple=False)  # (N_b, 2) — [row, col]
            counts.append(len(obs_idx))
            if len(obs_idx) > 0:
                r_norm = obs_idx[:, 0].float() / H
                c_norm = obs_idx[:, 1].float() / W
                u_vals = sparse_u[b, obs_idx[:, 0], obs_idx[:, 1]]
                v_vals = sparse_v[b, obs_idx[:, 0], obs_idx[:, 1]]
                all_coords.append(torch.stack([r_norm, c_norm], dim=-1))
                all_values.append(torch.stack([u_vals, v_vals], dim=-1))
            else:
                # No observations — will be fully masked
                all_coords.append(torch.zeros(1, 2, device=cond.device))
                all_values.append(torch.zeros(1, 2, device=cond.device))

        N_max = max(counts) if max(counts) > 0 else 1

        # Pad to N_max and build padding mask
        coords = torch.zeros(B, N_max, 2, device=cond.device)
        values = torch.zeros(B, N_max, 2, device=cond.device)
        padding_mask = torch.ones(B, N_max, dtype=torch.bool, device=cond.device)

        for b in range(B):
            n = counts[b]
            if n > 0:
                coords[b, :n] = all_coords[b]
                values[b, :n] = all_values[b]
                padding_mask[b, :n] = False  # Real observations: don't mask

        return coords, values, padding_mask

    def forward(self, x: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x: (N, 5, H, W) — [x_t(2ch), mask(1ch), sparse_u(1ch), sparse_v(1ch)]
            t: (N,) or (N, 1) time step indices
        Returns:
            (N, 2, H, W) predicted x₀ = curl(ψ) + grad(φ)
        """
        B = x.shape[0]
        x_t = x[:, :2]   # (B, 2, H, W)
        cond = x[:, 2:]   # (B, 3, H, W) — [mask, sparse_u, sparse_v]

        # Extract point observations from grid
        coords, values, padding_mask = self._extract_points(cond)

        # Cross-attention conditioning encoder
        c1, c2, c3, c4, c5 = self.cond_encoder(coords, values, padding_mask, B)

        # Time embedding
        t_emb = self.time_embed_table(t)
        if t_emb.dim() == 3:
            t_emb = t_emb.squeeze(1)
        t_emb = self.time_mlp(t_emb)

        # ── Encoder with FiLM ────────────────────────────────────────
        h = x_t
        for block in self.enc1:
            h = block(h, t_emb)
        h = self.film_enc1(h, c1)
        skip1 = h

        h = self.down1(h)
        for block in self.enc2:
            h = block(h, t_emb)
        h = self.film_enc2(h, c2)
        skip2 = h

        h = self.down2(h)
        for block in self.enc3:
            h = block(h, t_emb)
        h = self.film_enc3(h, c3)
        skip3 = h

        h = self.down3(h)
        for block in self.enc4:
            h = block(h, t_emb)
        h = self.film_enc4(h, c4)
        skip4 = h

        h = self.down4(h)

        # ── Bottleneck with FiLM ─────────────────────────────────────
        for block in self.mid:
            h = block(h, t_emb)
        h = self.film_mid(h, c5)

        # ── Shared low-res decoder with FiLM (dec4 → dec3) ──────────
        h = self.up4(h)
        h = torch.cat([skip4, h], dim=1)
        for block in self.dec4:
            h = block(h, t_emb)
        h = self.film_dec4(h, c4)

        h = self.up3(h)
        h = torch.cat([skip3, h], dim=1)
        for block in self.dec3:
            h = block(h, t_emb)
        h = self.film_dec3(h, c3)

        # ── ψ branch with FiLM ──────────────────────────────────────
        h_psi = self.up2_psi(h)
        h_psi = torch.cat([skip2, h_psi], dim=1)
        for block in self.dec2_psi:
            h_psi = block(h_psi, t_emb)
        h_psi = self.film_dec2_psi(h_psi, c2)

        h_psi = self.up1_psi(h_psi)
        h_psi = torch.cat([skip1, h_psi], dim=1)
        for block in self.dec1_psi:
            h_psi = block(h_psi, t_emb)
        h_psi = self.film_dec1_psi(h_psi, c1)

        psi = self.psi_conv(self.psi_act(self.psi_norm(h_psi)))
        psi = psi.squeeze(1)
        psi_padded = F.pad(psi, (0, 1, 0, 1), mode="constant", value=0.0)
        v_sol = self._curl_from_streamfunction(psi_padded)

        # ── φ branch with FiLM ──────────────────────────────────────
        h_phi = self.up2_phi(h)
        h_phi = torch.cat([skip2, h_phi], dim=1)
        for block in self.dec2_phi:
            h_phi = block(h_phi, t_emb)
        h_phi = self.film_dec2_phi(h_phi, c2)

        h_phi = self.up1_phi(h_phi)
        h_phi = torch.cat([skip1, h_phi], dim=1)
        for block in self.dec1_phi:
            h_phi = block(h_phi, t_emb)
        h_phi = self.film_dec1_phi(h_phi, c1)

        phi = self.phi_conv(self.phi_act(self.phi_norm(h_phi)))
        v_irr = self._grad_potential(phi)

        # Store for diagnostics
        self.last_psi = psi
        self.last_phi = phi.squeeze(1)
        self.last_v_sol = v_sol
        self.last_v_irr = v_irr

        if self.detach_heads:
            return v_sol.detach() + v_irr.detach()
        return v_sol + v_irr
