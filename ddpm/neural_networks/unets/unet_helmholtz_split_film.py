"""FiLM-conditioned Helmholtz dual-head UNet with split high-resolution decoder.

Combines two proven ideas:
  1. Helmholtz Split architecture (unet_helmholtz_split.py) — dual decoder
     heads producing ψ (stream function → curl → v_sol) and φ (velocity
     potential → grad → v_irr), with independent high-res decoders
  2. AdaGN-style FiLM conditioning — Adaptive Group Normalization with
     Feature-wise Linear Modulation injects observation info [mask(1ch),
     known_u(1ch), known_v(1ch)] at every resolution level, rather than
     concatenating extra input channels

Why AdaGN-FiLM over concat:
  - The UNet's encoder processes only x_t (2ch), keeping the noisy-signal
    pathway clean and identical to the unconditional version
  - Conditioning is pooled to channel-wise vectors and modulates via
    (1+γ)·GroupNorm(h)+β — stable because: (a) no per-pixel drift,
    (b) features are normalized before scaling, (c) residual γ=0 init
  - FiLM layers start as pure GroupNorm (γ=0, β=0) and smoothly learn
    to exploit conditioning

Architecture:
  Input:  (N, 5, H, W) = [x_t(2ch), mask(1ch), cond_u(1ch), cond_v(1ch)]
  Cond encoder: 3ch → multi-scale features at 4 resolution levels + bottleneck
  Shared: encoder → bottleneck → dec4 (8×16) → dec3 (16×32)   [all FiLM'd]
  ψ branch: dec2_psi (32×64) → dec1_psi (64×128) → psi_conv → curl → v_sol
  φ branch: dec2_phi (32×64) → dec1_phi (64×128) → phi_conv → grad → v_irr
  Output: (N, 2, H, W) — v = curl(ψ) + grad(φ)

Conditioning channels:
  The conditioning [mask, cond_u, cond_v] should be **dense** signals.
  Use Voronoi-interpolated fields rather than sparse known observations —
  with 99%+ missing data the sparse signal washes out in the CNN encoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ddpm.neural_networks.unets.unet_xl_attn import (
    sinusoidal_embedding,
    ResAttnBlock,
)


# ─── FiLM building blocks ──────────────────────────────────────────────────


class FiLMLayer(nn.Module):
    """Adaptive Group Normalization + FiLM (AdaGN-style).

    Stable modulation via three design choices from ADM / DiT / Palette:
      1. Pool spatial conditioning → channel-wise vectors (no per-pixel drift)
      2. GroupNorm on features before modulation (bounded activations)
      3. Residual formulation: (1 + γ) · GroupNorm(h) + β  (γ init 0 → identity)
    """

    def __init__(self, cond_channels, feature_channels, num_groups=32):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, feature_channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.scale_fc = nn.Linear(cond_channels, feature_channels)
        self.shift_fc = nn.Linear(cond_channels, feature_channels)

        # γ = 0, β = 0 at init → (1+0)·norm(h)+0 = norm(h) ≈ identity
        nn.init.zeros_(self.scale_fc.weight)
        nn.init.zeros_(self.scale_fc.bias)
        nn.init.zeros_(self.shift_fc.weight)
        nn.init.zeros_(self.shift_fc.bias)

    def forward(self, h, cond):
        cond_vec = self.pool(cond).flatten(1)                   # (B, C_cond)
        gamma = self.scale_fc(cond_vec)[:, :, None, None]       # (B, C_feat, 1, 1)
        beta = self.shift_fc(cond_vec)[:, :, None, None]
        return (1 + gamma) * self.norm(h) + beta


class HelmholtzCondEncoder(nn.Module):
    """Encodes [mask(1ch), known_u(1ch), known_v(1ch)] into multi-scale features.

    Produces features matching the Helmholtz Split UNet's 5 resolution levels:
        c1: (N,  64,  64, 128)   ← enc1 level
        c2: (N, 128,  32,  64)   ← enc2 level
        c3: (N, 256,  16,  32)   ← enc3 level
        c4: (N, 256,   8,  16)   ← enc4 level
        c5: (N, 256,   4,   8)   ← bottleneck level
    """

    def __init__(self, ch=(64, 128, 256, 256)):
        super().__init__()
        # Level 1: 64×128 → 64ch
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, ch[0], 3, 1, 1), nn.SiLU(),
            nn.Conv2d(ch[0], ch[0], 3, 1, 1), nn.SiLU(),
        )
        # Level 2: 32×64 → 128ch
        self.enc2 = nn.Sequential(
            nn.Conv2d(ch[0], ch[1], 4, 2, 1), nn.SiLU(),
            nn.Conv2d(ch[1], ch[1], 3, 1, 1), nn.SiLU(),
        )
        # Level 3: 16×32 → 256ch
        self.enc3 = nn.Sequential(
            nn.Conv2d(ch[1], ch[2], 4, 2, 1), nn.SiLU(),
            nn.Conv2d(ch[2], ch[2], 3, 1, 1), nn.SiLU(),
        )
        # Level 4: 8×16 → 256ch
        self.enc4 = nn.Sequential(
            nn.Conv2d(ch[2], ch[3], 4, 2, 1), nn.SiLU(),
            nn.Conv2d(ch[3], ch[3], 3, 1, 1), nn.SiLU(),
        )
        # Level 5 (bottleneck): 4×8 → 256ch
        self.enc5 = nn.Sequential(
            nn.Conv2d(ch[3], ch[3], 4, 2, 1), nn.SiLU(),
            nn.Conv2d(ch[3], ch[3], 3, 1, 1), nn.SiLU(),
        )

    def forward(self, cond):
        c1 = self.enc1(cond)    # (N,  64, 64, 128)
        c2 = self.enc2(c1)      # (N, 128, 32, 64)
        c3 = self.enc3(c2)      # (N, 256, 16, 32)
        c4 = self.enc4(c3)      # (N, 256, 8,  16)
        c5 = self.enc5(c4)      # (N, 256, 4,  8)
        return c1, c2, c3, c4, c5


# ─── Main model ────────────────────────────────────────────────────────────


class MyUNet_Helmholtz_Split_FiLM(nn.Module):
    """AdaGN-FiLM-conditioned Helmholtz UNet with independent high-res decoders.

    Input:  (N, 5, H, W) = [x_t(2ch), mask(1ch), cond_u(1ch), cond_v(1ch)]
    Output: (N, 2, H, W) — v = curl(ψ) + grad(φ)

    Internally splits input: UNet backbone sees only x_t (2ch),
    conditioning [mask, cond] enters through AdaGN-FiLM modulation at every
    level.  Use dense Voronoi-fill fields for the conditioning channels.
    """

    def __init__(self, n_steps: int = 1000, time_emb_dim: int = 256,
                 in_channels: int = 5):
        super().__init__()
        self.in_channels = in_channels

        ch = [64, 128, 256, 256]

        # ── conditioning encoder + FiLM layers ──────────────────────
        self.cond_encoder = HelmholtzCondEncoder(ch=ch)

        # FiLM layers — encoder path (after each encoder level)
        self.film_enc1 = FiLMLayer(cond_channels=ch[0], feature_channels=ch[0])
        self.film_enc2 = FiLMLayer(cond_channels=ch[1], feature_channels=ch[1])
        self.film_enc3 = FiLMLayer(cond_channels=ch[2], feature_channels=ch[2])
        self.film_enc4 = FiLMLayer(cond_channels=ch[3], feature_channels=ch[3])

        # FiLM — bottleneck
        self.film_mid = FiLMLayer(cond_channels=ch[3], feature_channels=ch[3])

        # FiLM — shared decoder (dec4, dec3)
        self.film_dec4 = FiLMLayer(cond_channels=ch[3], feature_channels=ch[2])
        self.film_dec3 = FiLMLayer(cond_channels=ch[2], feature_channels=ch[1])

        # FiLM — ψ branch (dec2_psi, dec1_psi)
        self.film_dec2_psi = FiLMLayer(cond_channels=ch[1], feature_channels=ch[0])
        self.film_dec1_psi = FiLMLayer(cond_channels=ch[0], feature_channels=ch[0])

        # FiLM — φ branch (dec2_phi, dec1_phi)
        self.film_dec2_phi = FiLMLayer(cond_channels=ch[1], feature_channels=ch[0])
        self.film_dec1_phi = FiLMLayer(cond_channels=ch[0], feature_channels=ch[0])

        # ── time embedding ───────────────────────────────────────────
        self.time_embed_table = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed_table.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed_table.requires_grad_(False)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        # ── encoder (2-channel input: just x_t) ─────────────────────
        self.enc1 = nn.ModuleList([
            ResAttnBlock(2, ch[0], time_emb_dim, use_attn=False),
            ResAttnBlock(ch[0], ch[0], time_emb_dim, use_attn=False),
        ])
        self.down1 = nn.Conv2d(ch[0], ch[0], 4, 2, 1)

        self.enc2 = nn.ModuleList([
            ResAttnBlock(ch[0], ch[1], time_emb_dim, use_attn=False),
            ResAttnBlock(ch[1], ch[1], time_emb_dim, use_attn=False),
        ])
        self.down2 = nn.Conv2d(ch[1], ch[1], 4, 2, 1)

        self.enc3 = nn.ModuleList([
            ResAttnBlock(ch[1], ch[2], time_emb_dim, use_attn=True, num_heads=4),
            ResAttnBlock(ch[2], ch[2], time_emb_dim, use_attn=True, num_heads=4),
        ])
        self.down3 = nn.Conv2d(ch[2], ch[2], 4, 2, 1)

        self.enc4 = nn.ModuleList([
            ResAttnBlock(ch[2], ch[3], time_emb_dim, use_attn=True, num_heads=4),
            ResAttnBlock(ch[3], ch[3], time_emb_dim, use_attn=True, num_heads=4),
        ])
        self.down4 = nn.Conv2d(ch[3], ch[3], 4, 2, 1)

        # ── bottleneck ───────────────────────────────────────────────
        self.mid = nn.ModuleList([
            ResAttnBlock(ch[3], ch[3], time_emb_dim, use_attn=True, num_heads=4),
            ResAttnBlock(ch[3], ch[3], time_emb_dim, use_attn=True, num_heads=4),
        ])

        # ── shared low-res decoder (dec4 + dec3) ────────────────────
        self.up4 = nn.ConvTranspose2d(ch[3], ch[3], 4, 2, 1)
        self.dec4 = nn.ModuleList([
            ResAttnBlock(ch[3] * 2, ch[3], time_emb_dim, use_attn=True, num_heads=4),
            ResAttnBlock(ch[3], ch[2], time_emb_dim, use_attn=True, num_heads=4),
        ])

        self.up3 = nn.ConvTranspose2d(ch[2], ch[2], 4, 2, 1)
        self.dec3 = nn.ModuleList([
            ResAttnBlock(ch[2] * 2, ch[2], time_emb_dim, use_attn=True, num_heads=4),
            ResAttnBlock(ch[2], ch[1], time_emb_dim, use_attn=True, num_heads=4),
        ])

        # ── ψ branch: independent high-res decoder ──────────────────
        self.up2_psi = nn.ConvTranspose2d(ch[1], ch[1], 4, 2, 1)
        self.dec2_psi = nn.ModuleList([
            ResAttnBlock(ch[1] * 2, ch[1], time_emb_dim, use_attn=False),
            ResAttnBlock(ch[1], ch[0], time_emb_dim, use_attn=False),
        ])

        self.up1_psi = nn.ConvTranspose2d(ch[0], ch[0], 4, 2, 1)
        self.dec1_psi = nn.ModuleList([
            ResAttnBlock(ch[0] * 2, ch[0], time_emb_dim, use_attn=False),
            ResAttnBlock(ch[0], ch[0], time_emb_dim, use_attn=False),
        ])

        self.psi_norm = nn.GroupNorm(8, ch[0])
        self.psi_act = nn.SiLU()
        self.psi_conv = nn.Conv2d(ch[0], 1, 3, 1, 1)

        # ── φ branch: independent high-res decoder ──────────────────
        self.up2_phi = nn.ConvTranspose2d(ch[1], ch[1], 4, 2, 1)
        self.dec2_phi = nn.ModuleList([
            ResAttnBlock(ch[1] * 2, ch[1], time_emb_dim, use_attn=False),
            ResAttnBlock(ch[1], ch[0], time_emb_dim, use_attn=False),
        ])

        self.up1_phi = nn.ConvTranspose2d(ch[0], ch[0], 4, 2, 1)
        self.dec1_phi = nn.ModuleList([
            ResAttnBlock(ch[0] * 2, ch[0], time_emb_dim, use_attn=False),
            ResAttnBlock(ch[0], ch[0], time_emb_dim, use_attn=False),
        ])

        self.phi_norm = nn.GroupNorm(8, ch[0])
        self.phi_act = nn.SiLU()
        self.phi_conv = nn.Conv2d(ch[0], 1, 3, 1, 1)

        # Initialize φ branch output to near-zero so model starts solenoidal-biased
        nn.init.zeros_(self.phi_conv.weight)
        nn.init.zeros_(self.phi_conv.bias)

        # ── gradient kernels for φ → (u_irr, v_irr) ─────────────────
        dx = torch.tensor([[[[0.0, -0.5, 0.5]]]])
        dy = torch.tensor([[[[0.0], [-0.5], [0.5]]]])
        self.register_buffer("_dx_kernel", dx)
        self.register_buffer("_dy_kernel", dy)

        # Diagnostics
        self.last_psi = None
        self.last_phi = None
        self.last_v_sol = None
        self.last_v_irr = None

    # ── physics operators ────────────────────────────────────────────

    @staticmethod
    def _curl_from_streamfunction(psi: torch.Tensor) -> torch.Tensor:
        """Forward-diff curl: ψ (B, H+1, W+1) → velocity (B, 2, H, W)."""
        u = psi[:, 1:, :-1] - psi[:, :-1, :-1]
        v = -(psi[:, :-1, 1:] - psi[:, :-1, :-1])
        return torch.stack([u, v], dim=1)

    def _grad_potential(self, phi: torch.Tensor) -> torch.Tensor:
        """Central-diff gradient: φ (B, 1, H, W) → velocity (B, 2, H, W)."""
        u_irr = F.conv2d(phi, self._dx_kernel, padding=(0, 1))
        v_irr = F.conv2d(phi, self._dy_kernel, padding=(1, 0))
        return torch.cat([u_irr, v_irr], dim=1)

    # ── forward ──────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x: (N, 5, H, W) — [x_t(2ch), mask(1ch), cond_u(1ch), cond_v(1ch)]
               cond channels should be dense (e.g. Voronoi fill), not sparse.
            t: (N,) or (N, 1) time step indices
        Returns:
            (N, 2, H, W) predicted x₀ = curl(ψ) + grad(φ)
        """
        # Split: UNet sees only x_t; conditioning goes through FiLM
        x_t = x[:, :2]     # (N, 2, H, W) — noisy field
        cond = x[:, 2:]    # (N, 3, H, W) — [mask, cond_u, cond_v]

        # Encode conditioning at 5 resolution scales
        c1, c2, c3, c4, c5 = self.cond_encoder(cond)

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
        # h is now (B, 128, 16, 32) — shared representation

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

        psi = self.psi_conv(self.psi_act(self.psi_norm(h_psi)))  # (B, 1, H, W)
        psi = psi.squeeze(1)                                       # (B, H, W)
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

        phi = self.phi_conv(self.phi_act(self.phi_norm(h_phi)))   # (B, 1, H, W)
        v_irr = self._grad_potential(phi)

        # Store for diagnostics / penalty losses
        self.last_psi = psi
        self.last_phi = phi.squeeze(1)
        self.last_v_sol = v_sol
        self.last_v_irr = v_irr

        return v_sol + v_irr
