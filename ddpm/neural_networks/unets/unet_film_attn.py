"""FiLM-conditioned UNet with self-attention for ocean velocity inpainting.

Combines the best of both worlds:
  - **MyUNet_Attn** backbone: ResBlocks with AdaGN time conditioning,
    self-attention at 16×32, 8×16, and 4×8 resolutions (~18M params)
  - **FiLM conditioning**: a separate encoder processes [mask(1ch),
    known_values(2ch)] into multi-scale features that modulate the
    main UNet via learned scale (γ) and shift (β) at every level.

Interface: forward(x, t) where x is (N, 5, H, W) = [x_t, mask, known],
identical to MyUNet_FiLM so training code needs no changes — just set
``unet_type: film_attn`` in the config.

Architecture
────────────
  Resolution   Channels  Attention  FiLM
  64×128          64        ✗         ★
  32×64          128        ✗         ★
  16×32          256        ★         ★
   8×16          256        ★         ★
   4×8 (btl)    256        ★         ★

Conditioning encoder mirrors the UNet resolution path, producing
features at each scale that FiLM layers use to modulate activations.
FiLM layers are zero-initialized (γ=1, β=0) so the model starts
as if unconditioned and gradually learns to use the mask/known info.
"""

import torch
import torch.nn as nn

from ddpm.neural_networks.unets.unet_xl_attn import (
    sinusoidal_embedding,
    ResAttnBlock,
)


# ── FiLM building blocks ────────────────────────────────────────────


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: h_out = γ(cond) * h + β(cond).

    Initialized to identity (γ=1, β=0) so the network starts as if
    unconditioned and gradually learns to use the conditioning signal.
    """

    def __init__(self, cond_channels: int, feature_channels: int):
        super().__init__()
        self.scale_conv = nn.Conv2d(cond_channels, feature_channels, 1)
        self.shift_conv = nn.Conv2d(cond_channels, feature_channels, 1)

        # Identity init
        nn.init.zeros_(self.scale_conv.weight)
        nn.init.ones_(self.scale_conv.bias)
        nn.init.zeros_(self.shift_conv.weight)
        nn.init.zeros_(self.shift_conv.bias)

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma = self.scale_conv(cond)
        beta = self.shift_conv(cond)
        return gamma * h + beta


class ConditioningEncoderAttn(nn.Module):
    """Encodes [mask(1ch), known_values(2ch)] into features matching
    the attention UNet's resolution/channel schedule.

    Produces:
        c1: (N,  64, 64, 128)
        c2: (N, 128, 32,  64)
        c3: (N, 256, 16,  32)
        c4: (N, 256,  8,  16)
        c5: (N, 256,  4,   8)   ← bottleneck
    """

    def __init__(self):
        super().__init__()
        # Level 1: 64×128, 64 ch
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
        )
        # Level 2: 32×64, 128 ch
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
        )
        # Level 3: 16×32, 256 ch
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.GroupNorm(8, 256),
            nn.SiLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.GroupNorm(8, 256),
            nn.SiLU(),
        )
        # Level 4: 8×16, 256 ch
        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 256, 4, 2, 1),
            nn.GroupNorm(8, 256),
            nn.SiLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.GroupNorm(8, 256),
            nn.SiLU(),
        )
        # Level 5 (bottleneck): 4×8, 256 ch
        self.enc5 = nn.Sequential(
            nn.Conv2d(256, 256, 4, 2, 1),
            nn.GroupNorm(8, 256),
            nn.SiLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.GroupNorm(8, 256),
            nn.SiLU(),
        )

    def forward(self, cond: torch.Tensor):
        c1 = self.enc1(cond)    # (N,  64, 64, 128)
        c2 = self.enc2(c1)     # (N, 128, 32,  64)
        c3 = self.enc3(c2)     # (N, 256, 16,  32)
        c4 = self.enc4(c3)     # (N, 256,  8,  16)
        c5 = self.enc5(c4)     # (N, 256,  4,   8)
        return c1, c2, c3, c4, c5


# ── Main network ────────────────────────────────────────────────────


class MyUNet_FiLM_Attn(nn.Module):
    """Attention UNet with FiLM conditioning for inpainting.

    Same architecture as MyUNet_Attn but with FiLM modulation from
    mask + known values injected after each encoder/decoder level.

    Input:  (N, 5, 64, 128)  — [x_t(2ch), mask(1ch), known_values(2ch)]
    Output: (N, 2, 64, 128)  — predicted ε or x̂₀
    """

    def __init__(self, n_steps: int = 1000, time_emb_dim: int = 256,
                 in_channels: int = 5):
        super().__init__()
        self.in_channels = in_channels   # for compatibility checks

        ch = [64, 128, 256, 256]

        # ── Conditioning encoder ─────────────────────────────────────
        self.cond_encoder = ConditioningEncoderAttn()

        # FiLM layers — encoder path
        self.film_enc1 = FiLMLayer(cond_channels=ch[0], feature_channels=ch[0])
        self.film_enc2 = FiLMLayer(cond_channels=ch[1], feature_channels=ch[1])
        self.film_enc3 = FiLMLayer(cond_channels=ch[2], feature_channels=ch[2])
        self.film_enc4 = FiLMLayer(cond_channels=ch[3], feature_channels=ch[3])
        self.film_mid  = FiLMLayer(cond_channels=ch[3], feature_channels=ch[3])

        # FiLM layers — decoder path (reuse encoder cond features)
        self.film_dec4 = FiLMLayer(cond_channels=ch[3], feature_channels=ch[2])
        self.film_dec3 = FiLMLayer(cond_channels=ch[2], feature_channels=ch[1])
        self.film_dec2 = FiLMLayer(cond_channels=ch[1], feature_channels=ch[0])
        self.film_dec1 = FiLMLayer(cond_channels=ch[0], feature_channels=ch[0])

        # ── Time embedding ───────────────────────────────────────────
        self.time_embed_table = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed_table.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed_table.requires_grad_(False)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        # ── Encoder ──────────────────────────────────────────────────
        # Level 1: 64×128, 64 ch (no attention)
        self.enc1 = nn.ModuleList([
            ResAttnBlock(2, ch[0], time_emb_dim, use_attn=False),
            ResAttnBlock(ch[0], ch[0], time_emb_dim, use_attn=False),
        ])
        self.down1 = nn.Conv2d(ch[0], ch[0], 4, 2, 1)       # → 32×64

        # Level 2: 32×64, 128 ch (no attention)
        self.enc2 = nn.ModuleList([
            ResAttnBlock(ch[0], ch[1], time_emb_dim, use_attn=False),
            ResAttnBlock(ch[1], ch[1], time_emb_dim, use_attn=False),
        ])
        self.down2 = nn.Conv2d(ch[1], ch[1], 4, 2, 1)       # → 16×32

        # Level 3: 16×32, 256 ch ★ attention
        self.enc3 = nn.ModuleList([
            ResAttnBlock(ch[1], ch[2], time_emb_dim, use_attn=True, num_heads=4),
            ResAttnBlock(ch[2], ch[2], time_emb_dim, use_attn=True, num_heads=4),
        ])
        self.down3 = nn.Conv2d(ch[2], ch[2], 4, 2, 1)       # → 8×16

        # Level 4: 8×16, 256 ch ★ attention
        self.enc4 = nn.ModuleList([
            ResAttnBlock(ch[2], ch[3], time_emb_dim, use_attn=True, num_heads=4),
            ResAttnBlock(ch[3], ch[3], time_emb_dim, use_attn=True, num_heads=4),
        ])
        self.down4 = nn.Conv2d(ch[3], ch[3], 4, 2, 1)       # → 4×8

        # ── Bottleneck: 4×8, 256 ch ★ attention ─────────────────────
        self.mid = nn.ModuleList([
            ResAttnBlock(ch[3], ch[3], time_emb_dim, use_attn=True, num_heads=4),
            ResAttnBlock(ch[3], ch[3], time_emb_dim, use_attn=True, num_heads=4),
        ])

        # ── Decoder ──────────────────────────────────────────────────
        self.up4 = nn.ConvTranspose2d(ch[3], ch[3], 4, 2, 1)    # 4×8 → 8×16
        self.dec4 = nn.ModuleList([
            ResAttnBlock(ch[3] * 2, ch[3], time_emb_dim, use_attn=True, num_heads=4),
            ResAttnBlock(ch[3], ch[2], time_emb_dim, use_attn=True, num_heads=4),
        ])

        self.up3 = nn.ConvTranspose2d(ch[2], ch[2], 4, 2, 1)    # 8×16 → 16×32
        self.dec3 = nn.ModuleList([
            ResAttnBlock(ch[2] * 2, ch[2], time_emb_dim, use_attn=True, num_heads=4),
            ResAttnBlock(ch[2], ch[1], time_emb_dim, use_attn=True, num_heads=4),
        ])

        self.up2 = nn.ConvTranspose2d(ch[1], ch[1], 4, 2, 1)    # 16×32 → 32×64
        self.dec2 = nn.ModuleList([
            ResAttnBlock(ch[1] * 2, ch[1], time_emb_dim, use_attn=False),
            ResAttnBlock(ch[1], ch[0], time_emb_dim, use_attn=False),
        ])

        self.up1 = nn.ConvTranspose2d(ch[0], ch[0], 4, 2, 1)    # 32×64 → 64×128
        self.dec1 = nn.ModuleList([
            ResAttnBlock(ch[0] * 2, ch[0], time_emb_dim, use_attn=False),
            ResAttnBlock(ch[0], ch[0], time_emb_dim, use_attn=False),
        ])

        # ── Output ───────────────────────────────────────────────────
        self.out_norm = nn.GroupNorm(8, ch[0])
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(ch[0], 2, 3, 1, 1)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, 5, H, W) — [x_t(2ch), mask(1ch), known_values(2ch)]
            t: (N, 1) or (N,) time embedding indices
        Returns:
            (N, 2, H, W)  predicted ε or x̂₀
        """
        # Split: UNet sees only x_t; conditioning goes through encoder
        x_t = x[:, :2]     # (N, 2, H, W)
        cond = x[:, 2:]    # (N, 3, H, W) — [mask, known_u, known_v]

        # Conditioning features at each resolution
        c1, c2, c3, c4, c5 = self.cond_encoder(cond)

        # Time embedding
        t_emb = self.time_embed_table(t)
        if t_emb.dim() == 3:
            t_emb = t_emb.squeeze(1)
        t_emb = self.time_mlp(t_emb)      # (N, time_emb_dim)

        # ── Encoder with FiLM ────────────────────────────────────────
        h = x_t
        for block in self.enc1:
            h = block(h, t_emb)
        h = self.film_enc1(h, c1)          # FiLM at 64×128
        skip1 = h

        h = self.down1(h)
        for block in self.enc2:
            h = block(h, t_emb)
        h = self.film_enc2(h, c2)          # FiLM at 32×64
        skip2 = h

        h = self.down2(h)
        for block in self.enc3:
            h = block(h, t_emb)
        h = self.film_enc3(h, c3)          # FiLM at 16×32
        skip3 = h

        h = self.down3(h)
        for block in self.enc4:
            h = block(h, t_emb)
        h = self.film_enc4(h, c4)          # FiLM at 8×16
        skip4 = h

        h = self.down4(h)

        # ── Bottleneck with FiLM ─────────────────────────────────────
        for block in self.mid:
            h = block(h, t_emb)
        h = self.film_mid(h, c5)           # FiLM at 4×8

        # ── Decoder with FiLM ────────────────────────────────────────
        h = self.up4(h)                    # → 8×16
        h = torch.cat([skip4, h], dim=1)   # (N, 512, 8, 16)
        for block in self.dec4:
            h = block(h, t_emb)            # → (N, 256, 8, 16)
        h = self.film_dec4(h, c4)

        h = self.up3(h)                    # → 16×32
        h = torch.cat([skip3, h], dim=1)   # (N, 512, 16, 32)
        for block in self.dec3:
            h = block(h, t_emb)            # → (N, 128, 16, 32)
        h = self.film_dec3(h, c3)

        h = self.up2(h)                    # → 32×64
        h = torch.cat([skip2, h], dim=1)   # (N, 256, 32, 64)
        for block in self.dec2:
            h = block(h, t_emb)            # → (N, 64, 32, 64)
        h = self.film_dec2(h, c2)

        h = self.up1(h)                    # → 64×128
        h = torch.cat([skip1, h], dim=1)   # (N, 128, 64, 128)
        for block in self.dec1:
            h = block(h, t_emb)            # → (N, 64, 64, 128)
        h = self.film_dec1(h, c1)

        # ── Output ───────────────────────────────────────────────────
        h = self.out_act(self.out_norm(h))
        return self.out_conv(h)
