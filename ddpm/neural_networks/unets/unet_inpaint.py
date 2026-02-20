"""Palette-style mask-aware UNet for inpainting.

Identical architecture to MyUNet but accepts 5 input channels:
    [x_t (2ch), mask (1ch), known_values (2ch)]
instead of 2.  All other layers (down/up blocks, skip connections,
time embeddings, final conv → 2ch output) are unchanged.
"""

import torch
import torch.nn as nn

from ddpm.neural_networks.unets.unet_xl import sinusoidal_embedding, my_block


class MyUNet_Inpaint(nn.Module):
    """UNet conditioned on mask + known values (Palette-style)."""

    def __init__(self, n_steps=1000, time_emb_dim=100, in_channels=5):
        super().__init__()
        self.in_channels = in_channels   # x_t(2) + mask(1) + known(2)

        # Sinusoidal time embedding (frozen)
        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)

        # ── Down path ────────────────────────────────────────────────
        self.te1 = self._make_te(time_emb_dim, 1)
        self.b1 = nn.Sequential(
            my_block((in_channels, 64, 128), in_channels, 16),   # <── 5→16
            my_block((16, 64, 128), 16, 16),
            my_block((16, 64, 128), 16, 16),
        )
        self.down1 = nn.Conv2d(16, 16, 4, 2, 1)        # 64×128 → 32×64

        self.te2 = self._make_te(time_emb_dim, 16)
        self.b2 = nn.Sequential(
            my_block((16, 32, 64), 16, 32),
            my_block((32, 32, 64), 32, 32),
            my_block((32, 32, 64), 32, 32),
        )
        self.down2 = nn.Conv2d(32, 32, 4, 2, 1)        # 32×64 → 16×32

        self.te3 = self._make_te(time_emb_dim, 32)
        self.b3 = nn.Sequential(
            my_block((32, 16, 32), 32, 64),
            my_block((64, 16, 32), 64, 64),
            my_block((64, 16, 32), 64, 64),
        )
        self.down3 = nn.Conv2d(64, 64, 4, 2, 1)        # 16×32 → 8×16

        self.te4 = self._make_te(time_emb_dim, 64)
        self.b4 = nn.Sequential(
            my_block((64, 8, 16), 64, 128),
            my_block((128, 8, 16), 128, 128),
            my_block((128, 8, 16), 128, 128),
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(128, 128, 4, 2, 1),               # 8×16 → 4×8
            nn.SiLU(),
            nn.Conv2d(128, 128, 4, 2, 1),               # 4×8  → 2×4
        )

        # ── Bottleneck ───────────────────────────────────────────────
        self.te_mid = self._make_te(time_emb_dim, 128)
        self.b_mid = nn.Sequential(
            my_block((128, 2, 4), 128, 256),
            my_block((256, 2, 4), 256, 256),
            my_block((256, 2, 4), 256, 128),
        )

        # ── Up path ─────────────────────────────────────────────────
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, 2, 1),      # 2×4  → 4×8
            nn.SiLU(),
            nn.ConvTranspose2d(128, 128, 4, 2, 1),      # 4×8  → 8×16
        )
        self.te5 = self._make_te(time_emb_dim, 256)
        self.b5 = nn.Sequential(
            my_block((256, 8, 16), 256, 128),
            my_block((128, 8, 16), 128, 64),
            my_block((64, 8, 16), 64, 64),
        )

        self.up2 = nn.ConvTranspose2d(64, 64, 4, 2, 1)  # 8×16 → 16×32
        self.te6 = self._make_te(time_emb_dim, 128)
        self.b6 = nn.Sequential(
            my_block((128, 16, 32), 128, 64),
            my_block((64, 16, 32), 64, 32),
            my_block((32, 16, 32), 32, 32),
        )

        self.up3 = nn.ConvTranspose2d(32, 32, 4, 2, 1)  # 16×32 → 32×64
        self.te7 = self._make_te(time_emb_dim, 64)
        self.b7 = nn.Sequential(
            my_block((64, 32, 64), 64, 32),
            my_block((32, 32, 64), 32, 16),
            my_block((16, 32, 64), 16, 16),
        )

        self.up4 = nn.ConvTranspose2d(16, 16, 4, 2, 1)  # 32×64 → 64×128
        self.te_out = self._make_te(time_emb_dim, 32)
        self.b_out = nn.Sequential(
            my_block((32, 64, 128), 32, 16),
            my_block((16, 64, 128), 16, 16),
            my_block((16, 64, 128), 16, 16, normalize=False),
        )

        self.conv_out = nn.Conv2d(16, 2, 3, 1, 1)       # output: 2 channels (u, v)

    # ─── forward ─────────────────────────────────────────────────────
    def forward(self, x, t):
        """
        Args:
            x: (N, 5, 64, 128) — concatenation of [x_t, mask, known_values]
            t: (N, 1)  time embedding indices
        Returns:
            (N, 2, 64, 128) predicted noise ε_θ
        """
        t = self.time_embed(t)
        n = len(x)

        out1 = self.b1(x + self.te1(t).reshape(n, -1, 1, 1))
        out2 = self.b2(self.down1(out1) + self.te2(t).reshape(n, -1, 1, 1))
        out3 = self.b3(self.down2(out2) + self.te3(t).reshape(n, -1, 1, 1))
        out4 = self.b4(self.down3(out3) + self.te4(t).reshape(n, -1, 1, 1))

        out_mid = self.b_mid(self.down4(out4) + self.te_mid(t).reshape(n, -1, 1, 1))

        out5 = torch.cat((out4, self.up1(out_mid)), dim=1)
        out5 = self.b5(out5 + self.te5(t).reshape(n, -1, 1, 1))

        out6 = torch.cat((out3, self.up2(out5)), dim=1)
        out6 = self.b6(out6 + self.te6(t).reshape(n, -1, 1, 1))

        out7 = torch.cat((out2, self.up3(out6)), dim=1)
        out7 = self.b7(out7 + self.te7(t).reshape(n, -1, 1, 1))

        out = torch.cat((out1, self.up4(out7)), dim=1)
        out = self.b_out(out + self.te_out(t).reshape(n, -1, 1, 1))

        return self.conv_out(out)

    # ─── helpers ─────────────────────────────────────────────────────
    def _make_te(self, dim_in, dim_out):
        return nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            nn.Linear(dim_out, dim_out),
        )
