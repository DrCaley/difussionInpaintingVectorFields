"""FiLM-conditioned UNet for inpainting.

Instead of concatenating [mask, known_values] as extra input channels
(which the model can ignore), this architecture uses Feature-wise Linear
Modulation (FiLM) to inject conditioning at every resolution level.

A separate conditioning encoder processes [mask(1ch), known_values(2ch)]
into multi-scale features, which then modulate the UNet's activations
via learned scale (γ) and shift (β) parameters at each block.

Interface: forward(x, t) where x is (N, 5, H, W) = [x_t, mask, known],
so training and inference code don't need changes — just swap the UNet class.
"""

import torch
import torch.nn as nn

from ddpm.neural_networks.unets.unet_xl import sinusoidal_embedding, my_block


# ─── Building blocks ────────────────────────────────────────────────────────


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: h_out = γ(cond) * h + β(cond).

    Initialized to identity (γ=1, β=0) so the model starts as if
    there is no conditioning and gradually learns to use it.
    """

    def __init__(self, cond_channels, feature_channels):
        super().__init__()
        self.scale_conv = nn.Conv2d(cond_channels, feature_channels, 1)
        self.shift_conv = nn.Conv2d(cond_channels, feature_channels, 1)

        # γ = 1, β = 0 at init  → FiLM is identity
        nn.init.zeros_(self.scale_conv.weight)
        nn.init.ones_(self.scale_conv.bias)
        nn.init.zeros_(self.shift_conv.weight)
        nn.init.zeros_(self.shift_conv.bias)

    def forward(self, h, cond):
        gamma = self.scale_conv(cond)
        beta = self.shift_conv(cond)
        return gamma * h + beta


class ConditioningEncoder(nn.Module):
    """Encodes [mask(1ch), known_values(2ch)] into multi-scale features.

    Produces features matching the UNet's 5 resolution levels:
        c1: (N,  16, 64, 128)
        c2: (N,  32, 32,  64)
        c3: (N,  64, 16,  32)
        c4: (N, 128,  8,  16)
        c5: (N, 128,  2,   4)   ← bottleneck
    """

    def __init__(self):
        super().__init__()
        # Level 1: 64×128
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1), nn.SiLU(),
            nn.Conv2d(16, 16, 3, 1, 1), nn.SiLU(),
        )
        # Level 2: 32×64
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(32, 32, 3, 1, 1), nn.SiLU(),
        )
        # Level 3: 16×32
        self.enc3 = nn.Sequential(
            nn.Conv2d(32, 64, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(64, 64, 3, 1, 1), nn.SiLU(),
        )
        # Level 4: 8×16
        self.enc4 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(128, 128, 3, 1, 1), nn.SiLU(),
        )
        # Level 5 (bottleneck): 2×4
        self.enc5 = nn.Sequential(
            nn.Conv2d(128, 128, 4, 2, 1), nn.SiLU(),   # 8×16 → 4×8
            nn.Conv2d(128, 128, 4, 2, 1), nn.SiLU(),   # 4×8  → 2×4
        )

    def forward(self, cond):
        c1 = self.enc1(cond)
        c2 = self.enc2(c1)
        c3 = self.enc3(c2)
        c4 = self.enc4(c3)
        c5 = self.enc5(c4)
        return c1, c2, c3, c4, c5


# ─── Main model ─────────────────────────────────────────────────────────────


class MyUNet_FiLM(nn.Module):
    """UNet with FiLM conditioning from mask + known values.

    Same interface as MyUNet_Inpaint: forward(x, t) where x is 5 channels.
    Internally splits x into x_t (2ch) and cond (3ch), processes them
    separately, and applies FiLM modulation after every UNet block.
    """

    def __init__(self, n_steps=1000, time_emb_dim=100, in_channels=5):
        super().__init__()
        self.in_channels = in_channels   # kept for compatibility checks

        # ── Conditioning pathway ──────────────────────────────────────
        self.cond_encoder = ConditioningEncoder()

        # FiLM layers — down path
        self.film_d1 = FiLMLayer(cond_channels=16,  feature_channels=16)
        self.film_d2 = FiLMLayer(cond_channels=32,  feature_channels=32)
        self.film_d3 = FiLMLayer(cond_channels=64,  feature_channels=64)
        self.film_d4 = FiLMLayer(cond_channels=128, feature_channels=128)
        self.film_mid = FiLMLayer(cond_channels=128, feature_channels=128)

        # FiLM layers — up path (reuse encoder features at matching resolutions)
        self.film_u5 = FiLMLayer(cond_channels=128, feature_channels=64)
        self.film_u6 = FiLMLayer(cond_channels=64,  feature_channels=32)
        self.film_u7 = FiLMLayer(cond_channels=32,  feature_channels=16)
        self.film_out = FiLMLayer(cond_channels=16,  feature_channels=16)

        # ── Sinusoidal time embedding (frozen) ────────────────────────
        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)

        # ── Down path (2-channel input: just x_t) ────────────────────
        self.te1 = self._make_te(time_emb_dim, 1)
        self.b1 = nn.Sequential(
            my_block((2, 64, 128), 2, 16),
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

        # ── Bottleneck ────────────────────────────────────────────────
        self.te_mid = self._make_te(time_emb_dim, 128)
        self.b_mid = nn.Sequential(
            my_block((128, 2, 4), 128, 256),
            my_block((256, 2, 4), 256, 256),
            my_block((256, 2, 4), 256, 128),
        )

        # ── Up path ──────────────────────────────────────────────────
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
            x: (N, 5, H, W) — [x_t(2ch), mask(1ch), known_values(2ch)]
            t: (N, 1) time embedding indices
        Returns:
            (N, 2, H, W) predicted noise ε_θ
        """
        # Split input: UNet sees only x_t; conditioning goes through encoder
        x_t = x[:, :2]     # (N, 2, H, W) — noisy image
        cond = x[:, 2:]    # (N, 3, H, W) — [mask, known_u, known_v]

        # Encode conditioning at 5 resolution scales
        c1, c2, c3, c4, c5 = self.cond_encoder(cond)

        t = self.time_embed(t)
        n = len(x_t)

        # ── Down path with FiLM modulation ───────────────────────────
        out1 = self.b1(x_t + self.te1(t).reshape(n, -1, 1, 1))
        out1 = self.film_d1(out1, c1)

        out2 = self.b2(self.down1(out1) + self.te2(t).reshape(n, -1, 1, 1))
        out2 = self.film_d2(out2, c2)

        out3 = self.b3(self.down2(out2) + self.te3(t).reshape(n, -1, 1, 1))
        out3 = self.film_d3(out3, c3)

        out4 = self.b4(self.down3(out3) + self.te4(t).reshape(n, -1, 1, 1))
        out4 = self.film_d4(out4, c4)

        # ── Bottleneck with FiLM ─────────────────────────────────────
        out_mid = self.b_mid(self.down4(out4) + self.te_mid(t).reshape(n, -1, 1, 1))
        out_mid = self.film_mid(out_mid, c5)

        # ── Up path with FiLM (reuse encoder features at each scale) ─
        out5 = torch.cat((out4, self.up1(out_mid)), dim=1)
        out5 = self.b5(out5 + self.te5(t).reshape(n, -1, 1, 1))
        out5 = self.film_u5(out5, c4)

        out6 = torch.cat((out3, self.up2(out5)), dim=1)
        out6 = self.b6(out6 + self.te6(t).reshape(n, -1, 1, 1))
        out6 = self.film_u6(out6, c3)

        out7 = torch.cat((out2, self.up3(out6)), dim=1)
        out7 = self.b7(out7 + self.te7(t).reshape(n, -1, 1, 1))
        out7 = self.film_u7(out7, c2)

        out = torch.cat((out1, self.up4(out7)), dim=1)
        out = self.b_out(out + self.te_out(t).reshape(n, -1, 1, 1))
        out = self.film_out(out, c1)

        return self.conv_out(out)

    # ─── helpers ─────────────────────────────────────────────────────
    def _make_te(self, dim_in, dim_out):
        return nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            nn.Linear(dim_out, dim_out),
        )
