"""Guided-Diffusion-style UNet with self-attention for ocean velocity fields.

Architecture modelled on Dhariwal & Nichol (2021) "Diffusion Models Beat GANs",
adapted for 64×128 two-channel ocean velocity inpainting.

Key design choices
──────────────────
  Resolution path : 64×128 → 32×64 → 16×32 → 8×16 → 4×8  (bottleneck)
  Channel schedule: base 64, multipliers [1, 2, 4, 4]  →  64, 128, 256, 256
  Self-attention   :  ✗        ✗        ★       ★           ★  (bottleneck)
  ResBlocks/level  : 2 (matching Guided Diffusion ``num_res_blocks=2``)

Comparison with reference architectures
───────────────────────────────────────
  MyUNet (unet_xl.py) :  8.8 M params, no attention,  2×4 bottleneck (8 pos)
  This network         : ~18 M params, attn × 3 levels, 4×8 bottleneck (32 pos)
  Guided Diff 256×256 : ~500 M params, attn × 3 levels, 8×8 bottleneck (64 pos)

The 4×8 bottleneck (32 spatial positions) is proportionally similar to Guided
Diffusion's 8×8 (64 positions) for 256×256 input — our image is ⅛ the area.
Self-attention at 16×32 (512 pos) and 8×16 (128 pos) provides the global
spatial communication that was missing in MyUNet, critical for coherent
inpainting under high mask coverage (70–95 %+).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── helpers ──────────────────────────────────────────────────────────

def sinusoidal_embedding(n: int, d: int) -> torch.Tensor:
    """Fixed sinusoidal position embedding (timestep → vector)."""
    emb = torch.zeros(n, d)
    wk = torch.tensor([1 / 10_000 ** (2 * j / d) for j in range(d)])
    t = torch.arange(n).unsqueeze(1)
    emb[:, 0::2] = torch.sin(t * wk[0::2])
    emb[:, 1::2] = torch.cos(t * wk[1::2])
    return emb


# ── building blocks ─────────────────────────────────────────────────

class ResBlock(nn.Module):
    """Residual block with GroupNorm + time-embedding shift/scale (AdaGN).

    Architecture per block:
        GroupNorm → SiLU → Conv → GroupNorm → SiLU → Conv → + skip
    Time embedding is projected to 2*out_c and split into (scale, shift)
    applied after the second GroupNorm (before the second SiLU).
    """

    def __init__(self, in_c: int, out_c: int, time_emb_dim: int,
                 num_groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(num_groups, in_c), in_c)
        self.conv1 = nn.Conv2d(in_c, out_c, 3, 1, 1)
        self.norm2 = nn.GroupNorm(min(num_groups, out_c), out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.act = nn.SiLU()

        # Time → scale & shift for AdaGN
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, 2 * out_c),
        )

        # 1×1 conv skip when channel counts differ
        self.skip = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h)

        # AdaGN: apply time-dependent scale & shift after norm2
        ts = self.time_mlp(t_emb)                       # (N, 2*out_c)
        scale, shift = ts.chunk(2, dim=1)                # each (N, out_c)
        scale = scale[:, :, None, None]                  # broadcast to (N, C, 1, 1)
        shift = shift[:, :, None, None]

        h = self.norm2(h) * (1 + scale) + shift
        h = self.act(h)
        h = self.conv2(h)

        return h + self.skip(x)


class SelfAttention2d(nn.Module):
    """Multi-head self-attention over spatial dims (H*W sequence length).

    Uses pre-norm (GroupNorm) and residual connection, following
    Dhariwal & Nichol (2021) - "Diffusion Models Beat GANs".
    """

    def __init__(self, channels: int, num_heads: int = 4, num_groups: int = 8):
        super().__init__()
        assert channels % num_heads == 0, (
            f"channels ({channels}) must be divisible by num_heads ({num_heads})"
        )
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm = nn.GroupNorm(min(num_groups, channels), channels)
        self.qkv = nn.Conv2d(channels, 3 * channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)

        qkv = self.qkv(h)                                          # (B, 3C, H, W)
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]                 # each (B, heads, head_dim, HW)
        q = q.permute(0, 1, 3, 2)                                  # (B, heads, HW, head_dim)
        k = k.permute(0, 1, 3, 2)                                  # (B, heads, HW, head_dim)
        v = v.permute(0, 1, 3, 2)                                  # (B, heads, HW, head_dim)

        # Scaled dot-product attention (uses Flash Attention when available)
        out = F.scaled_dot_product_attention(q, k, v)               # (B, heads, HW, head_dim)

        out = out.permute(0, 1, 3, 2)                               # (B, heads, head_dim, HW)
        out = out.reshape(B, C, H, W)
        out = self.proj(out)

        return x + out                                               # residual


class ResAttnBlock(nn.Module):
    """ResBlock optionally followed by self-attention."""

    def __init__(self, in_c: int, out_c: int, time_emb_dim: int,
                 use_attn: bool = False, num_heads: int = 4):
        super().__init__()
        self.res = ResBlock(in_c, out_c, time_emb_dim)
        self.attn = SelfAttention2d(out_c, num_heads=num_heads) if use_attn else None

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.res(x, t_emb)
        if self.attn is not None:
            h = self.attn(h)
        return h


# ── main network ─────────────────────────────────────────────────────

class MyUNet_Attn(nn.Module):
    """Guided-Diffusion-style UNet for 64×128 ocean velocity inpainting.

    Resolution path : 64×128 → 32×64 → 16×32 → 8×16 → 4×8  (bottleneck)
    Channels (enc)  :   64       128      256     256     256
    Attention       :   ✗         ✗        ★       ★       ★

    Input:  (N, 2, 64, 128)  — 2-channel velocity field
    Output: (N, 2, 64, 128)
    """

    def __init__(self, n_steps: int = 1000, time_emb_dim: int = 256,
                 in_channels: int = 2):
        super().__init__()
        self.in_channels = in_channels

        # Channel schedule: base 64, multipliers [1, 2, 4, 4]
        ch = [64, 128, 256, 256]

        # ── time embedding ───────────────────────────────────────────
        self.time_embed_table = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed_table.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed_table.requires_grad_(False)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        # ── encoder ──────────────────────────────────────────────────
        # Level 1: 64×128, 64 ch  (no attention — 8 192 positions too large)
        self.enc1 = nn.ModuleList([
            ResAttnBlock(in_channels, ch[0], time_emb_dim, use_attn=False),
            ResAttnBlock(ch[0], ch[0], time_emb_dim, use_attn=False),
        ])
        self.down1 = nn.Conv2d(ch[0], ch[0], 4, 2, 1)       # → 32×64

        # Level 2: 32×64, 128 ch  (no attention — 2 048 positions still large)
        self.enc2 = nn.ModuleList([
            ResAttnBlock(ch[0], ch[1], time_emb_dim, use_attn=False),
            ResAttnBlock(ch[1], ch[1], time_emb_dim, use_attn=False),
        ])
        self.down2 = nn.Conv2d(ch[1], ch[1], 4, 2, 1)       # → 16×32

        # Level 3: 16×32, 256 ch  ★ attention (512 positions)
        self.enc3 = nn.ModuleList([
            ResAttnBlock(ch[1], ch[2], time_emb_dim, use_attn=True, num_heads=4),
            ResAttnBlock(ch[2], ch[2], time_emb_dim, use_attn=True, num_heads=4),
        ])
        self.down3 = nn.Conv2d(ch[2], ch[2], 4, 2, 1)       # → 8×16

        # Level 4: 8×16, 256 ch  ★ attention (128 positions)
        self.enc4 = nn.ModuleList([
            ResAttnBlock(ch[2], ch[3], time_emb_dim, use_attn=True, num_heads=4),
            ResAttnBlock(ch[3], ch[3], time_emb_dim, use_attn=True, num_heads=4),
        ])
        self.down4 = nn.Conv2d(ch[3], ch[3], 4, 2, 1)       # → 4×8  (single clean downsample)

        # ── bottleneck: 4×8, 256 ch  ★ attention (32 positions) ─────
        self.mid = nn.ModuleList([
            ResAttnBlock(ch[3], ch[3], time_emb_dim, use_attn=True, num_heads=4),
            ResAttnBlock(ch[3], ch[3], time_emb_dim, use_attn=True, num_heads=4),
        ])

        # ── decoder ──────────────────────────────────────────────────
        # Level 4 decode: 4×8 → 8×16, concat skip4 (256+256 = 512 in)
        self.up4 = nn.ConvTranspose2d(ch[3], ch[3], 4, 2, 1)
        self.dec4 = nn.ModuleList([
            ResAttnBlock(ch[3] * 2, ch[3], time_emb_dim, use_attn=True, num_heads=4),
            ResAttnBlock(ch[3], ch[2], time_emb_dim, use_attn=True, num_heads=4),
        ])

        # Level 3 decode: 8×16 → 16×32, concat skip3 (256+256 = 512 in)
        self.up3 = nn.ConvTranspose2d(ch[2], ch[2], 4, 2, 1)
        self.dec3 = nn.ModuleList([
            ResAttnBlock(ch[2] * 2, ch[2], time_emb_dim, use_attn=True, num_heads=4),
            ResAttnBlock(ch[2], ch[1], time_emb_dim, use_attn=True, num_heads=4),
        ])

        # Level 2 decode: 16×32 → 32×64, concat skip2 (128+128 = 256 in)
        self.up2 = nn.ConvTranspose2d(ch[1], ch[1], 4, 2, 1)
        self.dec2 = nn.ModuleList([
            ResAttnBlock(ch[1] * 2, ch[1], time_emb_dim, use_attn=False),
            ResAttnBlock(ch[1], ch[0], time_emb_dim, use_attn=False),
        ])

        # Level 1 decode: 32×64 → 64×128, concat skip1 (64+64 = 128 in)
        self.up1 = nn.ConvTranspose2d(ch[0], ch[0], 4, 2, 1)
        self.dec1 = nn.ModuleList([
            ResAttnBlock(ch[0] * 2, ch[0], time_emb_dim, use_attn=False),
            ResAttnBlock(ch[0], ch[0], time_emb_dim, use_attn=False),
        ])

        # ── output ───────────────────────────────────────────────────
        self.out_norm = nn.GroupNorm(8, ch[0])
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(ch[0], 2, 3, 1, 1)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Time embedding
        t_emb = self.time_embed_table(t)
        if t_emb.dim() == 3:              # (N,1,D) from some callers
            t_emb = t_emb.squeeze(1)
        t_emb = self.time_mlp(t_emb)      # (N, time_emb_dim)

        # ── Encoder ──────────────────────────────────────────────────
        h = x
        for block in self.enc1:
            h = block(h, t_emb)
        skip1 = h                          # (N, 64, 64, 128)

        h = self.down1(h)
        for block in self.enc2:
            h = block(h, t_emb)
        skip2 = h                          # (N, 128, 32, 64)

        h = self.down2(h)
        for block in self.enc3:
            h = block(h, t_emb)
        skip3 = h                          # (N, 256, 16, 32)

        h = self.down3(h)
        for block in self.enc4:
            h = block(h, t_emb)
        skip4 = h                          # (N, 256, 8, 16)

        h = self.down4(h)                  # (N, 256, 4, 8)

        # ── Bottleneck ───────────────────────────────────────────────
        for block in self.mid:
            h = block(h, t_emb)            # (N, 256, 4, 8)

        # ── Decoder ──────────────────────────────────────────────────
        h = self.up4(h)                    # (N, 256, 8, 16)
        h = torch.cat([skip4, h], dim=1)   # (N, 512, 8, 16)
        for block in self.dec4:
            h = block(h, t_emb)            # → (N, 256, 8, 16)

        h = self.up3(h)                    # (N, 256, 16, 32)
        h = torch.cat([skip3, h], dim=1)   # (N, 512, 16, 32)
        for block in self.dec3:
            h = block(h, t_emb)            # → (N, 128, 16, 32)

        h = self.up2(h)                    # (N, 128, 32, 64)
        h = torch.cat([skip2, h], dim=1)   # (N, 256, 32, 64)
        for block in self.dec2:
            h = block(h, t_emb)            # → (N, 64, 32, 64)

        h = self.up1(h)                    # (N, 64, 64, 128)
        h = torch.cat([skip1, h], dim=1)   # (N, 128, 64, 128)
        for block in self.dec1:
            h = block(h, t_emb)            # → (N, 64, 64, 128)

        # ── Output ───────────────────────────────────────────────────
        h = self.out_act(self.out_norm(h))
        return self.out_conv(h)
