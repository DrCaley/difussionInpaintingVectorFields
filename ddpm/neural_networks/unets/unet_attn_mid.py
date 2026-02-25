"""Mid-size UNet with self-attention and dropout for ocean velocity inpainting.

Bridges the gap between MyUNet_Attn_Slim (6.1M, too little capacity) and
MyUNet_Attn (23.2M, overfits heavily):
  - Channel schedule: [48, 96, 192, 192]
  - Dropout in ResBlocks (default p=0.1)
  - Attention at 8×16 (level 4, 128 positions) AND bottleneck (4×8, 32 positions)

Resolution path : 64×128 → 32×64 → 16×32 → 8×16 → 4×8  (bottleneck)
Channel schedule: base 48, multipliers [1, 2, 4, 4]  →  48, 96, 192, 192
Self-attention  :   ✗       ✗        ✗       ★       ★  (level4 + bottleneck)
Dropout         : p=0.1 in every ResBlock

Parameter count : ~13.5M  (~1,470 params/sample for 9,180 training examples)

Design rationale:
  - Slim model (6.1M) peaked at test_loss=0.016 — insufficient capacity
  - Big model (23.2M) peaked at test_loss=0.0093 but overfit heavily
  - Mid-size targets the sweet spot: enough capacity for good denoising,
    with dropout + attention placement to control overfitting
  - Attention at 8×16 (128 positions) captures medium-range spatial
    correlations that convolutions alone miss, without the cost of
    attention at 16×32 (512 positions)
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
    """Residual block with GroupNorm + AdaGN time conditioning + dropout.

    Architecture per block:
        GroupNorm → SiLU → Conv → GroupNorm → SiLU → Dropout → Conv → + skip
    Time embedding is projected to 2*out_c and split into (scale, shift)
    applied after the second GroupNorm (before the second SiLU).
    """

    def __init__(self, in_c: int, out_c: int, time_emb_dim: int,
                 dropout: float = 0.1, num_groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(num_groups, in_c), in_c)
        self.conv1 = nn.Conv2d(in_c, out_c, 3, 1, 1)
        self.norm2 = nn.GroupNorm(min(num_groups, out_c), out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

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
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]

        h = self.norm2(h) * (1 + scale) + shift
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.skip(x)


class SelfAttention2d(nn.Module):
    """Multi-head self-attention over spatial dims (H*W sequence length).

    Uses pre-norm (GroupNorm) and residual connection, following
    Dhariwal & Nichol (2021).
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

        qkv = self.qkv(h)
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        q = q.permute(0, 1, 3, 2)
        k = k.permute(0, 1, 3, 2)
        v = v.permute(0, 1, 3, 2)

        out = F.scaled_dot_product_attention(q, k, v)

        out = out.permute(0, 1, 3, 2)
        out = out.reshape(B, C, H, W)
        out = self.proj(out)

        return x + out


class ResAttnBlock(nn.Module):
    """ResBlock optionally followed by self-attention."""

    def __init__(self, in_c: int, out_c: int, time_emb_dim: int,
                 use_attn: bool = False, num_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.res = ResBlock(in_c, out_c, time_emb_dim, dropout=dropout)
        self.attn = SelfAttention2d(out_c, num_heads=num_heads) if use_attn else None

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.res(x, t_emb)
        if self.attn is not None:
            h = self.attn(h)
        return h


# ── main network ─────────────────────────────────────────────────────

class MyUNet_Attn_Mid(nn.Module):
    """Mid-size UNet with attention for 64×128 ocean velocity.

    Resolution path : 64×128 → 32×64 → 16×32 → 8×16 → 4×8  (bottleneck)
    Channels (enc)  :   48       96      192     192     192
    Attention       :   ✗         ✗        ✗       ★       ★
    Dropout         : p=0.1 in every ResBlock

    Input:  (N, 2, 64, 128)  — 2-channel velocity field
    Output: (N, 2, 64, 128)
    """

    def __init__(self, n_steps: int = 1000, time_emb_dim: int = 256,
                 in_channels: int = 2, dropout: float = 0.1):
        super().__init__()
        self.in_channels = in_channels

        # Channel schedule: base 48, multipliers [1, 2, 4, 4]
        ch = [48, 96, 192, 192]

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
        # Level 1: 64×128, 48 ch  (no attention)
        self.enc1 = nn.ModuleList([
            ResAttnBlock(in_channels, ch[0], time_emb_dim, use_attn=False, dropout=dropout),
            ResAttnBlock(ch[0], ch[0], time_emb_dim, use_attn=False, dropout=dropout),
        ])
        self.down1 = nn.Conv2d(ch[0], ch[0], 4, 2, 1)       # → 32×64

        # Level 2: 32×64, 96 ch  (no attention)
        self.enc2 = nn.ModuleList([
            ResAttnBlock(ch[0], ch[1], time_emb_dim, use_attn=False, dropout=dropout),
            ResAttnBlock(ch[1], ch[1], time_emb_dim, use_attn=False, dropout=dropout),
        ])
        self.down2 = nn.Conv2d(ch[1], ch[1], 4, 2, 1)       # → 16×32

        # Level 3: 16×32, 192 ch  (no attention — conv receptive field sufficient)
        self.enc3 = nn.ModuleList([
            ResAttnBlock(ch[1], ch[2], time_emb_dim, use_attn=False, dropout=dropout),
            ResAttnBlock(ch[2], ch[2], time_emb_dim, use_attn=False, dropout=dropout),
        ])
        self.down3 = nn.Conv2d(ch[2], ch[2], 4, 2, 1)       # → 8×16

        # Level 4: 8×16, 192 ch  ★ attention (128 positions)
        self.enc4 = nn.ModuleList([
            ResAttnBlock(ch[2], ch[3], time_emb_dim, use_attn=True, num_heads=4, dropout=dropout),
            ResAttnBlock(ch[3], ch[3], time_emb_dim, use_attn=True, num_heads=4, dropout=dropout),
        ])
        self.down4 = nn.Conv2d(ch[3], ch[3], 4, 2, 1)       # → 4×8

        # ── bottleneck: 4×8, 192 ch  ★ attention (32 positions) ─────
        self.mid = nn.ModuleList([
            ResAttnBlock(ch[3], ch[3], time_emb_dim, use_attn=True, num_heads=4, dropout=dropout),
            ResAttnBlock(ch[3], ch[3], time_emb_dim, use_attn=True, num_heads=4, dropout=dropout),
        ])

        # ── decoder ──────────────────────────────────────────────────
        # Level 4 decode: 4×8 → 8×16, concat skip4 (192+192 = 384 in)
        self.up4 = nn.ConvTranspose2d(ch[3], ch[3], 4, 2, 1)
        self.dec4 = nn.ModuleList([
            ResAttnBlock(ch[3] * 2, ch[3], time_emb_dim, use_attn=True, num_heads=4, dropout=dropout),
            ResAttnBlock(ch[3], ch[2], time_emb_dim, use_attn=True, num_heads=4, dropout=dropout),
        ])

        # Level 3 decode: 8×16 → 16×32, concat skip3 (192+192 = 384 in)
        self.up3 = nn.ConvTranspose2d(ch[2], ch[2], 4, 2, 1)
        self.dec3 = nn.ModuleList([
            ResAttnBlock(ch[2] * 2, ch[2], time_emb_dim, use_attn=False, dropout=dropout),
            ResAttnBlock(ch[2], ch[1], time_emb_dim, use_attn=False, dropout=dropout),
        ])

        # Level 2 decode: 16×32 → 32×64, concat skip2 (96+96 = 192 in)
        self.up2 = nn.ConvTranspose2d(ch[1], ch[1], 4, 2, 1)
        self.dec2 = nn.ModuleList([
            ResAttnBlock(ch[1] * 2, ch[1], time_emb_dim, use_attn=False, dropout=dropout),
            ResAttnBlock(ch[1], ch[0], time_emb_dim, use_attn=False, dropout=dropout),
        ])

        # Level 1 decode: 32×64 → 64×128, concat skip1 (48+48 = 96 in)
        self.up1 = nn.ConvTranspose2d(ch[0], ch[0], 4, 2, 1)
        self.dec1 = nn.ModuleList([
            ResAttnBlock(ch[0] * 2, ch[0], time_emb_dim, use_attn=False, dropout=dropout),
            ResAttnBlock(ch[0], ch[0], time_emb_dim, use_attn=False, dropout=dropout),
        ])

        # ── output ───────────────────────────────────────────────────
        self.out_norm = nn.GroupNorm(8, ch[0])
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(ch[0], 2, 3, 1, 1)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Time embedding
        t_emb = self.time_embed_table(t)
        if t_emb.dim() == 3:
            t_emb = t_emb.squeeze(1)
        t_emb = self.time_mlp(t_emb)

        # ── Encoder ──────────────────────────────────────────────────
        h = x
        for block in self.enc1:
            h = block(h, t_emb)
        skip1 = h

        h = self.down1(h)
        for block in self.enc2:
            h = block(h, t_emb)
        skip2 = h

        h = self.down2(h)
        for block in self.enc3:
            h = block(h, t_emb)
        skip3 = h

        h = self.down3(h)
        for block in self.enc4:
            h = block(h, t_emb)
        skip4 = h

        h = self.down4(h)

        # ── Bottleneck ───────────────────────────────────────────────
        for block in self.mid:
            h = block(h, t_emb)

        # ── Decoder ──────────────────────────────────────────────────
        h = self.up4(h)
        h = torch.cat([skip4, h], dim=1)
        for block in self.dec4:
            h = block(h, t_emb)

        h = self.up3(h)
        h = torch.cat([skip3, h], dim=1)
        for block in self.dec3:
            h = block(h, t_emb)

        h = self.up2(h)
        h = torch.cat([skip2, h], dim=1)
        for block in self.dec2:
            h = block(h, t_emb)

        h = self.up1(h)
        h = torch.cat([skip1, h], dim=1)
        for block in self.dec1:
            h = block(h, t_emb)

        # ── Output ───────────────────────────────────────────────────
        h = self.out_act(self.out_norm(h))
        return self.out_conv(h)
