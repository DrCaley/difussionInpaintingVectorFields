"""Spatiotemporal UNet: MyUNet_Attn + temporal mixing for T-frame sequences.

Architecture
────────────
Extends the spatial-only ``MyUNet_Attn`` with interleaved temporal
processing layers that learn cross-frame correlations.  Each frame is
processed spatially (identically to ``MyUNet_Attn``), then temporal
mixing layers share information across the T consecutive frames.

    ┌─────────────────── Per-frame spatial path ───────────────────┐
    │                                                              │
    │  Input (B*T, 2, 64, 128)                                    │
    │    │                                                         │
    │    ├─ Enc L1  (64ch,  no attn)  ─→  TemporalConv            │
    │    ├─ Enc L2  (128ch, no attn)  ─→  TemporalConv            │
    │    ├─ Enc L3  (256ch, ★ attn)   ─→  TemporalConv + TempAttn │
    │    ├─ Enc L4  (256ch, ★ attn)   ─→  TemporalConv + TempAttn │
    │    ├─ Bottleneck (256ch, ★ attn)─→  TemporalConv + TempAttn │
    │    ├─ Dec L4  (256ch, ★ attn)   ─→  TemporalConv + TempAttn │
    │    ├─ Dec L3  (128ch, ★ attn)   ─→  TemporalConv + TempAttn │
    │    ├─ Dec L2  (64ch,  no attn)  ─→  TemporalConv            │
    │    └─ Dec L1  (64ch,  no attn)  ─→  TemporalConv            │
    │                                                              │
    │  Output (B*T, 2, 64, 128)                                   │
    └──────────────────────────────────────────────────────────────┘

Temporal layers are **zero-initialized**, so at init the model behaves
identically to running ``MyUNet_Attn`` independently on each frame.
This enables direct fine-tuning from a pretrained spatial checkpoint
with zero initial performance degradation.

Input/output format
───────────────────
    Input:  (B, T*2, H, W) — T velocity frames stacked channel-wise
    Output: (B, T*2, H, W)
    Channel layout: [u₀, v₀, u₁, v₁, …, u_{T-1}, v_{T-1}]

Internally reshaped to (B*T, 2, H, W) for spatial processing.
Compatible with ``GaussianDDPM`` using ``image_chw = (T*2, 64, 128)``.

Parameter budget (T=13)
───────────────────────
    Spatial (inherited):  ~23.2 M  (identical to MyUNet_Attn)
    Temporal (new):       ~2.4 M   (~10% overhead)
    Total:                ~25.6 M

Weight loading
──────────────
Spatial layers share identical names with ``MyUNet_Attn``, enabling
direct weight loading from an existing checkpoint::

    model = MyUNet_ST.from_pretrained_spatial(
        "experiments/.../inpaint_gaussian_t250_best_checkpoint.pt",
        T=13, n_steps=250,
    )

Training strategy
─────────────────
Two-phase training is recommended:
  1. ``model.freeze_spatial()`` — train temporal layers only (fast)
  2. ``model.unfreeze_spatial()`` — fine-tune everything (lower LR)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ddpm.neural_networks.unets.unet_xl_attn import (
    sinusoidal_embedding,
    ResBlock,
    SelfAttention2d,
    ResAttnBlock,
    MyUNet_Attn,
)


# ── temporal building blocks ─────────────────────────────────────────


class TemporalConvBlock(nn.Module):
    """Temporal convolution via Conv1d along the T dimension.

    Applies a 1D convolution along the time dimension independently
    at each spatial position.  Uses pre-norm (GroupNorm), SiLU
    activation, and residual connection.

    Implementation uses Conv1d on reshaped (B*H*W, C, T) tensors
    rather than Conv3d, which has poor MPS (Apple Silicon) support.
    The computation is mathematically identical to Conv3d with
    kernel (k, 1, 1).

    **Zero-initialized** so that the output starts as identity
    (no temporal mixing), preserving pretrained spatial behavior.
    """

    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.conv = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # Zero-init → identity residual at start
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor, B: int, T: int) -> torch.Tensor:
        """
        Args:
            x: (B*T, C, H, W) — batched frames.
            B: Original batch size.
            T: Number of temporal frames.

        Returns:
            (B*T, C, H, W) with temporal mixing applied via residual.
        """
        BT, C, H, W = x.shape
        h = self.norm(x)
        # (B*T, C, H, W) → (B*H*W, C, T) for Conv1d along temporal dim
        h = (h.reshape(B, T, C, H, W)
              .permute(0, 3, 4, 2, 1)                    # (B, H, W, C, T)
              .reshape(B * H * W, C, T))
        h = self.dropout(F.silu(self.conv(h)))             # (B*H*W, C, T)
        h = (h.reshape(B, H, W, C, T)
              .permute(0, 4, 3, 1, 2)                     # (B, T, C, H, W)
              .reshape(BT, C, H, W))
        return x + h


class TemporalAttnBlock(nn.Module):
    """Multi-head self-attention across the temporal dimension.

    For each spatial position (h, w), performs self-attention across
    the T frames with learned positional embedding.  Sequence length
    is T (e.g. 13), so this is very efficient even at moderate
    spatial resolutions.

    Uses pre-norm (GroupNorm), residual connection, and
    **zero-initialized output projection** for identity at start.
    """

    def __init__(self, channels: int, T: int = 13, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        assert channels % num_heads == 0, (
            f"channels ({channels}) must be divisible by num_heads ({num_heads})"
        )
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.channels = channels

        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.qkv = nn.Linear(channels, 3 * channels)
        self.proj = nn.Linear(channels, channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Learned temporal position embedding (zero-init)
        self.pos_emb = nn.Parameter(torch.zeros(1, T, channels))

        # Zero-init output projection → identity residual at start
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, B: int, T: int) -> torch.Tensor:
        """
        Args:
            x: (B*T, C, H, W) — batched frames.
            B: Original batch size.
            T: Number of temporal frames.

        Returns:
            (B*T, C, H, W) with temporal attention applied via residual.
        """
        BT, C, H, W = x.shape
        h = self.norm(x)

        # Reshape to (B*H*W, T, C) for temporal attention
        h = (h.reshape(B, T, C, H, W)
              .permute(0, 3, 4, 1, 2)           # (B, H, W, T, C)
              .reshape(B * H * W, T, C))

        # Add temporal positional embedding
        h = h + self.pos_emb[:, :T, :]

        # QKV projection
        qkv = self.qkv(h)                       # (B*H*W, T, 3*C)
        qkv = qkv.reshape(B * H * W, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)        # (3, B*H*W, heads, T, d_k)
        q, k, v = qkv.unbind(0)

        # Scaled dot-product attention (uses Flash Attention when available)
        out = F.scaled_dot_product_attention(q, k, v)  # (B*H*W, heads, T, d_k)
        out = (out.permute(0, 2, 1, 3)                  # (B*H*W, T, heads, d_k)
                  .reshape(B * H * W, T, C))

        # Output projection (zero-initialized)
        out = self.dropout(self.proj(out))                # (B*H*W, T, C)

        # Reshape back to (B*T, C, H, W)
        out = (out.reshape(B, H, W, T, C)
                  .permute(0, 3, 4, 1, 2)                # (B, T, C, H, W)
                  .reshape(BT, C, H, W))

        return x + out


class TemporalMixBlock(nn.Module):
    """Combined temporal processing: temporal Conv3d + optional temporal attention.

    Always applies a temporal convolution; optionally follows it with
    temporal self-attention.  Both sub-layers are zero-initialized for
    identity behavior at initialization.
    """

    def __init__(
        self,
        channels: int,
        T: int = 13,
        use_attn: bool = False,
        num_heads: int = 4,
        conv_kernel: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.conv = TemporalConvBlock(channels, conv_kernel, dropout=dropout)
        self.attn = (
            TemporalAttnBlock(channels, T, num_heads, dropout=dropout)
            if use_attn else None
        )

    def forward(self, x: torch.Tensor, B: int, T: int) -> torch.Tensor:
        """x: (B*T, C, H, W) → (B*T, C, H, W)"""
        x = self.conv(x, B, T)
        if self.attn is not None:
            x = self.attn(x, B, T)
        return x


# ── main network ─────────────────────────────────────────────────────


class MyUNet_ST(MyUNet_Attn):
    """Spatiotemporal UNet for T-frame sequence denoising.

    Inherits all spatial blocks from ``MyUNet_Attn`` and adds
    interleaved temporal mixing layers.  The forward pass is
    overridden to handle the (B, T*C, H, W) input format and
    interleave temporal processing with spatial blocks.

    Temporal mixing layers are prefixed with ``temp_`` in the
    parameter namespace for easy identification and selective
    freezing/unfreezing.

    Parameters
    ----------
    n_steps : int
        Number of diffusion timesteps.
    T : int
        Number of consecutive temporal frames (default 13 = one
        M2 tidal cycle at hourly resolution).
    time_emb_dim : int
        Dimension of the sinusoidal time embedding.
    in_channels : int
        Number of input channels per frame (default 2 for u, v).
    """

    def __init__(
        self,
        n_steps: int = 1000,
        T: int = 13,
        time_emb_dim: int = 256,
        in_channels: int = 2,
        temporal_dropout: float = 0.0,
    ):
        # Initialize all spatial blocks from MyUNet_Attn
        super().__init__(
            n_steps=n_steps,
            time_emb_dim=time_emb_dim,
            in_channels=in_channels,
        )
        self.T = T
        self.temporal_dropout = temporal_dropout

        ch = [64, 128, 256, 256]

        # ── Temporal mixing layers ───────────────────────────────────
        # Temporal attention mirrors spatial attention placement:
        # levels 3, 4, and bottleneck (all ≤ 16×32 spatial resolution).
        self.temp_enc1 = TemporalMixBlock(ch[0], T, use_attn=False, dropout=temporal_dropout)
        self.temp_enc2 = TemporalMixBlock(ch[1], T, use_attn=False, dropout=temporal_dropout)
        self.temp_enc3 = TemporalMixBlock(ch[2], T, use_attn=True, dropout=temporal_dropout)
        self.temp_enc4 = TemporalMixBlock(ch[3], T, use_attn=True, dropout=temporal_dropout)

        self.temp_mid = TemporalMixBlock(ch[3], T, use_attn=True, dropout=temporal_dropout)

        self.temp_dec4 = TemporalMixBlock(ch[2], T, use_attn=True, dropout=temporal_dropout)   # dec4 out: 256ch
        self.temp_dec3 = TemporalMixBlock(ch[1], T, use_attn=True, dropout=temporal_dropout)   # dec3 out: 128ch
        self.temp_dec2 = TemporalMixBlock(ch[0], T, use_attn=False, dropout=temporal_dropout)  # dec2 out:  64ch
        self.temp_dec1 = TemporalMixBlock(ch[0], T, use_attn=False, dropout=temporal_dropout)  # dec1 out:  64ch

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T*C, H, W) — T velocity frames stacked channel-wise.
                Channel layout: [u₀,v₀, u₁,v₁, …, u_{T-1},v_{T-1}]
            t: (B,) — diffusion timestep (shared for all T frames).

        Returns:
            (B, T*C, H, W) — predicted noise ε or clean data x₀.
        """
        B = x.shape[0]
        T = self.T
        C = self.in_channels
        H, W = x.shape[2], x.shape[3]

        # ── Reshape: (B, T*C, H, W) → (B*T, C, H, W) ───────────────
        h = x.view(B, T, C, H, W).reshape(B * T, C, H, W)

        # ── Time embedding: (B, D) → replicate T times → (B*T, D) ───
        t_emb = self.time_embed_table(t)
        if t_emb.dim() == 3:
            t_emb = t_emb.squeeze(1)
        t_emb = self.time_mlp(t_emb)                                # (B, D)
        t_emb = t_emb.unsqueeze(1).expand(-1, T, -1)
        t_emb = t_emb.reshape(B * T, -1)                            # (B*T, D)

        # ── Encoder ──────────────────────────────────────────────────
        for block in self.enc1:
            h = block(h, t_emb)
        h = self.temp_enc1(h, B, T)
        skip1 = h                                                    # (B*T, 64, 64, 128)

        h = self.down1(h)
        for block in self.enc2:
            h = block(h, t_emb)
        h = self.temp_enc2(h, B, T)
        skip2 = h                                                    # (B*T, 128, 32, 64)

        h = self.down2(h)
        for block in self.enc3:
            h = block(h, t_emb)
        h = self.temp_enc3(h, B, T)
        skip3 = h                                                    # (B*T, 256, 16, 32)

        h = self.down3(h)
        for block in self.enc4:
            h = block(h, t_emb)
        h = self.temp_enc4(h, B, T)
        skip4 = h                                                    # (B*T, 256, 8, 16)

        h = self.down4(h)                                            # (B*T, 256, 4, 8)

        # ── Bottleneck ───────────────────────────────────────────────
        for block in self.mid:
            h = block(h, t_emb)
        h = self.temp_mid(h, B, T)                                   # (B*T, 256, 4, 8)

        # ── Decoder ──────────────────────────────────────────────────
        h = self.up4(h)                                              # (B*T, 256, 8, 16)
        h = torch.cat([skip4, h], dim=1)                             # (B*T, 512, 8, 16)
        for block in self.dec4:
            h = block(h, t_emb)
        h = self.temp_dec4(h, B, T)                                  # (B*T, 256, 8, 16)

        h = self.up3(h)                                              # (B*T, 256, 16, 32)
        h = torch.cat([skip3, h], dim=1)                             # (B*T, 512, 16, 32)
        for block in self.dec3:
            h = block(h, t_emb)
        h = self.temp_dec3(h, B, T)                                  # (B*T, 128, 16, 32)

        h = self.up2(h)                                              # (B*T, 128, 32, 64)
        h = torch.cat([skip2, h], dim=1)                             # (B*T, 256, 32, 64)
        for block in self.dec2:
            h = block(h, t_emb)
        h = self.temp_dec2(h, B, T)                                  # (B*T, 64, 32, 64)

        h = self.up1(h)                                              # (B*T, 64, 64, 128)
        h = torch.cat([skip1, h], dim=1)                             # (B*T, 128, 64, 128)
        for block in self.dec1:
            h = block(h, t_emb)
        h = self.temp_dec1(h, B, T)                                  # (B*T, 64, 64, 128)

        # ── Output ───────────────────────────────────────────────────
        h = self.out_act(self.out_norm(h))
        h = self.out_conv(h)                                         # (B*T, 2, H, W)

        # ── Reshape back: (B*T, C, H, W) → (B, T*C, H, W) ──────────
        h = h.reshape(B, T, C, H, W).reshape(B, T * C, H, W)
        return h

    # ------------------------------------------------------------------
    # Weight loading & freezing utilities
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained_spatial(
        cls,
        checkpoint_path: str,
        T: int = 13,
        n_steps: int = 250,
        **kwargs,
    ) -> "MyUNet_ST":
        """Create a ``MyUNet_ST`` and load spatial weights from a
        ``MyUNet_Attn`` / ``GaussianDDPM`` checkpoint.

        Temporal layers (prefixed ``temp_``) remain at their
        zero-initialization, so the model initially behaves identically
        to T independent copies of the pretrained spatial model.

        Args:
            checkpoint_path: Path to a ``.pt`` checkpoint file.  Accepts
                both full ``GaussianDDPM`` checkpoints (keys prefixed
                with ``network.``) and raw UNet state dicts.
            T: Number of temporal frames.
            n_steps: Diffusion timesteps (must match checkpoint).
            **kwargs: Forwarded to ``MyUNet_ST.__init__``.

        Returns:
            ``MyUNet_ST`` with pretrained spatial weights loaded.
        """
        model = cls(n_steps=n_steps, T=T, temporal_dropout=kwargs.pop('temporal_dropout', 0.0), **kwargs)

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Handle both full DDPM checkpoints and raw state dicts
        if "model_state_dict" in ckpt:
            sd = ckpt["model_state_dict"]
        else:
            sd = ckpt

        # Map checkpoint keys → our spatial block names
        spatial_sd = {}
        for k, v in sd.items():
            # Strip 'network.' prefix from GaussianDDPM wrapper
            clean_key = k[len("network."):] if k.startswith("network.") else k
            # Only load keys that exist in our model (spatial layers)
            if clean_key in model.state_dict():
                spatial_sd[clean_key] = v

        missing, unexpected = model.load_state_dict(spatial_sd, strict=False)

        # Verify that only temporal layers are missing
        n_temporal = sum(1 for key in missing if "temp_" in key)
        n_other = len(missing) - n_temporal

        total = len(model.state_dict())
        print(f"[MyUNet_ST] Loaded {len(spatial_sd)}/{total} params from checkpoint")
        print(f"[MyUNet_ST] {len(missing)} missing keys "
              f"({n_temporal} temporal, {n_other} other)")
        if n_other > 0:
            non_temp = [k for k in missing if "temp_" not in k]
            print(f"[MyUNet_ST] WARNING — non-temporal missing keys: {non_temp}")
        if unexpected:
            print(f"[MyUNet_ST] WARNING — unexpected keys: {unexpected}")

        return model

    def freeze_spatial(self) -> None:
        """Freeze all spatial (inherited) parameters.

        Only temporal layers (``temp_*``) remain trainable.
        Useful for phase-1 training: learning temporal correlations
        while preserving pretrained spatial representations.
        """
        for name, param in self.named_parameters():
            if "temp_" not in name:
                param.requires_grad = False

        n_frozen = sum(1 for n, p in self.named_parameters() if not p.requires_grad)
        n_trainable = sum(1 for n, p in self.named_parameters() if p.requires_grad)
        print(f"[MyUNet_ST] Froze {n_frozen} spatial params, "
              f"{n_trainable} temporal params remain trainable")

    def unfreeze_spatial(self) -> None:
        """Unfreeze all parameters for end-to-end fine-tuning.

        Recommended for phase-2 training with a lower learning rate
        after temporal layers have been warmed up.
        """
        for param in self.parameters():
            param.requires_grad = True
        n_total = sum(1 for _ in self.parameters())
        print(f"[MyUNet_ST] Unfroze all {n_total} parameters")

    def param_groups(self, spatial_lr: float, temporal_lr: float) -> list:
        """Return optimizer param groups with differential learning rates.

        Parameters
        ----------
        spatial_lr : float
            Learning rate for pretrained spatial parameters.
        temporal_lr : float
            Learning rate for temporal (``temp_*``) parameters.

        Returns
        -------
        list[dict]
            Two-element list suitable for ``torch.optim.Adam(param_groups)``.
        """
        spatial_params = [p for n, p in self.named_parameters() if "temp_" not in n]
        temporal_params = [p for n, p in self.named_parameters() if "temp_" in n]
        return [
            {"params": spatial_params, "lr": spatial_lr, "name": "spatial"},
            {"params": temporal_params, "lr": temporal_lr, "name": "temporal"},
        ]

    @property
    def num_spatial_params(self) -> int:
        """Number of spatial (inherited) parameters."""
        return sum(
            p.numel() for n, p in self.named_parameters() if "temp_" not in n
        )

    @property
    def num_temporal_params(self) -> int:
        """Number of temporal (new) parameters."""
        return sum(
            p.numel() for n, p in self.named_parameters() if "temp_" in n
        )

    @property
    def num_total_params(self) -> int:
        """Total parameter count."""
        return sum(p.numel() for p in self.parameters())
