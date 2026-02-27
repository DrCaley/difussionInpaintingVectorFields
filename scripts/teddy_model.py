"""
Teddy-style network for direct eddy prediction from sparse velocity observations.

Adapted from:
  Bolmer et al. 2024, "Estimating daily semantic segmentation maps of classified
  ocean eddies using sea level anomaly data from along-track altimetry"
  (Frontiers in AI, doi: 10.3389/frai.2024.1298283)

Architecture:
  1) Transformer encoder — processes 1D row of velocity observations with
     sinusoidal positional encoding (analogous to Teddy's AT encoder).
  2) Positional decoding — places encoded features at the observation row
     (row 22) in a 2D (D, H, W) feature grid.
  3) Sparsity-invariant CNN — propagates features from the sparse observation
     row to the full domain using normalized convolutions (Uhrig et al. 2017).
  4) Prediction head — 1×1 conv producing binary eddy probability map.

Input:  (B, W_obs, 2)  — velocity (u,v) at each observed column
Output: (B, 1, H, W)   — per-pixel eddy logits on the ocean domain
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal PE with a learnable-free scale factor (q=0.1 as in Teddy)."""

    def __init__(self, d_model: int, max_len: int = 200, scale: float = 0.1):
        super().__init__()
        self.scale = scale
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D)"""
        return x + self.scale * self.pe[:, : x.size(1)]


class SparsityInvariantConv2d(nn.Module):
    """Normalized convolution following Uhrig et al. 2017.

    For each output pixel the convolution sum is rescaled by
    ``k² / valid_count`` so that missing inputs don't bias the result.
    The validity mask is propagated: a pixel becomes valid when *any*
    neighbour in the kernel was valid.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 7, padding: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=True)
        self.register_buffer(
            "ones_kernel", torch.ones(1, 1, kernel_size, kernel_size)
        )
        self.k2 = kernel_size * kernel_size
        self.pad = padding

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """
        x    : (B, C, H, W)  features (0 where invalid)
        mask : (B, 1, H, W)  binary validity (1 = valid)
        Returns (out, new_mask).
        """
        x = x * mask                                         # zero invalid
        out = self.conv(x)

        with torch.no_grad():
            valid = F.conv2d(mask, self.ones_kernel, padding=self.pad)
            scale = self.k2 / (valid + 1e-8)
            scale = scale.clamp(max=float(self.k2))
            new_mask = (valid > 0).float()

        return out * scale, new_mask


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class TeddyNet(nn.Module):
    """Sparse-observation → eddy segmentation network."""

    def __init__(
        self,
        obs_dim: int = 2,       # features per observation (u, v)
        d_model: int = 64,      # Transformer width
        n_heads: int = 4,
        n_enc_layers: int = 3,
        n_cnn_layers: int = 8,
        ocean_h: int = 44,
        ocean_w: int = 94,
        obs_row: int = 22,
    ):
        super().__init__()
        self.ocean_h = ocean_h
        self.ocean_w = ocean_w
        self.obs_row = obs_row
        self.d_model = d_model

        # ---- Transformer encoder ----
        self.input_proj = nn.Linear(obs_dim, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=ocean_w)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_enc_layers)

        # ---- Sparsity-invariant CNN decoder ----
        # Use 7×7 kernels for first layers (expand mask by 3 px/layer),
        # switch to 5×5 for refinement.
        cnn_channels = {
            8: [d_model, d_model, d_model, 32, 32, 32, 16, 16],
        }
        channels = cnn_channels.get(n_cnn_layers)
        if channels is None:
            channels = [d_model] * (n_cnn_layers // 2) + [32] * (n_cnn_layers - n_cnn_layers // 2)

        self.sparse_convs = nn.ModuleList()
        in_ch = d_model
        for i, out_ch in enumerate(channels):
            k, p = (7, 3) if i < 6 else (5, 2)
            self.sparse_convs.append(SparsityInvariantConv2d(in_ch, out_ch, k, p))
            in_ch = out_ch

        # prediction head
        self.head = nn.Conv2d(channels[-1], 1, 1)

    # ------------------------------------------------------------------ #
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs : (B, W, 2) — velocity at each observed column of row 22.
        Returns logits (B, 1, H, W).
        """
        B = obs.shape[0]
        device = obs.device

        # --- Transformer ---
        x = self.input_proj(obs)   # (B, W, D)
        x = self.pos_enc(x)
        x = self.transformer(x)   # (B, W, D)

        # --- positional decoding: place features at obs row ---
        grid = torch.zeros(B, self.d_model, self.ocean_h, self.ocean_w, device=device)
        grid[:, :, self.obs_row, :] = x.permute(0, 2, 1)   # (B, D, W)

        mask = torch.zeros(B, 1, self.ocean_h, self.ocean_w, device=device)
        mask[:, :, self.obs_row, :] = 1.0

        # --- sparsity-invariant CNN ---
        for conv in self.sparse_convs:
            grid, mask = conv(grid, mask)
            grid = torch.tanh(grid)
            grid = grid * mask            # keep invalid pixels at 0

        return self.head(grid)             # (B, 1, H, W)

    @torch.no_grad()
    def predict_proba(self, obs: torch.Tensor) -> torch.Tensor:
        """Return sigmoid probabilities."""
        return torch.sigmoid(self.forward(obs))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
