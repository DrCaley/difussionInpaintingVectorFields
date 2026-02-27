"""
Multi-row Teddy-style network for direct eddy prediction from sparse observations.

Same architecture as teddy_model.py (Bolmer et al. 2024 adaptation — Transformer
encoder + Sparsity-invariant CNN), but extended to accept observations from
**multiple rows** with 2D positional encoding.

This lets us test Teddy under conditions closer to its intended regime
(multi-track satellite altimetry) while keeping the method unchanged.

Architecture:
  1) Transformer encoder — processes ALL observations as a flat sequence
     with learnable 2D positional encoding (row + col).
  2) Positional decoding — places encoded features at their (row, col)
     coordinates in a 2D (D, H, W) feature grid.
  3) Sparsity-invariant CNN — propagates features from observed positions
     to the full domain using normalized convolutions.
  4) Prediction head — 1×1 conv producing binary eddy probability map.

Input:   (B, N_obs, 2)  — velocity (u,v) at each observed pixel
         (B, N_obs, 2)  — (row, col) positions of each observation
Output:  (B, 1, H, W)   — per-pixel eddy logits on the ocean domain
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnedPositionalEncoding2D(nn.Module):
    """Sinusoidal 2D positional encoding: row and column each get half the dims."""

    def __init__(self, d_model: int, max_h: int = 64, max_w: int = 128, scale: float = 0.1):
        super().__init__()
        self.scale = scale
        half = d_model // 2
        # Row encoding (half dims)
        pe_row = torch.zeros(max_h, half)
        pos_r = torch.arange(0, max_h, dtype=torch.float).unsqueeze(1)
        div_r = torch.exp(torch.arange(0, half, 2).float() * (-math.log(10000.0) / half))
        pe_row[:, 0::2] = torch.sin(pos_r * div_r)
        pe_row[:, 1::2] = torch.cos(pos_r * div_r[:half // 2] if half % 2 else div_r)
        self.register_buffer("pe_row", pe_row)  # (max_h, half)

        # Col encoding (remaining dims)
        other_half = d_model - half
        pe_col = torch.zeros(max_w, other_half)
        pos_c = torch.arange(0, max_w, dtype=torch.float).unsqueeze(1)
        div_c = torch.exp(torch.arange(0, other_half, 2).float() * (-math.log(10000.0) / other_half))
        pe_col[:, 0::2] = torch.sin(pos_c * div_c)
        pe_col[:, 1::2] = torch.cos(pos_c * div_c[:other_half // 2] if other_half % 2 else div_c)
        self.register_buffer("pe_col", pe_col)  # (max_w, other_half)

    def forward(self, x: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
        """
        x    : (B, N, D)
        rows : (B, N) int — row indices
        cols : (B, N) int — col indices
        """
        r_enc = self.pe_row[rows.long()]     # (B, N, half)
        c_enc = self.pe_col[cols.long()]     # (B, N, other_half)
        pe = torch.cat([r_enc, c_enc], dim=-1)  # (B, N, D)
        return x + self.scale * pe


class SparsityInvariantConv2d(nn.Module):
    """Normalized convolution (Uhrig et al. 2017) — identical to single-row version."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 7, padding: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=True)
        self.register_buffer("ones_kernel", torch.ones(1, 1, kernel_size, kernel_size))
        self.k2 = kernel_size * kernel_size
        self.pad = padding

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        x = x * mask
        out = self.conv(x)
        with torch.no_grad():
            valid = F.conv2d(mask, self.ones_kernel, padding=self.pad)
            scale = self.k2 / (valid + 1e-8)
            scale = scale.clamp(max=float(self.k2))
            new_mask = (valid > 0).float()
        return out * scale, new_mask


class TeddyNetMultiRow(nn.Module):
    """Sparse-observation → eddy segmentation network (multi-row variant)."""

    def __init__(
        self,
        obs_dim: int = 2,
        d_model: int = 64,
        n_heads: int = 4,
        n_enc_layers: int = 3,
        n_cnn_layers: int = 8,
        ocean_h: int = 44,
        ocean_w: int = 94,
    ):
        super().__init__()
        self.ocean_h = ocean_h
        self.ocean_w = ocean_w
        self.d_model = d_model

        # ---- Transformer encoder ----
        self.input_proj = nn.Linear(obs_dim, d_model)
        self.pos_enc = LearnedPositionalEncoding2D(d_model, max_h=ocean_h, max_w=ocean_w)
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

        self.head = nn.Conv2d(channels[-1], 1, 1)

    def forward(self, obs: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
        """
        obs  : (B, N_obs, 2) — velocity at each observed pixel
        rows : (B, N_obs)    — row index for each observation
        cols : (B, N_obs)    — col index for each observation
        Returns logits (B, 1, H, W).
        """
        B = obs.shape[0]
        N_obs = obs.shape[1]
        device = obs.device

        # --- Transformer ---
        x = self.input_proj(obs)        # (B, N_obs, D)
        x = self.pos_enc(x, rows, cols) # add 2D position
        x = self.transformer(x)         # (B, N_obs, D)

        # --- positional decoding: place features at observation positions ---
        grid = torch.zeros(B, self.d_model, self.ocean_h, self.ocean_w, device=device)
        mask = torch.zeros(B, 1, self.ocean_h, self.ocean_w, device=device)

        # Scatter encoded features into the grid
        rows_long = rows.long()  # (B, N_obs)
        cols_long = cols.long()  # (B, N_obs)
        for b in range(B):
            r = rows_long[b]  # (N_obs,)
            c = cols_long[b]  # (N_obs,)
            grid[b, :, r, c] = x[b].T   # (D, N_obs) → place at (r, c)
            mask[b, 0, r, c] = 1.0

        # --- sparsity-invariant CNN ---
        for conv in self.sparse_convs:
            grid, mask = conv(grid, mask)
            grid = torch.tanh(grid)
            grid = grid * mask

        return self.head(grid)

    @torch.no_grad()
    def predict_proba(self, obs, rows, cols):
        return torch.sigmoid(self.forward(obs, rows, cols))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
