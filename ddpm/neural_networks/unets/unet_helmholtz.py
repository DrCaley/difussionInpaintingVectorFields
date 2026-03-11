"""Helmholtz-decomposed dual-head UNet for ocean velocity inpainting.

Builds on MyUNet_Attn (Guided-Diffusion-style with self-attention) but
replaces the single 2-channel output with two physics-informed heads:

  ψ head (streamfunction)  →  curl(ψ)   = solenoidal velocity   (exactly div-free)
  φ head (velocity potential) →  grad(φ) = irrotational velocity  (exactly curl-free)

  v_total = curl(ψ) + grad(φ)

The forward-diff curl operator from ForwardDiffDivFreeNoise guarantees that
the solenoidal component has EXACTLY zero discrete divergence.  The gradient
operator guarantees the irrotational component has EXACTLY zero discrete curl.

This Helmholtz-Hodge inductive bias lets the network internally separate
rotational (geostrophic eddies, jets) from divergent (ageostrophic, tidal)
dynamics, each represented in its natural mathematical form.

Diagnostic: psi and phi are stored as attributes after each forward pass
so training code can access them for monitoring or penalty losses.
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
)


class MyUNet_Helmholtz(nn.Module):
    """Helmholtz-decomposed dual-head UNet.

    Shares encoder + bottleneck + decoder with MyUNet_Attn.
    Two lightweight output heads produce ψ (1ch) and φ (1ch),
    converted to velocity via physics operators.

    Input:  (N, 2, 64, 128)      — 2-channel velocity field (noisy)
    Output: (N, 2, 64, 128)      — v = curl(ψ) + grad(φ)

    After forward(), self.last_psi and self.last_phi hold the raw
    scalar potentials for diagnostics or penalty losses.
    """

    def __init__(self, n_steps: int = 1000, time_emb_dim: int = 256,
                 in_channels: int = 2, n_stage_tokens: int = 0,
                 self_cond_channels: int = 0):
        super().__init__()
        self.in_channels = in_channels
        self.n_stage_tokens = int(n_stage_tokens)
        self.self_cond_channels = self_cond_channels

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
        if self.n_stage_tokens > 0:
            self.stage_embed = nn.Embedding(self.n_stage_tokens, time_emb_dim)
            nn.init.normal_(self.stage_embed.weight, mean=0.0, std=0.02)
        else:
            self.stage_embed = None

        # ── self-conditioning projection ─────────────────────────────
        if self.self_cond_channels > 0:
            self.self_cond_proj = nn.Sequential(
                nn.Conv2d(self_cond_channels, ch[0], 3, 1, 1),
                nn.SiLU(),
                nn.Conv2d(ch[0], ch[0], 3, 1, 1),
            )
        else:
            self.self_cond_proj = None

        # ── encoder ──────────────────────────────────────────────────
        self.enc1 = nn.ModuleList([
            ResAttnBlock(in_channels, ch[0], time_emb_dim, use_attn=False),
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

        # ── shared decoder ───────────────────────────────────────────
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

        self.up2 = nn.ConvTranspose2d(ch[1], ch[1], 4, 2, 1)
        self.dec2 = nn.ModuleList([
            ResAttnBlock(ch[1] * 2, ch[1], time_emb_dim, use_attn=False),
            ResAttnBlock(ch[1], ch[0], time_emb_dim, use_attn=False),
        ])

        self.up1 = nn.ConvTranspose2d(ch[0], ch[0], 4, 2, 1)
        self.dec1 = nn.ModuleList([
            ResAttnBlock(ch[0] * 2, ch[0], time_emb_dim, use_attn=False),
            ResAttnBlock(ch[0], ch[0], time_emb_dim, use_attn=False),
        ])

        # ── ψ head (streamfunction → solenoidal velocity) ───────────
        self.psi_norm = nn.GroupNorm(8, ch[0])
        self.psi_act = nn.SiLU()
        self.psi_conv = nn.Conv2d(ch[0], 1, 3, 1, 1)

        # ── φ head (velocity potential → irrotational velocity) ──────
        self.phi_norm = nn.GroupNorm(8, ch[0])
        self.phi_act = nn.SiLU()
        self.phi_conv = nn.Conv2d(ch[0], 1, 3, 1, 1)

        # Initialize φ head to near-zero so model starts solenoidal-biased
        nn.init.zeros_(self.phi_conv.weight)
        nn.init.zeros_(self.phi_conv.bias)

        # ── gradient kernels for φ → (u_irr, v_irr) ─────────────────
        # Central-difference: ∂φ/∂x via (1,1,1,3) kernel [-0.5, 0, 0.5]
        # Note: x-axis = columns (dim=3), y-axis = rows (dim=2)
        dx = torch.tensor([[[[0.0, -0.5, 0.5]]]]) # ∂/∂x (column direction)
        dy = torch.tensor([[[[0.0], [-0.5], [0.5]]]])  # ∂/∂y (row direction)
        self.register_buffer("_dx_kernel", dx)
        self.register_buffer("_dy_kernel", dy)

        # Diagnostics: stored after each forward pass
        self.last_psi = None
        self.last_phi = None

    # ── physics operators ────────────────────────────────────────────

    @staticmethod
    def _curl_from_streamfunction(psi: torch.Tensor) -> torch.Tensor:
        """Forward-diff curl: ψ (B, H+1, W+1) → velocity (B, 2, H, W).

        Exactly zero discrete divergence by construction:
            u[i,j] = ψ[i+1,j] - ψ[i,j]
            v[i,j] = -(ψ[i,j+1] - ψ[i,j])
        """
        u = psi[:, 1:, :-1] - psi[:, :-1, :-1]
        v = -(psi[:, :-1, 1:] - psi[:, :-1, :-1])
        return torch.stack([u, v], dim=1)

    def _grad_potential(self, phi: torch.Tensor) -> torch.Tensor:
        """Central-diff gradient: φ (B, 1, H, W) → velocity (B, 2, H, W).

        u_irr = ∂φ/∂x (column direction), v_irr = ∂φ/∂y (row direction)
        Exactly zero discrete curl by construction.
        """
        u_irr = F.conv2d(phi, self._dx_kernel, padding=(0, 1))
        v_irr = F.conv2d(phi, self._dy_kernel, padding=(1, 0))
        return torch.cat([u_irr, v_irr], dim=1)

    # ── forward ──────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        stage: torch.Tensor | None = None,
        self_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Time embedding
        t_emb = self.time_embed_table(t)
        if t_emb.dim() == 3:
            t_emb = t_emb.squeeze(1)
        t_emb = self.time_mlp(t_emb)
        if self.stage_embed is not None:
            if stage is None:
                stage = torch.zeros(x.shape[0], device=x.device, dtype=torch.long)
            elif stage.dim() > 1:
                stage = stage.reshape(stage.shape[0])
            stage = stage.long().clamp(min=0, max=self.n_stage_tokens - 1)
            t_emb = t_emb + self.stage_embed(stage)

        # ── Encoder ──────────────────────────────────────────────────
        h = x
        for block in self.enc1:
            h = block(h, t_emb)
        if self.self_cond_proj is not None and self_cond is not None:
            h = h + self.self_cond_proj(self_cond)
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

        # ── Shared Decoder ───────────────────────────────────────────
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

        # ── ψ head: streamfunction → solenoidal velocity ─────────
        psi = self.psi_conv(self.psi_act(self.psi_norm(h)))  # (B, 1, H, W)
        psi = psi.squeeze(1)                                   # (B, H, W)
        # Zero-pad to (B, H+1, W+1) for forward-diff curl
        # This imposes ψ=0 Dirichlet BC at the bottom/right boundary
        psi_padded = F.pad(psi, (0, 1, 0, 1), mode="constant", value=0.0)
        v_sol = self._curl_from_streamfunction(psi_padded)    # (B, 2, H, W)

        # ── φ head: velocity potential → irrotational velocity ────
        phi = self.phi_conv(self.phi_act(self.phi_norm(h)))   # (B, 1, H, W)
        v_irr = self._grad_potential(phi)                      # (B, 2, H, W)

        # Store for diagnostics / penalty losses
        self.last_psi = psi
        self.last_phi = phi.squeeze(1)
        self.last_v_sol = v_sol
        self.last_v_irr = v_irr

        return v_sol + v_irr
