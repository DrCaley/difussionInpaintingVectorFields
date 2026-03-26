"""Helmholtz dual-head UNet with split high-resolution decoder (Option C).

Same as MyUNet_Helmholtz but splits the decoder after dec3 (16×32 resolution).
Each head (ψ, φ) gets its own independent dec2 + dec1 + output conv, operating
at 32×64 and 64×128 resolution respectively.

This breaks the degeneracy where both heads learn equal-and-opposite patterns
through shared high-res features. The expensive attention layers (dec4, dec3)
remain shared, adding only ~6% more parameters vs the original.

Architecture:
  Shared: encoder → bottleneck → dec4 (8×16) → dec3 (16×32)
  ψ branch: dec2_psi (32×64) → dec1_psi (64×128) → psi_conv → curl → v_sol
  φ branch: dec2_phi (32×64) → dec1_phi (64×128) → phi_conv → grad → v_irr
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ddpm.neural_networks.unets.unet_xl_attn import (
    sinusoidal_embedding,
    ResAttnBlock,
)


class MyUNet_Helmholtz_Split(nn.Module):
    """Helmholtz UNet with independent high-res decoders per head.

    Input:  (N, 2, 64, 128)
    Output: (N, 2, 64, 128)  — v = curl(ψ) + grad(φ)
    """

    def __init__(self, n_steps: int = 1000, time_emb_dim: int = 256,
                 in_channels: int = 2, n_stage_tokens: int = 0,
                 self_cond_channels: int = 0, detach_heads: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.n_stage_tokens = int(n_stage_tokens)
        self.self_cond_channels = self_cond_channels
        self.detach_heads = detach_heads

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

        # ── Shared low-res decoder (dec4 → dec3) ────────────────────
        h = self.up4(h)
        h = torch.cat([skip4, h], dim=1)
        for block in self.dec4:
            h = block(h, t_emb)

        h = self.up3(h)
        h = torch.cat([skip3, h], dim=1)
        for block in self.dec3:
            h = block(h, t_emb)
        # h is now (B, 128, 16, 32) — shared representation

        # ── ψ branch: independent high-res decoder ──────────────────
        h_psi = self.up2_psi(h)
        h_psi = torch.cat([skip2, h_psi], dim=1)
        for block in self.dec2_psi:
            h_psi = block(h_psi, t_emb)

        h_psi = self.up1_psi(h_psi)
        h_psi = torch.cat([skip1, h_psi], dim=1)
        for block in self.dec1_psi:
            h_psi = block(h_psi, t_emb)

        psi = self.psi_conv(self.psi_act(self.psi_norm(h_psi)))  # (B, 1, H, W)
        psi = psi.squeeze(1)                                       # (B, H, W)
        psi_padded = F.pad(psi, (0, 1, 0, 1), mode="constant", value=0.0)
        v_sol = self._curl_from_streamfunction(psi_padded)

        # ── φ branch: independent high-res decoder ──────────────────
        h_phi = self.up2_phi(h)
        h_phi = torch.cat([skip2, h_phi], dim=1)
        for block in self.dec2_phi:
            h_phi = block(h_phi, t_emb)

        h_phi = self.up1_phi(h_phi)
        h_phi = torch.cat([skip1, h_phi], dim=1)
        for block in self.dec1_phi:
            h_phi = block(h_phi, t_emb)

        phi = self.phi_conv(self.phi_act(self.phi_norm(h_phi)))   # (B, 1, H, W)
        v_irr = self._grad_potential(phi)

        # Store for diagnostics / penalty losses
        self.last_psi = psi
        self.last_phi = phi.squeeze(1)
        self.last_v_sol = v_sol
        self.last_v_irr = v_irr

        if self.detach_heads:
            return v_sol.detach() + v_irr.detach()
        return v_sol + v_irr
