# 03 — Conditioning Method

## Research Question

Does providing the known observations directly to the denoiser via
conditioning (FiLM, concat) improve inpainting quality compared to
unconditional models that rely solely on RePaint-style replacement?

## Controlled Variables

- **Noise function**: forward_diff_div_free (divergence-free noise)
- **Standardizer**: zscore_unified (auto-resolved)
- **Diffusion steps**: 250
- **Dataset**: rams_head
- **Prediction target**: x0

## Varied Variables

| Experiment             | Conditioning | UNet backbone | Key difference |
|------------------------|-------------|---------------|----------------|
| `film_attn_divfree`    | FiLM        | Attn UNet     | FiLM modulation + self-attention |

## Context

Previous experiments (01_noise_strategy, 02_inpaint_algorithm) used
unconditional UNets with RePaint-style inpainting. The DDPM sees only x_t
(2 channels) and relies on known-pixel replacement at each reverse step.

Conditioned models instead receive [x_t, mask, known_values] (5 channels)
so the denoiser can directly reason about known observations. FiLM
conditioning processes the mask+known signal through a separate encoder
and modulates the main UNet via learned scale/shift at every resolution.
