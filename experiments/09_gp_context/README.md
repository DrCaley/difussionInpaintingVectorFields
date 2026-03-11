# 09 — GP-Context Dense Conditioning

## Research Question

Does providing the UNet with **dense spatial context channels** — inspired
by V-CNN's 5-channel input — improve inpainting quality compared to
unconditional RePaint (S6) and FiLM conditioning?

## Background

- **V-CNN** (1.9M params) achieves strong results at 0.1–1% coverage using
  5 dense channels: voronoi_u, voronoi_v, distance_to_sensor, sensor_mask,
  ocean_mask. Every pixel has meaningful values.
- **FiLM conditioning** failed at multi-step inference (2.18× GP MSE with
  250-step reverse). The sparse conditioning signal (99% zeros at low
  coverage) compounds errors across diffusion steps.
- **GP-context** takes the best of both: dense GP-derived context (every
  pixel has values), concat-style simplicity, and standard DDPM iterative
  refinement.

## Architecture

`MyUNet_Attn(in_channels=8)` — same 23.2M-param backbone, but with 8
input channels:

| Channel | Size | Source | Notes |
|---------|------|--------|-------|
| x_t (u, v) | 2ch | Forward diffusion | Evolves each timestep |
| mask | 1ch | Fixed sensor layout | 1=missing, 0=known |
| GP mean (u, v) | 2ch | Precomputed GP posterior | Dense everywhere |
| GP variance (max) | 1ch | Precomputed GP posterior | Uncertainty map |
| Distance to sensor | 1ch | EDT from mask | Spatial proximity |
| Ocean mask | 1ch | Domain geometry | 1=ocean, 0=padding |

The GP context channels are **static** (same for all timesteps). Only
x_t evolves. This is the Palette (Saharia et al. 2022) concat approach
but with richer, physically-meaningful conditioning.

## Controlled Variables

- **Architecture backbone**: MyUNet_Attn (23.2M params)
- **Training regime**: T=250, mask_xt=true, loss on missing region only
- **Data**: rams_head, fixed sensor layout
- **Noise schedule**: linear β from 0.0001 to 0.02

## Varied Variables

| Experiment | prediction_target | noise_function | Notes |
|------------|------------------|----------------|-------|
| `gp_context_eps` | eps | gaussian | Standard DDPM noise prediction |
| (future) `gp_context_x0` | x0 | gaussian | Direct prediction variant |
| (future) `gp_context_divfree` | eps | forward_diff_div_free | Div-free noise |

## Key Comparisons

- vs S6 GP-Diff (unconditional, 2ch input): Does context help?
- vs FiLM conditioned (sparse 3ch conditioning): Does dense > sparse?
- vs V-CNN (5ch direct regression, 1.9M params): Can diffusion match/beat V-CNN?
