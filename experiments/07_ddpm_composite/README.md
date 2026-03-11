# 07 — DDPM Composite

## Research Question

Can we combine a GP-conditioned CNN with a DDPM ensemble via
variance-weighted compositing to produce better ocean velocity
reconstructions than either component alone?

## Pipeline

1. **GP interpolation** — RBF kernel (ℓ=14.1) fills the sparse
   observations analytically, producing a posterior mean and variance.
2. **GP-CNN** — A 6-channel U-Net refines the GP output using
   (GP_mean_u/v, GP_std_u/v, sensor_mask, ocean_mask) as input.
3. **DDPM ensemble** — The GP-CNN prediction (standardized) is noised
   to timestep t and denoised by a FiLM+Attn conditional diffusion
   model; 5 ensemble members are averaged.
4. **Variance-weighted composite** — Near observations (low GP variance)
   the CNN dominates; far from observations (high GP variance) the DDPM
   ensemble dominates:
   `result = (1 − w) · CNN + w · DDPM_mean`,  where w = normalized GP σ.

## Components

| Component | Description | Checkpoint |
|-----------|-------------|------------|
| **GP** | Analytical RBF interpolation (ℓ=14.1) | — |
| **GP-CNN** | 6-ch U-Net (base_ch=32, depth=3, ~1.9M params) | `results/gp_cnn_diverse/gp_cnn_diverse_best.pt` |
| **DDPM** | FiLM+Attn, T=250, x₀-prediction, EMA (23.2M params) | `experiments/06_gp_forward/gp_conditioned/results/inpaint_gaussian_t250_best_ema_weights.pt` |
| **VCNN** | Voronoi-CNN baseline for comparison (base_ch=64, depth=4) | `results/voronoi_cnn/voronoi_cnn_best.pt` |

## Controlled Variables

- Validation dataset: `data.pickle` (1965 samples, 44×94 ocean grid)
- Mask type: random Gaussian (pixel-level)
- Evaluation seed: 42

## Varied Variables

- **Observation coverage**: 1%, 5%, 10%
- **DDPM timestep**: t=200 (1%), t=75 (5%, 10%)

## Experiments

### `visual_quiver/`
Single-panel quiver-plot comparisons (GT, GP, VCNN, Composite) at
configurable coverage levels. Uses the project's standard `plot_vector_field`
style (blue arrows, green land, white background).

### `eddy_detection/`
*(Planned)* Gamma-1 eddy detection comparison across coverage levels.
Results already computed in `results/eddy_compare/` — to be migrated here.
