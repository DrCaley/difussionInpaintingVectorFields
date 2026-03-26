# multires_standard_diffusion

## What's being tested
Same multires FiLM architecture as `helmholtz_film_multires`, but with
**standard diffusion** (noise GT) instead of Voronoi-forward diffusion
(noise Voronoi fill).

## Motivation
Voronoi-forward creates a train/inference distribution mismatch during
multi-step reverse diffusion: the model trains on noised Voronoi fill but
at inference time sees noised predictions of clean fields. Standard
diffusion avoids this because noised GT matches the distribution seen
during the reverse chain.

## Changes from helmholtz_film_multires
- `voronoi_forward: false` (was `true`)
- Everything else identical

## Log

### 2025-03-19: Training converged, fundamental failure at inpainting

**Training**: Converged at epoch ~147/800 (EMA test loss 0.0135, plateaued).

**Evaluation** (scripts/eval_standard_diffusion.py on Server 2, 10 samples at 95% coverage):

| Method | MSE | Ratio vs V-CNN |
|--------|-----|----------------|
| V-CNN | 0.000147 | 1.0x |
| Oracle@t=50 | 0.00585 | 40x worse |
| Oracle@t=249 | 0.00724 | 49x worse |
| 1S-Noise@t=25 | 0.01435 | 97x worse |
| Rev+RP@50 | 0.04478 | 304x worse |

**Root cause**: `mask_xt: false` means the model learns unconditional whole-image
denoising. Two problems:
1. x_t contains noised GT everywhere — no incentive to use FiLM conditioning
2. Loss is computed over entire image (via loss strategy), not just missing region

Even the oracle test (GT-noised x_t, exact training input distribution) is 40x
worse than V-CNN, proving the model never learned to use FiLM conditioning for
reconstruction.

**Fix**: New experiment `multires_maskxt` with `mask_xt: true`.
