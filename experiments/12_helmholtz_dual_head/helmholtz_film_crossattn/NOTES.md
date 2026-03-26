# helmholtz_film_crossattn — Experiment Notes

## What's Being Tested

Cross-attention (Neural Process-style) sparse encoder for FiLM conditioning,
replacing the CNN-based encoder (Voronoi fill) and FPN-based encoder (pooled sparse).

**Hypothesis:** Grid-based encoders (CNN on Voronoi, FPN on pooled sparse) lose
sub-pixel position information and introduce artifacts. A cross-attention encoder
that processes observations as unordered point sets should:
1. Preserve exact continuous positions (no grid quantization)
2. Give every query pixel global access to all observations
3. Handle variable observation counts natively

## Architecture

- Per-point MLP: [r/H, c/W, u, v] → 128-dim token
- 5 cross-attention levels with learned grid queries (one per UNet resolution)
- Multi-head cross-attention (4 heads) at each level
- Output projected to match FiLM interface: c1(64ch), c2(128ch), c3(256ch), c4(256ch), c5(256ch)
- Same UNet backbone + FiLM modulation as spatial FiLM variant

## Controlled Variables (same as other conditioning experiments)

- Helmholtz split-decoder backbone
- helmholtz_matched noise
- Unified standardizer
- helmholtz_supervised loss (λ_decomp=0.1, λ_orth=0.01)
- voronoi_forward training (noise Voronoi, predict x0)
- lr=0.0005, cosine schedule, warmup=10, EMA
- batch_size=16, 800 epochs

## Log

### 2026-03-15 — Launched on Server 1

- Replaced spatial FiLM (plateaued at 1.61x V-CNN @ ep 336)
- Training on RTX 5060 Ti (Server 1)
- Conditioning: sparse observations → cross-attention point encoder
- mask_xt: false (model does NOT see known values in x_t)
