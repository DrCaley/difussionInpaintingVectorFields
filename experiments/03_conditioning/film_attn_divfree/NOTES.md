# FiLM + Attention UNet — Div-Free Noise

## 2026-02-23 — Experiment Created

**Goal**: Test whether FiLM conditioning with the large attention UNet
improves inpainting quality for divergence-free noise, compared to
unconditional RePaint approaches explored in `02_inpaint_algorithm/`.

**Architecture**: `MyUNet_FiLM_Attn`
- Attention UNet backbone: channels [64, 128, 256, 256], self-attention
  at 16×32, 8×16, and 4×8 (bottleneck)
- FiLM conditioning encoder: mirrors resolution path, produces γ/β
  modulation at every level
- Zero-initialized FiLM (γ=1, β=0 at init) — starts as unconditioned

**Key differences from prior work**:
- Model receives 5-channel input [x_t, mask, known_values]
- Uses x0 prediction (not eps) with div-free noise
- mask_xt=true: known pixels replaced with noised ground truth in x_t
- Training uses cosine LR schedule, EMA, AdamW (same as attn experiments)

**Observations**:
- (pending smoke test and training)
