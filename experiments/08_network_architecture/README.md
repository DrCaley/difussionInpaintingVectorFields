# Experiment Group 08 — Network Architecture

## Research Question

**What UNet architecture gives the best sample quality for unconditional
eps-prediction diffusion on 44×94 ocean velocity fields?**

The original `MyUNet` (from group 02) has no self-attention and limited
receptive field. `MyUNet_Attn` adds residual blocks with AdaGN time
conditioning and multi-head self-attention at the two coarsest encoder
levels (16×32 and 8×16), giving the network long-range spatial awareness.
However, at 23.2 M parameters it may overfit the ~450-sample training set.

This group investigates:
1. Whether self-attention improves generation quality
2. The optimal model capacity (channels / attention placement)
3. Whether modern training recipes (EMA, cosine LR, AdamW) matter
   independently of architecture

## Controlled Variables (held constant)

| Variable | Value |
|----------|-------|
| Prediction target | `eps` |
| Noise function | `gaussian` |
| Noise steps | 250 |
| Beta schedule | `min_beta=0.0001`, `max_beta=0.02` |
| Dataset | rams_head |
| Standardizer | `zscore_unified` (auto) |
| Inpainting method | `repaint_standard` (no CG projection) |
| `mask_xt` | `false` |

## Varied Variables

| Experiment | UNet Type | Params | Attention | Training Recipe | Notes |
|------------|-----------|--------|-----------|-----------------|-------|
| `repaint_gaussian_attn` | `standard_attn` (MyUNet_Attn) | ~23.2 M | 16×32 + 8×16 + bottleneck | bs=16, lr=3e-4, cosine, EMA(0.9999) | Main attention model |
| `repaint_gaussian_attn_mid` | `standard_attn_mid` (MyUNet_Attn_Mid) | ~13.5 M | Level 4 + bottleneck | bs=32, lr=1e-3, cosine, EMA(0.9999) | Mid-capacity sweep |
| `repaint_gaussian_attn_slim` | `standard_attn_slim` (MyUNet_Attn_Slim) | ~6.1 M | Bottleneck only (4×8) | bs=64, lr=1e-3, cosine, EMA(0.9999) | Slim capacity sweep |
| `repaint_gaussian_attn_v2` | `standard_attn` (MyUNet_Attn) | ~23.2 M | Same as _attn | bs=16, lr=5e-4, grad_accum=5, EMA(0.999) | Training recipe ablation |
| `repaint_gaussian_attn_vanilla` | `standard` (MyUNet, no attention) | ~2 M | None | bs=16, lr=1e-3, cosine, EMA(0.9999) | A/B test: original arch + modern training |

### Baseline (in group 02)

`02_inpaint_algorithm/repaint_gaussian` — same `standard` UNet with the
original training recipe (bs=80, lr=1e-3, no EMA, no cosine). Comparing
`repaint_gaussian_attn_vanilla` against this baseline isolates the effect
of the training recipe from the architecture.

## Key Findings (from existing experiments)

- **MyUNet_Attn (23.2 M)** overfits on the small training set without
  careful regularization (dropout, weight decay, EMA)
- **MyUNet_Attn_Slim (6.1 M)** may have too little capacity — attention
  at only 4×8 may not span enough spatial range
- **MyUNet_Attn_Mid (13.5 M)** was designed as a sweet-spot compromise
- The `_v2` recipe (faster EMA warmup, gradient accumulation for effective
  bs=80) was an attempt to address overfitting in the full model

## Network Architecture Files

| UNet Type | Source File |
|-----------|-------------|
| `standard` (MyUNet) | `ddpm/neural_networks/unets/basic_unet.py` |
| `standard_attn` (MyUNet_Attn) | `ddpm/neural_networks/unets/unet_xl_attn.py` |
| `standard_attn_slim` (MyUNet_Attn_Slim) | `ddpm/neural_networks/unets/unet_attn_slim.py` |
| `standard_attn_mid` (MyUNet_Attn_Mid) | `ddpm/neural_networks/unets/unet_attn_mid.py` |

## History

These experiments were originally created inside `02_inpaint_algorithm/`
but were moved here (2026-03-03) because they primarily investigate
architecture and capacity, not inpainting algorithms.
