# Gaussian RePaint + Attention UNet — Experiment Notes

## 2026-02-19 — Config created

- Same as `repaint_gaussian` but uses `standard_attn` UNet (MyUNet_Attn)
- Adds residual blocks, multi-head self-attention at 16×32 and 8×16
- Tests whether long-range spatial awareness improves RePaint quality
- Not yet trained
