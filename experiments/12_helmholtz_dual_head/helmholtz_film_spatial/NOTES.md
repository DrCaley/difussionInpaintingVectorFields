# helmholtz_film_spatial — Experiment Notes

## What changed
Replaced AdaGN-style channel-wise FiLM (global avg pool → Linear) with
spatial per-pixel FiLM (1×1 Conv2d) in `MyUNet_Helmholtz_Split_FiLM`.

The conditioning encoder already produces multi-scale spatial features; the
old FiLM layer destroyed spatial structure via `AdaptiveAvgPool2d(1)`.
New layer preserves full per-pixel modulation.

## 2026-03-14 — Created
- Identical config to `helmholtz_film_nomaskxt` (mask_xt=false, same loss weights)
- Only difference: architectural fix in FiLMLayer class
