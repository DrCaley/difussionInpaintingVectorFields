# Forward-Diff Div-Free Noise — Experiment Notes

## Architecture

| Property | Value |
|----------|-------|
| UNet | FiLM-conditioned (5ch input: x_t + mask + known_values) |
| Prediction target | x₀ |
| Noise strategy | `forward_diff_div_free` |
| Noise steps | 250 |
| Conditioning | FiLM (mask-aware) |
| `mask_xt` | true |
| Standardizer | `zscore_unified` (auto-resolved) |

**Note:** This experiment inherits FiLM/x₀ defaults from `base_inpaint.yaml`.
It is a **conditioned** model — the UNet sees the mask and known values during
training, unlike the unconditional RePaint experiments in `02_inpaint_algorithm/`.

---

## 2026-02-17 — Initial training (bad beta schedule)

- FiLM-conditioned UNet (5ch input, 9.8M params), x₀ prediction
- `forward_diff_div_free` noise, 250 steps
- **Trained with aggressive beta schedule:** `min_beta=0.0004, max_beta=0.08`
- ᾱ₂₄₉ = 0.000033 — signal completely destroyed by t≈166
- Best test loss: 0.0441
- Used for early inference testing with `x0_full_reverse_inpaint()`

## 2026-02-19 — Beta schedule issue identified

The aggressive beta schedule was confirmed as root cause of magnitude blow-up
in the `repaint_cg` experiment (same noise strategy, different architecture):

| Schedule | min_beta | max_beta | ᾱ₂₄₉ |
|----------|----------|----------|-------|
| Old (broken) | 0.0004 | 0.08 | 0.000033 |
| Corrected | 0.0001 | 0.02 | 0.0797 |

The old schedule destroyed all signal by t≈166, leaving the last 84/250 steps
as noise→noise denoising. The near-zero ᾱ_T amplified prediction errors in
the eps→x₀ conversion.

**Needs retraining** with corrected schedule (`min_beta=0.0001, max_beta=0.02`).

## 2026-02-20 — Spectral gap in div-free noise identified

Analysis revealed that `forward_diff_div_free` noise has a fundamental spectral
gap: the curl operator is a high-pass filter, so div-free noise has almost zero
power at low spatial frequencies (0.2% energy below k=0.1, vs 3.1% for Gaussian).
This starves the DDPM reverse process of low-frequency content needed for
large-scale ocean current structure.

This is one of the motivations for the `fwd_divfree_equalized` experiment,
which corrects the spectral gap while preserving the divergence-free property.

## Status

**Not retrained.** Awaiting retraining with corrected beta schedule.
The spectral gap issue is an additional concern — equalized noise
(`fwd_divfree_equalized`) may be the better path forward.
