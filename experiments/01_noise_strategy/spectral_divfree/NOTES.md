# Spectral Div-Free Noise — Experiment Notes

## Architecture

| Property | Value |
|----------|-------|
| UNet | FiLM-conditioned (5ch input: x_t + mask + known_values) |
| Prediction target | x₀ |
| Noise strategy | `spectral_div_free` |
| Noise steps | 250 |
| Conditioning | FiLM (mask-aware) |
| `mask_xt` | true |
| Standardizer | `zscore_unified` (auto-resolved) |

**Note:** This experiment inherits FiLM/x₀ defaults from `base_inpaint.yaml`.
It is a **conditioned** model. Uses FFT-based Helmholtz decomposition for
div-free noise generation (as opposed to the forward-diff curl operator used
by `fwd_divfree` and `fwd_divfree_equalized`).

---

## 2026-02-17 — Config created

- Config created, validated via dry-run
- Not yet trained

## 2026-02-20 — Spectral gap note

The `spectral_div_free` noise likely shares the same spectral gap problem
identified in `forward_diff_div_free` — the curl operator (whether applied
via finite differences or FFT) is inherently a high-pass filter. The
equalization approach developed for `fwd_diff_eq_divfree` would need to be
adapted for the spectral noise pathway if this experiment is pursued.

## Status

**Not trained.** Lower priority given that:
1. The spectral gap issue affects all curl-based noise (including this one)
2. The equalized noise fix was developed for the forward-diff pathway first
3. If equalized forward-diff works, an equalized spectral variant may follow
