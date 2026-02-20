# fwd_divfree_equalized — Experiment Notes

## 2026-02-19: Created

**Motivation**: Analysis of forward_diff_div_free noise revealed a fundamental
spectral gap — the curl operator is a high-pass filter, so div-free noise has
almost zero power at low spatial frequencies (0.2% of energy below k=0.1, vs
3.1% for Gaussian). This starves the DDPM reverse process of low-frequency
content needed for large-scale ocean current structure, causing all div-free
noise models to produce systematically too-small magnitudes (~0.39× GT).

**Fix**: New noise strategy `fwd_diff_eq_divfree` colours ψ in Fourier space
with 1/√G(k) before applying the forward-diff curl, where G(k) is the
transfer function of the curl operator. This produces velocity noise that is:
- EXACTLY divergence-free (same discrete operator)
- Approximately white-spectrum (EQ/gaussian ratio 0.96–1.01 across all k)
- Unit per-pixel variance

**Verification** (verify_equalized_noise.py):
- |div|_max = 9.54e-07 (same floating-point level as original)
- Low-freq energy: 3.1% below k=0.1 (matches Gaussian, up from 0.2%)
- Spatial correlation ρ(u_adj) = -0.32 (less extreme than original's -0.50)

**Config**: Unconditional UNet (standard), eps-prediction, 250 steps.
Same setup as repaint_cg experiment but with equalized noise, so the
noise strategy is the ONLY difference.

**Status**: Config created. Awaiting dry-run + training.
