# Spectrally-Equalized Forward-Diff Div-Free Noise — Experiment Notes

## Architecture

| Property | Value |
|----------|-------|
| UNet | `standard` — unconditional (2ch input: u, v only) |
| Prediction target | eps |
| Noise strategy | `fwd_diff_eq_divfree` |
| Noise steps | 250 |
| Conditioning | NONE — model never sees mask or known values |
| `mask_xt` | false |
| Standardizer | `zscore_unified` (auto-resolved) |
| Beta schedule | `min_beta=0.0001, max_beta=0.02` (corrected) |

**Note:** This is an **unconditional** model. Inpainting is done via RePaint
(copy-paste at each step). The model only ever sees 2-channel (u, v) inputs.

---

## 2026-02-19 — Created

**Motivation:** Analysis of `forward_diff_div_free` noise revealed a fundamental
spectral gap — the curl operator is a high-pass filter, so div-free noise has
almost zero power at low spatial frequencies (0.2% of energy below k=0.1, vs
3.1% for Gaussian). This starves the DDPM reverse process of low-frequency
content needed for large-scale ocean current structure, causing all div-free
noise models to produce systematically too-small magnitudes (~0.39× GT).

**Fix:** New noise strategy `fwd_diff_eq_divfree` colours ψ in Fourier space
with 1/√G(k) before applying the forward-diff curl, where G(k) is the
transfer function of the curl operator. This produces velocity noise that is:
- EXACTLY divergence-free (same discrete operator)
- Approximately white-spectrum (EQ/gaussian ratio 0.96–1.01 across all k)
- Unit per-pixel variance

**Verification** (`verify_equalized_noise.py`):
- |div|_max = 9.54e-07 (same floating-point level as original)
- Low-freq energy: 3.1% below k=0.1 (matches Gaussian, up from 0.2%)
- Spatial correlation ρ(u_adj) = −0.32 (less extreme than original's −0.50)

**Config:** Same unconditional eps-prediction setup as `repaint_cg`, so the
noise strategy is the ONLY variable relative to that experiment.

## 2026-02-20 — Pipeline spectral analysis

Traced all noise injection and projection sites in the codebase to verify the
equalized noise fix propagates correctly:

| Site | Uses noise_strategy? | Spectral gap? |
|------|---------------------|---------------|
| Forward process (training) | YES | Fixed by equalization |
| Reverse process z (sampling) | YES | Fixed by equalization |
| RePaint paste/resample noise | YES | Fixed by equalization |
| Projections (FFT/CG/Jacobi) | N/A — flat transfer function | No bias |

All noise injection sites in `repaint_standard()` use the same `noise_strategy`
object, so the equalized noise propagates correctly everywhere.

## 2026-02-20 — Training on Colab (T4 GPU)

- Training launched on Google Colab (T4 GPU) via `train_equalized_colab.ipynb`
- Local training log shows 3 epochs before being moved to Colab:
  - Epoch 1: test loss 0.0766
  - Epoch 2: test loss 0.0490
  - Epoch 3: test loss 0.0371
- Colab checkpoint: **epoch 124, best test loss 0.01034** (better than Gaussian's 0.01385)
- Checkpoint saved as `results/inpaint_fwd_diff_eq_divfree_t250_colab.pt`

## 2026-02-20 — Inpainting results (standard RePaint)

First test with `generate_equalized_divfree_inpaint.py`, 1 sample, resample=3:

| Method | MSE | Mag Ratio | Ang Err | |div| |
|--------|-----|-----------|---------|------|
| EQ RePaint (plain) | 0.0495 | 0.727× | 81.3° | 4.06e-02 |
| EQ RePaint + CG | 0.1567 | 1.376× | 75.4° | 1.95e-03 |
| Gauss RePaint | 0.0017 | 0.736× | 13.7° | 5.84e-03 |
| GP baseline | 0.0007 | 0.920× | 13.4° | 3.84e-03 |

**Result: Equalized div-free noise model fails at inpainting** despite having
better test loss than the Gaussian model (0.0103 vs 0.0139). The ~81° angular
error (near-random) confirms the model denoises well but RePaint breaks it.

## 2026-02-20 — Coherent RePaint experiment

**Hypothesis:** Standard RePaint generates independent noise for the known-region
forward-noising (streamfunction ψ₁) and reverse-step stochastic noise (ψ₂).
The paste creates divergence spikes at the boundary. Can we fix this?

Implemented `repaint_coherent_divfree()` with two modes:

1. **Marginal** — predict x̂₀, clip, composite with known x₀, re-noise with
   single ε. Noise is *exactly* div-free (one ψ, one coefficient everywhere).
   Trade-off: loses DDPM posterior stabilization.

2. **Shared-ψ** — keep DDPM posterior for unknown region but share the noise
   sample ε with known-region noising. Noise coefficients differ at boundary
   (σ_t vs √(1−ᾱ_{t-1})), but ψ is coherent.

Results (1 sample, resample=3):

| Method | MSE | Mag Ratio | Ang Err | |div| |
|--------|-----|-----------|---------|------|
| EQ Coherent (marginal) | 0.2083 | 2.743× | 118.0° | 8.59e-02 |
| EQ Coherent (shared-ψ) | 0.0321 | 0.722× | 75.4° | 3.24e-02 |
| EQ Standard | 0.0398 | 0.707× | 66.4° | 3.88e-02 |
| Gauss RePaint | 0.0014 | 0.838× | 13.5° | 6.25e-03 |
| GP baseline | 0.0007 | 0.920× | 13.4° | 3.84e-03 |

**Analysis:**
- **Marginal mode fails** — losing DDPM posterior is too destructive.
  x̂₀ predictions at high t are noisy; without x_t stabilization, errors
  compound. Even with clipping (clip=5), MSE blows up.
- **Shared-ψ helps modestly** — MSE ↓19% (0.040→0.032), |div| ↓17%
  (3.88e-02→3.24e-02). Confirms boundary incoherence hypothesis *partially*.
- **Angular error remains ~66-75°** for ALL eq div-free variants (near-random
  is 90°). This means the problem goes **deeper than the paste boundary.**

**Root cause (revised):** RePaint is fundamentally incompatible with
spatially-correlated noise. The paste changes the noise distribution not just
at the boundary but *everywhere in the unknown region* — a spatially-truncated
excerpt of a div-free field is not itself div-free (the PDE constraint depends
on cross-boundary neighbors). The model sees out-of-distribution noise and
cannot denoise it correctly. With Gaussian noise this doesn't matter because
pixels are independent.

**Conclusion:** Div-free noise cannot be used with RePaint. Viable paths:
1. Gaussian training + post-hoc div-free projection (repaint_cg experiment)
2. Conditioned models (FiLM/concat UNet — no paste step at all)

## Status

**Negative result confirmed.** Equalized div-free noise + RePaint fails due
to fundamental incompatibility between spatially-correlated noise and the
RePaint paste operation. The model denoises well (test loss 0.0103) but
RePaint inference produces near-random vectors.

Coherent RePaint variants (marginal, shared-ψ) provide marginal improvements
but cannot fix the fundamental distribution mismatch.
