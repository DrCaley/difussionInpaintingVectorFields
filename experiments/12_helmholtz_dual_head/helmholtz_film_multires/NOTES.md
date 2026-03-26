# NOTES — Helmholtz FiLM Multi-Resolution Conditioning

## Purpose
Test whether replacing Voronoi fill conditioning with a multi-resolution
Feature Pyramid Network (FPN) encoder improves FiLM conditioning.

## Hypothesis
The spatial FiLM model (helmholtz_film_spatial, 2.08x V-CNN) faithfully uses
conditioning but gets bad information from Voronoi fill. The multi-res encoder
processes raw sparse observations at multiple scales without Voronoi artifacts:
- Pools sparse obs to each resolution level
- Normalizes by observation density → honest local means
- Top-down FPN propagates coarse-scale context (good coverage) to fine levels

## Controlled Variables (same as helmholtz_film_spatial)
- Backbone: Helmholtz split UNet (ψ/φ dual-head)
- FiLM layers: Spatial 1×1 conv, (1+γ)·GN(h)+β
- Training: voronoi_forward, matched noise, x0 prediction
- Hyperparams: batch_size=16, lr=0.0005, cosine, ema, 800 epochs
- Helmholtz loss: λ_decomp=0.1, λ_orth=0.01

## Varied Variable
- Conditioning encoder: MultiResCondEncoder (FPN, sparse input) vs
  HelmholtzCondEncoder (CNN, Voronoi fill input)
- Conditioning signal: raw sparse observations vs Voronoi fill

## Key Architecture: MultiResCondEncoder
- Input: [missing_mask(1ch), sparse_u(1ch), sparse_v(1ch)]
- At each resolution, pools sparse obs and computes:
  - normalized_obs = pooled_obs / observation_density (local mean)
  - density channel (tells model where it has information)
- Top-down processing: 4×8 → 8×16 → 16×32 → 32×64 → 64×128
- Each level: [pooled_features(3ch), upsampled_coarser(Cch)] → conv → output

## Results

### Training launched: 2025-03-15 on Server 2
- Converged at epoch 458, best_test_loss = 0.0557 (ocean-masked MSE)
- 28.71M params

### Inference evaluation: 2025-07-14

**Critical fix required in eval script**: `build_masks()` had three bugs:
1. Land/padding treated as "known" (mask=0) — flooded FPN encoder with ~4000 fake observations at standardized-zero (0.443)
2. 2-channel miss_mask from land_mask multiplication — training uses 1-ch
3. `known_std` at land/padding = 0.443 instead of 0

Also: `helmholtz_split_noise: false` (not in config) means standard DDPM schedule, but eval was trying to use HelmholtzSplitSchedule. Fixed.

**Before fix**: 38x worse than V-CNN
**After fix**: Competitive with V-CNN

#### Results (10 samples per coverage, single-step t=25)

| Coverage | V-CNN MSE | 1S@t=25 MSE | vs V-CNN | Div RMS (DDPM) | Div RMS (V-CNN) |
|----------|-----------|-------------|----------|----------------|-----------------|
| 0.5%     | 0.000688  | 0.000686    | 0.998x   | 0.00588        | 0.00652         |
| 1.0%     | 0.000450  | 0.000437    | 0.971x   | 0.00599        | 0.00662         |
| 2.0%     | 0.000249  | 0.000278    | 1.116x   | 0.00619        | 0.00672         |

- At 0.5-1% coverage (within training range): matches or slightly beats V-CNN
- At 2% coverage: 12% worse than V-CNN
- Divergence consistently lower than V-CNN across all coverages
- Ensemble (10 members) provides negligible improvement over single-step

#### Head diagnostics (0.5% coverage)
- Cancel ratio: 1.3x (healthy — not catastrophic)
- cos(ψ-head, GT_sol): 0.937
- cos(φ-head, GT_irr): 0.813
- GT solenoidal fraction: 81.4%, predicted: 79.7%

#### RevChain inference: still poor
- 4.8x at 0.5%, 7.3x at 1%, 11.1x at 2%
- Known-region repaste likely conflicts with sparse conditioning

### 2025-03-18: Alternative diffusion inference strategies

Tested 14 strategies beyond baseline single-step. Same 10 samples per coverage.
Unlike the unconditional model, multires receives conditioning at every step.

| Method | 0.5% vs V-CNN | 1.0% vs V-CNN | 2.0% vs V-CNN |
|--------|--------------|--------------|--------------|
| **1S@t=25** (baseline) | **0.998x** | **0.971x** | 1.116x |
| **1S@t=50** | 1.001x | **0.971x** | 1.116x |
| **Ens5-divT** | **0.995x** | **0.969x** | 1.116x |
| Warm2 (50→25) | 1.142x | 1.225x | 1.542x |
| Warm3 (75→50→25) | 1.324x | 1.519x | 1.996x |
| NoRP@10 | 1.627x | 1.902x | 2.717x |
| NoRP@25 | 1.984x | 2.435x | 3.741x |
| SoftRC@10 | 1.618x | 1.878x | 2.629x |
| SoftRC@25 | 1.964x | 2.384x | 3.559x |
| RePaint@25-r3 | 3.683x | 5.320x | 8.437x |
| RePaint@50-r3 | 4.367x | 6.630x | 10.068x |
| DDIM@50-5s | 1.213x | 1.296x | 1.630x |
| DDIM@50-10s | 1.334x | 1.462x | 1.923x |
| DDIM@100-10s | 1.366x | 1.506x | 2.001x |

**Observations:**
1. **Single-step remains best.** Even with conditioning, iterative methods degrade.
2. **Conditioning helps a lot** vs unconditional: NoRP@10 is 1.6x (vs 100x for
   unconditional), DDIM@50-5s is 1.2x (vs 54x). Not catastrophic, but still worse.
3. **Ens5-divT** is marginally best at 0.5%/1.0% (0.995x/0.969x) — ensemble of
   diverse noise levels slightly smooths predictions. Gain is tiny (0.3%).
4. **RePaint is worst** among iterative methods (3.7-10x). Resampling amplifies
   mismatch between sparse conditioning and noised-then-repasted known region.
5. **Warm-restart still hurts** (1.1-2.0x). Same compounding error from
   re-standardize round-trip seen in unconditional model.
6. **Conclusion: this model is a denoising autoencoder**, not a proper diffusion
   sampler. Both the conditioned and unconditioned models only work well as
   single-step "denoise the Voronoi fill" predictors.

### 2025-03-18: Traditional literature diffusion methods

Tested standard diffusion inpainting from the literature at proper noise levels
(t=75-249), full T=249 reverse chains, RePaint with r=5-10 resampling.
5 samples per coverage. Script: `scripts/eval_traditional_diffusion.py`.

Methods tested:
- **Rev+RP@t**: Full reverse chain from t with repaste at each step (Song/Ho)
- **Rev-NR@t**: Full reverse chain from t, no repaste (Palette-style)
- **RePaint@t-rN**: RePaint (Lugmayr 2022) with N resampling steps per timestep
- **PureNoise+RP**: Start from pure noise (t=249), full reverse + repaste
- **PureNoise-NR**: Start from pure noise, no repaste
- **DDIM@t-Ns**: DDIM deterministic reverse with N steps from t

| Method | 0.5% vs V-CNN | 1.0% vs V-CNN | 2.0% vs V-CNN |
|--------|--------------|--------------|--------------|
| **1S@t=25** (baseline) | 1.051x | **0.893x** | 1.066x |
| Rev+RP@75 | 2.671x | 2.599x | 5.132x |
| Rev+RP@150 | 3.034x | 3.105x | 6.423x |
| Rev+RP@249 | 3.399x | 3.703x | 7.526x |
| Rev-NR@75 | 2.703x | 2.651x | 5.565x |
| Rev-NR@150 | 3.074x | 3.151x | 7.058x |
| RePaint@75-r5 | 6.621x | 8.373x | 14.911x |
| RePaint@100-r5 | 6.894x | 8.993x | 15.702x |
| RePaint@150-r5 | 7.235x | 8.769x | 16.722x |
| RePaint@75-r10 | 8.108x | 10.721x | 18.672x |
| PureNoise+RP | 7.406x | 7.440x | 15.710x |
| PureNoise-NR | 7.421x | 10.840x | 18.529x |
| DDIM@150-25s | 1.717x | 1.481x | 2.385x |
| DDIM@249-50s | 2.004x | 1.922x | 3.153x |

**Observations:**
1. **All traditional methods fail** but conditioning keeps them 100x better than
   the unconditional model (2-19x vs 178-887x).
2. **DDIM is best among traditional** (1.5-3.2x) — deterministic reverse
   accumulates less error than stochastic methods.
3. **Longer chains are worse**: Rev+RP@249 > Rev+RP@150 > Rev+RP@75. More steps
   = more error accumulation. Opposite of what proper diffusion models show.
4. **RePaint resampling amplifies error** (r10 worse than r5), same as low-t tests.
5. **Pure noise start fails** — the model was never trained to denoise from pure
   noise, only from noised Voronoi fill. Starting from noise gives 7-19x worse.
6. **Repaste vs no-repaste** barely differs for conditioned model — the FiLM
   conditioning provides the known-region information, so repasting is redundant.
7. **Key comparison vs unconditional**:
   - Uncond Rev+RP@75: 253-631x → Multires Rev+RP@75: 2.7-5.1x
   - Uncond DDIM@249: 178-425x → Multires DDIM@249: 2.0-3.2x
   - Conditioning reduces iterative degradation by ~100x, but still can't make
     iterative methods competitive with single-step.
