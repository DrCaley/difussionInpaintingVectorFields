# Weakfullsup + Detach Heads

## Purpose
Isolate the effect of `detach_heads` by using the exact same setup as
`helmholtz_matched_weakfullsup` (unconditional, helmholtz_split UNet) but
with detached heads and lambda_decomp=1.0.

## Controlled variables (same as weakfullsup)
- UNet: helmholtz_split (unconditional, 2ch input)
- Noise: helmholtz_matched
- Prediction: x0
- Training: bs=16, lr=5e-4, cosine, 800 epochs, EMA

## Varied vs weakfullsup
| Parameter | weakfullsup | this |
|-----------|-------------|------|
| detach_heads | false | **true** |
| lambda_decomp | 0.1 | **1.0** |

## Log

### 2025-03-18: Training progress
- Epoch 277/800, best EMA test loss = 0.1138 (epoch 249)
- Loss plateau since ~epoch 180 (0.117 → 0.114)
- Training on Server 1 (RTX 5060 Ti)

### 2025-03-18: Inference evaluation (at epoch 249 checkpoint)

Evaluated on Server 2 using fixed `eval_helmholtz_split.py` (corrected build_masks).
10 samples per coverage level, single-step t=25.

| Coverage | V-CNN MSE | 1S@t=25 MSE | vs V-CNN | Div RMS (DDPM) | Div RMS (V-CNN) |
|----------|-----------|-------------|----------|----------------|-----------------|
| 0.5%     | 0.000688  | 0.000694    | 1.008x   | 0.00533        | 0.00652         |
| 1.0%     | 0.000450  | 0.000435    | 0.966x   | 0.00541        | 0.00662         |
| 2.0%     | 0.000249  | 0.000270    | 1.086x   | 0.00571        | 0.00672         |

**Observations:**
- At 1%: **3.4% better than V-CNN** — best result at this coverage
- At 0.5%: essentially tied (within 1%)
- At 2%: 8.6% worse than V-CNN
- Divergence consistently ~18% lower than V-CNN
- RevChain catastrophically bad (313-841x) — base model has no mask conditioning,
  so repaste doesn't work. Ignore these numbers.

**Head diagnostics (0.5% coverage):**
- Cancel ratio: 1.2x (excellent, better than multires 1.3x)
- cos(ψ-head, GT_sol): 0.969
- cos(φ-head, GT_irr): 0.915
- Pred sol fraction: 83.4%

**Comparison with multires at same coverages:**

| Coverage | multires vs V-CNN | weakfullsup_detach vs V-CNN |
|----------|-------------------|---------------------------|
| 0.5%     | 0.998x            | 1.008x                    |
| 1.0%     | 0.971x            | 0.966x ← better           |
| 2.0%     | 1.116x            | 1.086x ← better           |

weakfullsup_detach slightly better at 1-2% but slightly worse at 0.5%.
Both architectures competitive with V-CNN in the training range.

### 2025-03-18: Alternative diffusion inference strategies

Tested 12 strategies beyond the baseline single-step. Ran on Server 2, 10 samples
per coverage, same val indices as above.

| Method | 0.5% vs V-CNN | 1.0% vs V-CNN | 2.0% vs V-CNN |
|--------|--------------|--------------|--------------|
| **1S@t=25** (baseline) | 1.008x | **0.966x** | 1.086x |
| **1S@t=50** | 1.011x | **0.964x** | 1.090x |
| **Ens5-divT** (avg 5 t-values) | 1.006x | **0.971x** | 1.089x |
| Warm2 (50→25) | 1.045x | 1.032x | 1.267x |
| Warm3 (75→50→25) | 1.104x | 1.140x | 1.524x |
| NoRepaste@10 | 100.6x | 142.6x | 271.0x |
| NoRepaste@25 | 221.6x | 346.1x | 586.5x |
| SoftRC@10 | 105.6x | 146.1x | 272.6x |
| SoftRC@25 | 220.0x | 322.8x | 586.4x |
| DDIM@50-5s | 54.4x | 83.4x | 153.7x |
| DDIM@50-10s | 76.2x | 119.0x | 177.4x |
| DDIM@100-10s | 83.5x | 120.5x | 193.6x |

**Conclusions:**
1. **All multi-step iterative methods catastrophically fail** (50-600x worse).
   The unconditional model was trained only on noised Voronoi fills; after one
   DDPM reverse step the intermediate x_{t-1} is out-of-distribution (noised
   prediction, not noised Voronoi fill). This applies equally to NoRepaste,
   SoftRevChain, and DDIM.

2. **Single-step is unbeatable for this architecture.** Best: 1S@t=50 at 1%
   (0.964x vs V-CNN). Results are nearly invariant to t in [25, 125].

3. **Ensemble-diverse-t gives negligible improvement** (1.006x vs 1.008x at
   0.5%) at 5x compute cost. Not worth it.

4. **Warm-restart hurts** despite re-voronoi between steps. Error compounds
   at each refinement step (+3-5% per step). The re-standardize→unstandardize
   round-trip accumulates error faster than the denoising removes it.

5. **Fundamental limitation**: this unconditional model needs conditioning
   channels (mask/obs) to support any iterative diffusion. Without it, the
   only viable inference is single-shot — essentially a learned denoising
   autoencoder on Voronoi fill.

### 2025-03-18: Traditional literature diffusion methods

Tested standard diffusion inpainting approaches from the literature at
proper noise levels (t=75-249), including full T=249 reverse chains.
5 samples per coverage. Script: `scripts/eval_traditional_diffusion.py`.

Methods tested:
- **Rev+RP@t**: Full reverse chain from t with repaste at each step (Song/Ho inpainting)
- **Rev-NR@t**: Full reverse chain from t, no repaste (Palette-style)
- **RePaint@t-rN**: RePaint (Lugmayr et al. 2022) with N resampling steps per timestep
- **PureNoise+RP**: Start from pure noise (t=249), full reverse + repaste
- **PureNoise-NR**: Start from pure noise, no repaste
- **DDIM@t-Ns**: DDIM deterministic reverse with N steps from t

| Method | 0.5% vs V-CNN | 1.0% vs V-CNN | 2.0% vs V-CNN |
|--------|--------------|--------------|--------------|
| **1S@t=25** (baseline) | **0.974x** | 1.024x | **0.995x** |
| Rev+RP@75 | 253.2x | 408.2x | 631.5x |
| Rev+RP@150 | 245.2x | 370.8x | 600.9x |
| Rev+RP@249 | 205.3x | 264.6x | 412.3x |
| Rev-NR@75 | 255.1x | 403.8x | 640.3x |
| Rev-NR@150 | 247.1x | 376.8x | 600.9x |
| RePaint@75-r5 | 343.3x | 542.4x | 771.3x |
| RePaint@100-r5 | 359.4x | 475.2x | 766.6x |
| RePaint@150-r5 | 254.0x | 363.7x | 585.9x |
| RePaint@75-r10 | 336.4x | 534.2x | 862.3x |
| PureNoise+RP | 376.2x | 536.8x | 864.8x |
| PureNoise-NR | 378.6x | 544.0x | 886.9x |
| DDIM@150-25s | 186.0x | 283.0x | 414.8x |
| DDIM@249-50s | 177.7x | 258.3x | 425.3x |

**Conclusions:**
1. **Every traditional diffusion method is catastrophically bad** (178-887x worse).
   This confirms the unconditional model cannot function as a diffusion sampler.
2. **Even DDIM** (the best iterative method) is 178-425x worse — still orders of
   magnitude from usable.
3. **RePaint with resampling makes things worse**, not better. More resampling =
   more error accumulation (r10 worse than r5).
4. **Repaste vs no-repaste barely matters** — the model's predictions are so far
   out of distribution after one reverse step that repasting known values doesn't
   help.
5. **Pure noise start (true generation)** is no worse than starting from noised
   Voronoi fill — both rapidly diverge. The model never learned to generate from
   noise.
6. **Higher t slightly better** in some cases (Rev+RP@249 < Rev+RP@75) — likely
   because at high noise levels the Voronoi signal still dominates x_t.
