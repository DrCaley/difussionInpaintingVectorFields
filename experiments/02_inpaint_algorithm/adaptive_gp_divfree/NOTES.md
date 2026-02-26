# Adaptive GP-Refined RePaint + CG Div-Free Projection

## Research Question

**Does combining uncertainty-adaptive GP-refined RePaint with div-free noise
and CG projection outperform the same method with Gaussian noise?**

The adaptive GP-refined approach was shown to beat GP alone 85% of the time
(avg 0.744× GP MSE) with the Gaussian attention UNet. This experiment tests
whether the div-free noise model + CG projection can match or exceed that
result while also producing physically consistent (divergence-free) outputs.

## Architecture

| Property | Value |
|----------|-------|
| UNet | `standard` — unconditional (2ch input: u, v only) |
| Prediction target | eps |
| Noise strategy | `forward_diff_div_free` |
| Noise steps | 250 |
| Beta schedule | `min_beta=0.0001, max_beta=0.02` |
| Standardizer | `zscore_unified` (auto-resolved) |
| Conditioning | NONE — model never sees mask or known values |
| `mask_xt` | false |
| Model source | `../repaint_cg/results/inpaint_forward_diff_div_free_t250_best_checkpoint.pt` |
| Training | 357 epochs, converged (best test loss 0.003686) |

## Inpainting Method

`repaint_gp_init_adaptive()` with:
- `noise_strategy = forward_diff_div_free` (matching training noise)
- `project_div_free = True` (CG streamfunction projection every step)
- `project_final_steps = 3` (extra CG cleanup at end)
- `noise_floor = 0.2` (best value from Gaussian sweeps)
- `t_start = 75` (best value from Gaussian sweeps)
- `resample_steps = 5` (standard RePaint resampling)

## Comparisons (within this experiment)

1. **GP baseline** — standard GP interpolation
2. **Adaptive GP + div-free + CG** — our full method
3. **Uniform GP-refined + div-free + CG** — ablation (no adaptive weighting)
4. **Adaptive GP + div-free, no CG** — ablation (no projection)

## Key Hypotheses

1. Adaptive GP-refined should beat GP alone (>80% win rate), same as Gaussian
2. CG projection should produce lower divergence than no-projection variant
3. Div-free noise + CG may provide better physical consistency than Gaussian,
   even if MSE is comparable
4. Divergence at mask boundary should be significantly reduced vs plain RePaint

## Baseline Reference

| Method | Model | Wins/100 vs GP | Avg ratio |
|--------|-------|----------------|-----------|
| Adaptive GP (Gaussian) | standard_attn | 85/100 | 0.744× |
| Uniform GP (Gaussian) | standard_attn | 74/100 | 0.812× |
| Plain RePaint (Gaussian) | standard_attn | ~65/100 | ~0.85× |

## Observations

### 2026-02-23: SDEdit with div-free noise — identity reconstruction

Tested iterative SDEdit (no RePaint) with the repaint_cg div-free model at
various t values (t=5, t=25, t=75) and iteration counts (1–10). After fixing
critical bugs in the noise-scaling formulas (missing √(1-ᾱ) on forward noise,
wrong x0 reconstruction), the model consistently returns near-identity
reconstructions:

| Config | Avg ratio vs GP | Notes |
|--------|-----------------|-------|
| t=5, 1 iter | 0.998x | Near identity |
| t=5, 10 iters | 0.997x | Still near identity |
| t=25, 1 iter | 1.001x | Near identity |
| t=75, 1 iter | 1.009x | Near identity, 1/5 wins |

CG projection consistently hurts (1.46x GP at t=75). The model is a good
denoiser but SDEdit without known-data replacement doesn't improve on GP
because the GP field is already on-manifold — adding and removing the same
noise returns to the same place.

**Key insight:** SDEdit treats the whole field uniformly. Without an
information asymmetry between known and unknown pixels, there's no mechanism
to "improve" the masked region. This is fundamentally different from RePaint,
which provides hard known-data constraints at every step.

---

## Future Directions: Non-RePaint approaches with div-free noise

The following approaches avoid RePaint (no known-region pasting during reverse)
while using div-free noise to break the symmetry that causes identity
reconstruction. All are designed to work with the existing repaint_cg model
(8.8M params, eps-prediction, forward_diff_div_free noise, 250 steps).

### Approach 1a: GP-variance-weighted velocity noise (TESTED — FAILED)

**Script:** `scripts/test_variance_weighted_sdedit.py`

Add noise scaled by GP posterior variance — high variance (uncertain) pixels
get full noise, low variance (confident) pixels get minimal noise:

$$x_{\text{noisy}} = \sqrt{\bar\alpha_t}\, x_{\text{GP}} + \sqrt{1-\bar\alpha_t}\, w(x,y)\, \varepsilon_{\text{divfree}}$$

where $w(x,y) = \text{noise\_floor} + (1 - \text{noise\_floor}) \cdot \sigma_{\text{GP}}^{\text{norm}}(x,y)$
with $\sigma_{\text{GP}}^{\text{norm}}$ normalised to [0, 1].

**PROBLEM:** Multiplying velocity noise ε by a spatially-varying weight
∇·(w·ε) = ε·∇w ≠ 0, so this BREAKS pixel-wise divergence-free. Also,
non-uniform noise violates the training distribution (model trained on
uniform noise per timestep). Result: 0/5 wins, 1.030x GP (worse than
uniform SDEdit at 1.010x GP).

### Approach 1b: Streamfunction-weighted noise (TESTED)

**Script:** `scripts/test_streamfunc_weighted_sdedit.py`

Fix the div-free issue from 1a by weighting the STREAMFUNCTION ψ
before taking the curl:

$$\psi_{\text{weighted}} = w(x,y) \cdot \psi_{\text{white}}$$
$$(u, v) = \text{curl}(\psi_{\text{weighted}})$$

Since curl(any scalar) has EXACTLY zero discrete divergence (algebraic
cancellation), this maintains pixel-wise ∇·ε = 0 regardless of w.
The weight map w is interpolated from (H,W) variance grid to the
(H+1, W+1) streamfunction grid.

**Results (2026-02-23, t=75, iters=1, nf=0.1):**

| Samp | Mask  | GP MSE   | ψ-Weight | Uniform  | ψW/GP  | max|div(ε)| |
|------|-------|----------|----------|----------|--------|-----------|
|    0 | 90.3% | 0.000717 | 0.000711 | 0.000713 | 0.991x | 2.4e-07   |
|    1 | 93.7% | 0.000640 | 0.000648 | 0.000650 | 1.013x | 2.4e-07   |
|    2 | 91.2% | 0.002882 | 0.002915 | 0.002921 | 1.012x | 2.4e-07   |
|    3 | 89.8% | 0.000511 | 0.000514 | 0.000514 | 1.006x | 4.8e-07   |
|    4 | 90.2% | 0.000594 | 0.000601 | 0.000605 | 1.012x | 2.4e-07   |

ψ-Weight vs GP: 1/5 wins, avg 1.007x. Uniform vs GP: 1/5 wins, avg 1.010x.
ψ-Weight beats uniform 5/5 times (marginal: avg 0.997x).

**Div-free verification: PASSED** — max|div(ε)| ~ 2.4–4.8 × 10⁻⁷ (floating
point noise only). Exactly pixel-wise divergence-free as predicted.

**Analysis:** The div-free guarantee works perfectly, but the fundamental
SDEdit identity-reconstruction problem persists — weighting the streamfunction
produces slightly less noise overall (weight mean ~0.18–0.21) so the model
returns even closer to the input. The ψ-weighting is marginally better
than uniform SDEdit but neither meaningfully improves on GP without
RePaint-style known-data anchoring.

### Approach 2: Langevin score refinement

Skip the forward noise entirely. Use the trained model as a score estimator
and run Langevin dynamics:

$$x_{k+1} = x_k + \frac{\eta}{2}\, \nabla \log p(x_k) + \sqrt{\eta}\, z_{\text{divfree}}$$

where the score is $\nabla \log p(x) \approx -\varepsilon_\theta(x, t) / \sqrt{1-\bar\alpha_t}$
at a small fixed t. No forward corruption needed — directly walks uphill on
the learned density while div-free noise maintains exploration. Needs careful
step-size tuning.

### Approach 3: Masked-ψ noise injection (TESTED — FAILED)

**Script:** `scripts/test_masked_psi_sdedit.py`

Add noise ONLY in the masked (unknown) region, keep known pixels clean, then
denoise the whole field. To maintain exact pixel-wise div-free, mask the
streamfunction ψ before taking the curl:

1. Build binary mask m_ψ on (H+1, W+1) grid: for each KNOWN velocity pixel
   (i,j), zero out ψ nodes {(i,j), (i+1,j), (i,j+1)}
2. ψ_masked = m_ψ · ψ_white, then (u,v) = curl(ψ_masked)/scale
3. Forward: x_t = √ᾱ_t · x_GP + √(1-ᾱ_t) · ε_masked
4. Standard DDPM reverse from t→0 (uniform div-free z noise)
5. Paste known GT back

**Properties:**
- EXACT zero noise at known-interior pixels (all 3 ψ nodes zeroed)
- Full training-distribution noise at masked-interior pixels
- 1-pixel transition band at boundary (some ψ nodes shared → reduced amplitude)
- EXACT pixel-wise div-free (curl of any scalar ≡ 0)

**Verification:** max|div(ε)| ~ 4.8e-07 (PASS). Noise RMS ~0.03–0.06 in
known region, ~0.93–0.98 in masked region (close to 1 as expected).

**Results (2025-02-24, 10 samples, t-sweep 25/50/75/100):**

| t | Mψ wins/GP | Avg Mψ/GP | U wins/GP | Avg U/GP | Mψ beats U |
|---|------------|-----------|-----------|----------|------------|
| 25 | 8/10 | 0.999x | 8/10 | 0.998x | 5/10 |
| 50 | 6/10 | 0.999x | 6/10 | 0.998x | 5/10 |
| 75 | 6/10 | 1.000x | 6/10 | 0.999x | 6/10 |
| 100 | 5/10 | 1.003x | 6/10 | 1.002x | 7/10 |

Both masked-ψ and uniform SDEdit produce near-identity results (~0.998–1.003x GP).
Masked-ψ does NOT consistently outperform uniform SDEdit (coin-flip difference).

**Root cause:** The information asymmetry exists only at the forward step.
During the reverse process, the model was trained on UNIFORMLY noised images
and has no mechanism to exploit the partially-clean input. At each reverse
step, uniform z noise is added and the model applies standard denoising —
it doesn't know which pixels are anchors vs. which need reconstruction.
Additionally, the ~90% mask coverage means very few known pixels exist
as anchors. Without per-step known-data replacement (i.e., RePaint), the
model cannot condition on the known region during denoising.

**Key insight:** The fundamental limitation of ALL SDEdit-based approaches
(uniform, ψ-weighted, and masked-ψ) without RePaint is that an unconditional
model (no mask channel) has no way to distinguish known from unknown pixels
during the reverse process. The spatial structure of the forward noise is
irrelevant — what matters is the reverse process's ability to condition on
known data, which requires either:
1. Per-step replacement (RePaint)
2. Explicit mask conditioning (FiLM/concat architectures)
3. Score-based guidance (Approach 2, untested)

### Approach 4: Tweedie single-step projection (TESTED — FAILED)

**Script:** `scripts/test_tweedie_projection.py`

One-shot: evaluate the model at moderate t with the GP field as input
(no actual noise added):

$$\hat{x}_0 = \frac{x_{\text{GP}} - \sqrt{1-\bar\alpha_t}\, \varepsilon_\theta(x_{\text{GP}}, t)}{\sqrt{\bar\alpha_t}}$$

This asks "if this field were noisy at timestep t, what clean field explains
it?" The model's ε prediction reveals what it considers "noise-like" about
the GP field. Vary t to control reshaping aggressiveness. Cheapest approach:
single forward pass, no reverse chain.

**Results (2026-02-23, 10 samples, sweep t=1..150):**

| t | Wins/10 | Avg ratio | Med ratio | min | max |
|---|---------|-----------|-----------|-----|-----|
| 1 | 8/10 | 1.000x | 1.000x | 0.999x | 1.000x |
| 5 | 5/10 | 1.000x | 1.000x | 0.998x | 1.004x |
| 10 | 5/10 | 1.003x | 1.001x | 0.997x | 1.019x |
| 25 | 2/10 | 1.031x | 1.019x | 0.992x | 1.145x |
| 50 | 1/10 | 1.172x | 1.112x | 0.998x | 1.667x |
| 75 | 0/10 | 1.509x | 1.340x | 1.044x | 2.711x |
| 100 | 0/10 | 2.243x | 1.960x | 1.139x | 4.645x |
| 150 | 0/10 | 6.760x | 5.843x | 1.824x | 15.055x |

Also tested iterative Tweedie (3×), which is uniformly worse.

**Root cause:** GP errors don't match the model's learned noise distribution.
The model was trained on div-free noise ε with a specific spatial structure
(curl of white-noise streamfunction). GP interpolation errors have completely
different spatial structure — they are smooth, correlated, and concentrated
at long-range extrapolation areas. At low t, the model expects almost no
noise and barely changes the input (identity). At high t, the model expects
heavy noise and aggressively reshapes, but its reconstruction is worse because
the input doesn't look like a noisy training sample.

**Conclusion:** Tweedie projection is fundamentally limited by the mismatch
between GP error structure and training noise structure. The model can only
remove noise it was trained to see.
