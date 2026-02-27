# Algorithm Catalog

A reference for every paper-based technique used in this project, how we
adapted it, and what variants we test.  Each section covers one conceptual
"building block" (conditioning method, inpainting algorithm, noise
strategy, or divergence fix).  Cross-references point to the implementing
code and the experiment groups that exercise each combination.

---

## Table of Contents

1. [DDPM Foundation](#1-ddpm-foundation)
2. [Conditioning Method (how the model sees the mask)](#2-conditioning-method)
3. [Inpainting Algorithm (how we fill the missing region)](#3-inpainting-algorithm)
4. [Noise Strategy (what ε looks like)](#4-noise-strategy)
5. [Divergence-Free Projection (post-hoc repair)](#5-divergence-free-projection)
6. [Loss Functions](#6-loss-functions)
7. [Standardization](#7-standardization)
8. [Combination Matrix](#8-combination-matrix)

---

## 1. DDPM Foundation

**Paper**: Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020.

**Core idea**: Learn to reverse a gradual noising process.
Forward:  $x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\, \varepsilon$  
Reverse:  a UNet predicts either $\varepsilon$ or $x_0$, and we step backward
through the posterior $q(x_{t-1}|x_t, \hat x_0)$.

**Our implementation**: `ddpm/neural_networks/ddpm.py` → `GaussianDDPM`.
Wraps any UNet, stores the noise schedule ($\alpha_t, \bar\alpha_t, \beta_t$),
and provides `forward()` (add noise) and `backward()` (UNet predict).

**Prediction target** is a config key (`prediction_target`):

| Target | What UNet outputs | Inference function | Notes |
|--------|-------------------|--------------------|-------|
| `eps`  | Noise $\hat\varepsilon$ | `repaint_standard`, `mask_aware_inpaint`, `guided_inpaint`, `inpaint_generate_new_images` | Standard DDPM |
| `x0`   | Clean data $\hat x_0$ | `x0_full_reverse_inpaint` | Single-step estimate; posterior mean re-derived from $\hat x_0$ |

---

## 2. Conditioning Method

### 2a. Palette-style Concatenation

**Paper**: Saharia et al., "Palette: Image-to-Image Diffusion Models", SIGGRAPH 2022.

**Idea**: Concatenate conditioning information as extra input channels.
The UNet receives a 5-channel tensor `[x_t(2ch), mask(1ch), known_values(2ch)]`
and learns to use the conditioning natively.

**Our implementation**: `MyUNet_Inpaint` in `ddpm/neural_networks/unets/unet_inpaint.py`.
Config: `unet_type: concat`.  Identical to the base `MyUNet` except the first
conv block accepts 5 input channels instead of 2.

**Pros**: Simple; model can learn arbitrary conditioning relationships.  
**Cons**: Model can ignore the conditioning channels if the 2-channel `x_t` already
contains information about the known region.

### 2b. FiLM Conditioning

**Paper**: Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer", AAAI 2018.  
**Applied to diffusion**: Our adaptation injects conditioning at every resolution
level via learned scale ($\gamma$) and shift ($\beta$).

**Idea**: A separate encoder processes `[mask(1ch), known(2ch)]` → multi-scale
feature maps.  At each UNet block, the main features are modulated:
$h_{\text{out}} = \gamma(\text{cond}) \cdot h + \beta(\text{cond})$.

The UNet's main path sees only 2 channels ($x_t$), so there is no risk of
"leaking" the known region through the main path.  `mask_xt=true` training
replaces the known region of $x_t$ with independent noise to force the model
to rely on the FiLM conditioning.

**Our implementation**: `MyUNet_FiLM` in `ddpm/neural_networks/unets/unet_film.py`.
Config: `unet_type: film`.  Has a `ConditioningEncoder` producing features at
5 resolution levels and `FiLMLayer` modules after every UNet block.

**Pros**: Forces model to use conditioning; clean separation of signal and
context; ~9.8M params (similar to concat).  
**Cons**: More complex architecture; FiLM layers initialized to identity
($\gamma=1, \beta=0$) so model must learn to activate them.

**Training flag**: `mask_xt: true` — replace known region of $x_t$ with
independent noise during training.  Must match at inference.

### 2c. Unconditional (Standard UNet)

**Paper**: Ho et al. (original DDPM, 2020) — no conditioning at all.

**Idea**: The UNet sees only 2-channel $x_t$ and timestep $t$.  No mask,
no known values.  Inpainting is done externally via RePaint's copy-paste.

**Our implementation**: `MyUNet` in `ddpm/neural_networks/unets/unet_xl.py`.
Config: `unet_type: standard`.  The model learns an unconditional prior
over ocean vector fields; inpainting guidance comes entirely from the
RePaint algorithm at inference time.

**Pros**: Simplest model; decouples generation from inpainting.  
**Cons**: No access to conditioning during denoising; relies entirely on
copy-paste to steer toward known values.

---

## 3. Inpainting Algorithm

### 3a. RePaint

**Paper**: Lugmayr et al., "RePaint: Inpainting using Denoising Diffusion Probabilistic Models", CVPR 2022.

**Idea**: At each reverse step $t$, the known region is replaced with its
independently forward-noised version, then the model denoises the whole field.
A "resample" loop re-noises and re-denoises multiple times per step to allow
information to diffuse across the boundary.

The algorithm at step $t$:
1. Denoise: $x_{t-1}^{\text{denoised}} = \text{DDPM\_reverse}(x_t)$
2. Forward-noise known region: $x_{t-1}^{\text{known}} = \sqrt{\bar\alpha_{t-1}}\, x_0 + \sqrt{1-\bar\alpha_{t-1}}\, \varepsilon$
3. Paste: $x_{t-1} = x_{t-1}^{\text{known}} \cdot M_{\text{known}} + x_{t-1}^{\text{denoised}} \cdot M_{\text{miss}}$
4. (Optional) Re-noise $x_{t-1} \to x_t$ and repeat steps 1–3 (`resample_steps` times)

**Our implementation**: `repaint_standard()` in `ddpm/utils/inpainting_utils.py` line 779.
Works with any UNet type but designed for the unconditional `standard` UNet.
Supports both `eps` and `x0` prediction.

**Our variant** — **RePaint + CG projection** (step 3.5):
After paste, the copy-paste boundary introduces a divergence discontinuity.
We optionally apply the CG streamfunction projection
(`forward_diff_project_div_free`) after each paste step, then re-stamp the
known region.  This fixes boundary divergence at every step, not just at the end.

Config flags:
- `project_div_free=True`: apply CG projection after each paste step
- `project_final_steps=N`: apply N rounds of project→restore-known at the end

**Key parameter**: `resample_steps` (typically 5–10) controls how many
resample iterations to run per timestep.

### 3b. Full-Reverse x₀-Prediction Inpainting

**Paper**: Adapted from the DDPM posterior chain, not a single paper.
Conceptually similar to Palette but with explicit posterior stepping.

**Idea**: Run the full 250-step reverse process using the DDPM posterior mean
parameterized by the predicted $\hat x_0$:

$$\mu_t = \frac{\sqrt{\bar\alpha_{t-1}}\, \beta_t}{1-\bar\alpha_t}\, \hat x_0 + \frac{\sqrt{\alpha_t}\,(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\, x_t$$

The FiLM UNet provides conditioning at every step.  With `mask_xt=true`,
the known region of $x_t$ is replaced with independent noise, so the model
can only read known values through FiLM.

**Our implementation**: `x0_full_reverse_inpaint()` in `inpainting_utils.py` line 429.

**Our variant** — with optional RePaint resampling (`repaint_steps > 0`) and
CG projection (`project_steps > 0`) applied to the final result.

### 3c. Mask-Aware Denoising (Palette-style)

**Paper**: Saharia et al., "Palette", 2022.

**Idea**: The UNet directly receives `[x_t, mask, known_values]` as input.
Standard reverse process — no copy-paste needed — the model learns to
denoise conditioned on the known context.

**Our implementation**: `mask_aware_inpaint()` in `inpainting_utils.py` line 683.
Uses eps-prediction.  After the reverse process, the known region is restored.
Compatible with `mask_xt` training.

### 3d. Classifier-Free Guidance (CFG) Inpainting

**Paper**: Ho & Salimans, "Classifier-Free Diffusion Guidance", NeurIPSW 2021.

**Idea**: During training, randomly drop conditioning (mask and known values
set to zero) with probability `p_uncond`.  At inference, run two forward
passes — conditioned and unconditioned — and blend:

$$\varepsilon = \varepsilon_{\text{uncond}} + w \cdot (\varepsilon_{\text{cond}} - \varepsilon_{\text{uncond}})$$

$w > 1$ amplifies conditioning influence.

**Our implementation**: `mask_aware_inpaint_cfg()` in `inpainting_utils.py` line 587.
Training config: `p_uncond: 0.1` (or similar).  Inference: `guidance_scale: 3.0`.

### 3e. Gradient-Guided Inpainting

**Paper**: Dhariwal & Nichol, "Diffusion Models Beat GANs on Image Synthesis", NeurIPS 2021, §4 (classifier guidance).  Adapted for self-guided reconstruction.

**Idea**: Instead of copy-paste, steer the reverse chain with gradients of
two soft losses:

$$L = \lambda_b \| (\hat x_0 - x_{\text{known}}) \cdot (1-M) \|^2 + \lambda_d \| \nabla \cdot \hat x_0 \|^2$$

At each step, compute $\hat x_0$ via Tweedie's formula, backprop through
the UNet to get $\nabla_{x_t} L$, and shift the posterior mean by
$-\lambda \nabla$.

**Our implementation**: `guided_inpaint()` in `inpainting_utils.py` line 203.
Parameters: `guidance_scale_boundary`, `guidance_scale_div`.

**Pros**: No copy-paste boundary artifacts; directly penalizes divergence.  
**Cons**: Requires gradient computation through UNet at every step (slow,
memory-heavy); tuning the guidance scales is tricky.

### 3f. Legacy Copy-Paste Inpainting

**Our implementation**: `inpaint_generate_new_images()` in `inpainting_utils.py` line 941.
Older function with per-step boundary fix via the original `combine_fields()`
and spectral projection.  Uses eps-prediction with an unconditional UNet.
Kept for backward compatibility; superseded by `repaint_standard()`.

---

## 4. Noise Strategy

All noise strategies must produce (B, 2, H, W) tensors with approximately
unit variance.  See `ddpm/utils/noise_utils.py` for implementations and
`ddpm/protocols.py` → `NoiseStrategyProtocol` for the formal contract.

### 4a. Gaussian Noise (baseline)

**Standard**: $\varepsilon \sim \mathcal{N}(0, I)$ — each element i.i.d.

Config: `noise_function: gaussian`.  Class: `GaussianNoise`.  
Uses Gaussian scaling ($\sqrt{1-\bar\alpha_t}$).  
Works with any standardizer.

### 4b. Forward-Difference Divergence-Free Noise

**Our construction**: sample $\psi \sim \mathcal{N}(0,I)$ on an $(H{+}1) \times (W{+}1)$
grid, then compute velocity via forward-difference curl:

$$u_{i,j} = \psi_{i+1,j} - \psi_{i,j}, \quad v_{i,j} = -(\psi_{i,j+1} - \psi_{i,j})$$

**Why it's divergence-free**: the discrete forward-difference divergence
$\text{div}_{i,j} = (u_{i,j+1} - u_{i,j}) + (v_{i+1,j} - v_{i,j})$ is
identically zero for any $\psi$.  This is by algebraic cancellation, not
approximation.

Config: `noise_function: forward_diff_div_free`.  Class: `ForwardDiffDivFreeNoise`.  
Output is rescaled to unit variance via a cached calibration constant.  
**Requires** `UnifiedZScoreStandardizer`.

### 4c. Spectral (Central-Difference) Divergence-Free Noise

**Construction**: sample $\psi \sim \mathcal{N}(0,I)$ on an $H \times W$ grid,
then compute velocity via **central-difference** curl:

$$u_{i,j} = \frac{\psi_{i+1,j} - \psi_{i-1,j}}{2}, \quad v_{i,j} = -\frac{\psi_{i,j+1} - \psi_{i,j-1}}{2}$$

**Difference from 4b**: div-free under the central-difference operator, NOT
under the forward-difference operator.  Has boundary artifacts (no valid
stencil at edges).  Rescaled to unit variance.

Config: `noise_function: spectral_div_free`.  Class: `SpectralDivFreeNoise`.  
**Requires** `UnifiedZScoreStandardizer`.

### 4d. GP-Based Divergence-Free Noise (legacy)

**Paper**: Inspired by Macêdo & Castro (divergence-free Gaussian processes).

Generates divergence-free fields via an incompressible GP kernel.
Timestep-dependent (clamped to $t \ge 1$).  Computationally expensive.

Config: `noise_function: div_free`.  Class: `DivergenceFreeNoise`.  
**Requires** `UnifiedZScoreStandardizer`.

### 4e. Helmholtz–Hodge Decomposition Noise (legacy)

Generates Gaussian noise, then projects out the irrotational component using
Helmholtz–Hodge decomposition.  Does NOT use Gaussian scaling
(`get_gaussian_scaling() == False`).

Config: `noise_function: hh_decomp_div_free`.  Class: `HH_Decomp_Div_Free`.  
**Requires** `UnifiedZScoreStandardizer`.

---

## 5. Divergence-Free Projection

Applied at inference time (after copy-paste or at end of reverse chain) to
fix divergence introduced by the inpainting algorithm.  All projections are
in `ddpm/utils/inpainting_utils.py`.

### 5a. Forward-Difference CG Streamfunction Projection

Finds $\psi$ on $(H{+}1) \times (W{+}1)$ whose forward-difference curl best
fits the input velocity, using Conjugate Gradient on the normal equations
$A^T A \psi = A^T \mathbf{v}$.  Output is **exactly** div-free under the
forward-difference operator (same construction as `ForwardDiffDivFreeNoise`).

Function: `forward_diff_project_div_free(vel, cg_iters=200)` — line 83.  
Used with energy-preserving rescaling:
```python
pre_energy = (v ** 2).sum()
v = forward_diff_project_div_free(v)
v = v * (pre_energy / (v ** 2).sum()).sqrt()
```

**Converges** in ~60 iterations for 64×128 grid.  
**Must match** `ForwardDiffDivFreeNoise` — mixing operators breaks idempotency.

### 5b. Spectral FFT Helmholtz Projection

Exact Helmholtz decomposition in Fourier space.  Removes the irrotational
component at every mode: $\hat v_{\text{df}} = \hat v - k(k \cdot \hat v)/|k|^2$.

Function: `spectral_project_div_free(vel)` — line 161.  
Assumes periodic BC.  Compatible with `SpectralDivFreeNoise`.

### 5c. Jacobi Poisson Projection

Classical Helmholtz-Hodge via Jacobi iteration on the Poisson equation
$\nabla^2 p = \nabla \cdot \mathbf{v}$, then $\mathbf{v}_{\text{df}} = \mathbf{v} - \nabla p$.

Function: `project_div_free_2d(vel, jacobi_iters=50)` — line 73.  
5-point stencil, Neumann BC.  Slow convergence compared to CG.

---

## 6. Loss Functions

Defined in `ddpm/helper_functions/loss_functions.py`.

| Loss | Class | Description |
|------|-------|-------------|
| `mse` | `MSELossStrategy` | Standard MSE between prediction and target |
| `physical` | `PhysicalLossStrategy` | MSE + weighted divergence penalty on predicted ε |
| `best_loss` | `HotGarbage` | (Legacy) experimental loss — not recommended |

When `mask_xt=true`, loss is computed only in the missing region (where
the model has signal to predict).

---

## 7. Standardization

Defined in `ddpm/helper_functions/standardize_data.py`.

| Key | Class | Description | Div-free safe? |
|-----|-------|-------------|----------------|
| `zscore` | `ZScoreStandardizer` | Per-component z-score ($u, v$ get different $\mu, \sigma$) | **No** — breaks $\nabla \cdot v = 0$ |
| `zscore_unified` | `UnifiedZScoreStandardizer` | Shared ($\mu, \sigma$) for both components | **Yes** — scalar transform preserves linearity |

**Auto-resolution**: in `base_inpaint.yaml`, `standardizer_type: auto` looks up
the noise function in the `standardizer_by_noise` mapping.

---

## 8. Combination Matrix

Valid combinations, showing which building blocks work together:

| Config Name | UNet | Prediction | Inpainting | Noise | Projection | Status |
|-------------|------|------------|------------|-------|------------|--------|
| `fwd_diff_divfree_x0pred_film_t250` | FiLM | x0 | `x0_full_reverse_inpaint` | fwd_diff_div_free | CG (end-of-chain) | **Trained** (epoch 380, best 0.0441) |
| `repaint_cg_fwd_divfree_eps_t250` | standard | eps | `repaint_standard` + CG | fwd_diff_div_free | CG (per-step) | **Experiment created** |
| spectral baseline | FiLM | x0 | `x0_full_reverse_inpaint` | spectral_div_free | spectral FFT | Config ready |
| gaussian baseline | FiLM | x0 | `x0_full_reverse_inpaint` | gaussian | none | Config ready |
| palette-concat | concat | eps | `mask_aware_inpaint` | gaussian | none | Not yet created |
| CFG inpainting | concat | eps | `mask_aware_inpaint_cfg` | any | none | Not yet created |
| gradient-guided | standard | eps | `guided_inpaint` | fwd_diff_div_free | none (implicit) | Not yet created |

### Incompatible combinations (enforced by `ddpm/protocols.py`):

- `prediction_target: x0` + `repaint_standard()` ← works but designed for eps
- `prediction_target: eps` + `x0_full_reverse_inpaint()` ← algorithm expects x0
- Div-free noise + per-component `ZScoreStandardizer` ← breaks div-free
- `ForwardDiffDivFreeNoise` + `spectral_project_div_free()` ← operator mismatch
- `unet_type: standard` + `mask_xt: true` ← no conditioning channels to read

---

## Adding a New Algorithm

1. Implement it in the appropriate module (noise → `noise_utils.py`,
   inpainting → `inpainting_utils.py`, UNet → `unets/`).
2. Add it to the relevant registry/compatibility table in `ddpm/protocols.py`.
3. Add a section to this catalog.
4. Add an entry to `design/algorithms.yaml`.
5. Create an experiment config in the appropriate `experiments/NN_*/` group.
6. Run `--dry-run` to validate, `--smoke` for 3-epoch test.
