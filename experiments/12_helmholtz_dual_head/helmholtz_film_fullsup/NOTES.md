# Helmholtz Split FiLM — Conditioned Helmholtz Dual-Head UNet

## What's being tested
Adding FiLM conditioning (mask + known observations) to the Helmholtz split-decoder
architecture, to give the model explicit access to observation locations and values
— similar to what makes V-CNN effective.

## Hypothesis
The unconditional helmholtz_split model (fullsup_800) achieves ~1.07x V-CNN at 0.5%
coverage, but cannot exploit observation locations/values during denoising. FiLM
conditioning should close this gap by injecting observation info at every resolution
level, while keeping the physics-constrained dual-head decoder (ψ→curl, φ→grad).

## Architecture
- Base: MyUNet_Helmholtz_Split (24.5M params) — shared encoder/bottleneck/low-res decoder,
  independent ψ and φ branches at high resolution
- Addition: HelmholtzCondEncoder (3ch → multi-scale features at 5 levels) + 12 FiLM layers
  (4 encoder + 1 bottleneck + 2 shared decoder + 2 per branch × 2 branches)
- Input: 5ch [x_t(2), mask(1), known_u(1), known_v(1)], split internally
- mask_xt=True: known region in x_t replaced with independent noise, forcing model to
  read observations from the FiLM conditioning pathway

## Key design choices
- FiLM over concat: more principled — model can't ignore conditioning, identity init
  means it starts as the unconditional version
- Helmholtz supervision still applies globally (not just missing region) via helmholtz_aux()
- Reconstruction MSE applies only to missing region (mask_xt behavior)
- Same hyperparameters as fullsup_800 for controlled comparison

## Log

### 2025-03-12 — FiLM v1: Sparse conditioning collapse (epoch 6)
- With sparse known-point conditioning (0.1-1% known), the CNN encoder
  washed out the signal → FiLM modulated as identity → model collapsed
  to predicting mean. Fixed by switching to Voronoi-filled conditioning.

### 2025-03-12 — FiLM v2: NaN explosion (epoch 31)
- Training healthy through epoch 30, then all NaN at epoch 31.
- Root cause: unbounded FiLM gamma in `γ·h` compounded through 12 spatial
  FiLM layers → float overflow.
- Band-aid fix: `gamma.clamp(-5, 5)`. Prevented NaN but not the instability.

### 2025-03-12 — FiLM v3: Delayed collapse (epoch ~115)
- Trained from epoch 8 checkpoint (best test 0.167).
- Epochs 9-107: healthy, test loss decreased 0.179 → 0.155
- Epoch 114: loss spike to 0.217
- Epoch 120: loss jumped to 0.891
- Epoch 125+: locked at 1.1413 (predicting mean) — same collapse pattern
- Gamma clamp prevented NaN but spatial per-pixel FiLM was fundamentally
  unstable. Old results archived to `results_v3_collapsed/`.

### 2025-03-12 — FiLM v4: AdaGN-style rewrite (CURRENT)
- **Root cause of all collapses**: spatial per-pixel `γ·h + β` without
  normalization. Per-pixel scales drift independently → compounds through
  12 layers → model collapses.
- **Fix**: Replaced FiLMLayer with AdaGN-style modulation (commit b265e0a):
  1. Pool spatial conditioning → channel-wise vectors (AdaptiveAvgPool2d)
  2. GroupNorm on features before modulation (bounded activations)
  3. Residual: `(1+γ)·GroupNorm(h)+β` with γ=0, β=0 init (starts as pure GN)
- Fresh start from scratch (architecture changed, old ckpts incompatible).
- Server 1 (RTX 5070 Ti), 800 epochs, batch=16, lr=5e-4 cosine w/ 10ep warmup.
- **Early results** (epoch 1-5): smooth monotonic descent, every epoch is BEST
  - Ep1: test=1.179, ep2: 0.338, ep3: 0.254, ep4: 0.218, ep5: 0.203
  - ~70s/epoch, no spikes, no instability. Looks fundamentally different from v1-v3.
