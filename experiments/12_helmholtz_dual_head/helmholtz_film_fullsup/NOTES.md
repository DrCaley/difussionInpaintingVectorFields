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
