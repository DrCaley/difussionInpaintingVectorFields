# 12 — Helmholtz-Decomposed Dual-Head Architecture

## Research Question

Can a UNet with physics-informed output heads — one predicting a
streamfunction ψ (solenoidal via curl) and one predicting a velocity
potential φ (irrotational via gradient) — learn better ocean velocity
inpainting than a standard 2-channel direct-output UNet?

**Motivation**: Helmholtz-Hodge decomposition of our Rams Head dataset
shows **32% of kinetic energy is irrotational** (divergent). A pure
streamfunction model would have a 30% irreducible error floor. The
dual-head architecture lets the model: (a) structurally guarantee zero
divergence in the solenoidal component, (b) explicitly model the real
divergent dynamics, and (c) provide built-in diagnostics via the ψ/φ
split.

See `design/physics-proposals.md` § Proposal 5 and § Dataset Divergence
Analysis for full motivation.

## Controlled Variables
- Same training data (Voronoi warm-start, on-the-fly random masks)
- Same DDPM settings (Gaussian noise, 250 steps, x0 prediction)
- Same ~18M param count (shared encoder/decoder, only output heads differ)
- Same evaluation protocol

## Varied Variables
- **UNet architecture**: `standard_attn` (direct 2ch output) → `helmholtz`
  (dual-head ψ/φ with physics operators)
- **Output structure**: Raw velocity → curl(ψ) + grad(φ)
- **φ head initialisation**: Zeros (model starts solenoidal-biased)

## Experiments

### `helmholtz_baseline/`
First test: Helmholtz dual-head UNet with Voronoi warm-start, single-stage
training. Compare against the existing `standard_attn` Voronoi baseline.

### (future) `helmholtz_stage3/`
If baseline works: 3-stage detached rollout with Helmholtz UNet.

### (future) `helmholtz_spectral_loss/`
Combine with Proposal 2 (energy spectrum loss).
