# Helmholtz Matched Weakfullsup — Eps Prediction

## Hypothesis
Identical to `helmholtz_matched_weakfullsup` (x0 prediction) except
`prediction_target: eps`. The model predicts noise instead of clean x0.

Eps prediction is the standard DDPM formulation (Ho et al. 2020). It may
train more stably because the noise target has roughly unit variance at all
timesteps, whereas x0 targets have wildly different SNR across timesteps.

## Changes from weakfullsup (x0)
- `prediction_target: eps` (was `x0`)
- `voronoi_forward: false` (was `true`) — required because eps prediction
  recovers the diffusion source, which for voronoi_forward is the Voronoi fill,
  not GT. Standard DDPM: corrupt GT with noise, model predicts noise.
- All other settings identical

## Fixed: HelmholtzSupervisionLoss eps compatibility
The decomposition supervision now correctly decomposes the noise target
(not x0) when prediction_target=eps, so the heads are supervised to produce
solenoidal/irrotational noise components matching the helmholtz_matched
noise structure.

## Training Log

### 2025-03-14 — Launch
- Server 1 (RTX 5060 Ti), replacing failed FiLM mask_xt=true experiment
