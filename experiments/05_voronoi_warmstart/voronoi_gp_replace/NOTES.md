# Voronoi GP-Replace — Experiment Notes

## 2026-02-27 — Initial setup

**Goal**: Test Voronoi tessellation as warm-start replacement for GP mean in
the S6 adaptive RePaint pipeline. Uses same DDPM checkpoint, same 100 eval
samples, same inference hyperparameters.

**Key differences from GP-Diff**:
1. Prior image = Voronoi fill (nearest-neighbour from observations) instead of GP posterior mean
2. Variance proxy = distance² from nearest sensor (normalised) instead of GP posterior variance
3. No GP computation at all → much cheaper at inference

**Hypothesis**: Voronoi fill preserves observed values exactly and provides a
piecewise-constant "guess" for unobserved regions. The distance-based variance
proxy has similar spatial structure to GP variance (high far from obs, low near obs).
The DDPM should smooth out Voronoi cell boundaries while adding learned structure.
