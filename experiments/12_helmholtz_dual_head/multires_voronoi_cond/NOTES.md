# Multires Voronoi Conditioning

## What's being tested
Whether replacing the sparse observation conditioning (95%+ zeros) with dense
Voronoi (nearest-neighbour) fill in the FiLM encoder improves performance.

## Hypothesis
The MultiResCondEncoder's `_pool_sparse` produces 95% zeros at every resolution
level, giving FiLM layers almost no signal.  V-CNN outperforms because it uses
dense Voronoi fill as input.  Providing the same dense fill as FiLM conditioning
should dramatically improve the DDPM's ability to use observation information.

## Changed vs baseline (multires_splitnoise_subframes)
- `use_voronoi_fill: true` — Voronoi fill replaces sparse obs in conditioning
- Everything else identical: same arch, loss, Helmholtz split noise, etc.
