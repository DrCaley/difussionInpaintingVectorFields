# voronoi_stage3_maskxt_hightmax

## What's being tested
Two changes to improve iterative reverse-chain inference:

1. **`mask_xt: true`** — During training, replace known region of x_t with
   independent noise (matching RePaint-style inference). Previous experiment
   had `mask_xt: false`, creating a train/inference distribution mismatch.

2. **Higher `t_max` fractions**: `[0.50, 0.40, 0.30]` → t_max = [125, 100, 75]
   (previously `[0.30, 0.20, 0.20]` → [75, 50, 50]). More noise gives the
   reverse chain more room to work and reduces dependence on source quality.

## Baseline
`voronoi_detached_stage3` (epoch 192 best EMA weights):
- Single-step 3-stage: 0.95x V-CNN (5% better)
- Full reverse 3-stage: 16.4x V-CNN (broken)

## Hypothesis
With mask_xt and higher t_max, the full reverse chain should:
- No longer be distribution-shifted from training
- Have enough noise steps for iterative refinement to work
- Potentially beat single-step mode

## Log
