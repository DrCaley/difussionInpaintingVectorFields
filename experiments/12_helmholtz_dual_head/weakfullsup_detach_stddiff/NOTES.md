# weakfullsup_detach_stddiff — Standard Diffusion Variant

## Motivation
Clone of `weakfullsup_detach` with `voronoi_forward: false`.
The original noises Voronoi fill during training, but multi-step inference
re-noises model predictions (clean-like), creating a distribution mismatch.
This variant noises the GT instead, so train and inference distributions match.

## Changes from weakfullsup_detach
- `voronoi_forward: false` (was `true`)
- `model_name: weakfullsup_detach_stddiff`
- Everything else identical

## Training Log
