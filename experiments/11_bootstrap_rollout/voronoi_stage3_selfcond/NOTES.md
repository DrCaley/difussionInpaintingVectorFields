# voronoi_stage3_selfcond — Experiment Notes

## What's Being Tested
Self-conditioning: during training, with 50% probability, run the UNet twice —
first with `self_cond=None` to get a preliminary x0 prediction, then with
that prediction fed back as additional input. This teaches the model to
refine its own outputs, which directly benefits the multi-stage reverse chain.

Combined with `mask_xt: true` and higher `t_max` fractions from the
maskxt_hightmax experiment.

## Controlled Variables
- Same base weights (voronoi_detached_stage3 ep 192)
- Same 3-stage detached rollout
- Same topology-aware loss
- Same training hyperparameters as maskxt_hightmax

## Varied vs maskxt_hightmax
- `self_conditioning: true` / `p_self_cond: 0.5` (new self_cond_proj layer in UNet)

---
