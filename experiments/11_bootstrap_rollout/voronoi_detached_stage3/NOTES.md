# Notes — voronoi_detached_stage3

## 2026-03-06

- Added first implementation of 3-stage detached rollout training.
- Uses the unconditional topology-aware `standard_attn` backbone with an added learned stage token.
- Training loop now supports:
  - repeated outer stages with detached handoff,
  - per-stage timestep caps,
  - per-stage loss weights,
  - observation repaste between stages,
  - initialization from the converged Voronoi-forward weights.
- Default experiment setup:
  - stage weights: `[1.0, 0.8, 0.6]`
  - timestep caps: `[0.30, 0.20, 0.20] × T`
  - `init_from_weights` points to the converged Voronoi-topology model.
- Validation:
  - `experiments/run_experiment.py --dry-run` passed and wrote `results/resolved_config.yaml`.
  - Trainer construction from the resolved config succeeded.
  - Stage-aware warm start loaded with `strict=False`; only missing key was the new `network.stage_embed.weight`.
  - One detached-rollout batch forward/backward pass succeeded on MPS (`loss ≈ 0.0795`, batch size 16).
- Remote smoke test:
  - Synced the experiment to the remote GPU server and ran `experiments/run_experiment.py --smoke experiments/11_bootstrap_rollout/voronoi_detached_stage3/config.yaml`.
  - Smoke training completed successfully on CUDA for all 3 epochs with no runtime errors or OOM.
  - Best raw validation loss occurred at epoch 2: `test = 0.0439729`.
  - EMA validation was best at epoch 1: `ema_test = 0.0417973`.
  - Training artifacts were written under `results/`, including the remote loss plot.
- Next checks:
  1. run a short smoke training job,
  2. confirm stage-aware weights save/load cleanly after checkpoint write,
  3. add matching stage-aware inference path before full comparison runs.
