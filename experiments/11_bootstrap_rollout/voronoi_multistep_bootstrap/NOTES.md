# Voronoi multistep bootstrap — Experiment Notes

## 2026-03-06 — Setup

- Created bootstrap fine-tuning experiment starting from the converged Voronoi-forward topology-aware checkpoint.
- Added rollout-mixture fine-tuning with depth probabilities `{0: 0.6, 1: 0.3, 2: 0.1}`.
- Goal: reduce the train/inference mismatch seen when the Voronoi model is used in multistep diffusion.

## 2026-03-06 — Training launch

- Launching remote fine-tuning from `experiments/11_bootstrap_rollout/voronoi_multistep_bootstrap/config.yaml`.
- Base checkpoint: `experiments/10_topology_metrics/voronoi_topo_training/results/inpaint_gaussian_t250_best_checkpoint.pt`
- Fine-tune budget: 200 epochs, lr=1e-4, cosine schedule, EMA enabled.

## 2026-03-06 — S6 iterative trace

- Ran a stage-by-stage S6 trace on validation sample `482` at `0.5%` coverage using the bootstrap EMA weights.
- Saved tensor artifact: `experiments/11_bootstrap_rollout/voronoi_multistep_bootstrap/results/s6_trace/val482_cov0p5_s6_trace.pt`
- Saved rendered panels in: `experiments/11_bootstrap_rollout/voronoi_multistep_bootstrap/results/s6_trace/`
- Voronoi prior MSE was `0.001767`.
- Stage MSEs degraded monotonically across S6: `S1=0.002282`, `S2=0.003134`, `S3=0.003477`, `S4=0.003738`, `S5=0.004055`, `S6=0.006275`.
- This is consistent with the earlier benchmark result on the same sample: repeated S6 refinement pushes the bootstrap Voronoi model away from the useful one-shot correction regime instead of improving it.
