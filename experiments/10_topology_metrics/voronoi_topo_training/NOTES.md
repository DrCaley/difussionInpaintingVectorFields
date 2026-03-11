# Voronoi-Forward Topology-Aware Training — Experiment Log

## Motivation

Voronoi warmstart at inference time showed dramatic improvement over GP warmstart:
- Vor-Diff(topo) mean MSE = 0.00195 vs GP-Diff(topo) = 0.01350 at 0.5% coverage
- 6.91× better, wins 10/10 samples
- Gap to V-CNN narrowed from 15.79× (GP-Diff) to 2.29× (Vor-Diff)

However, the model was still trained on clean GT with eps-prediction — the
training distribution doesn't match the Voronoi-warmstart inference distribution.
By training with Voronoi-forward (noise the Voronoi fill instead of GT, predict x0),
we should close this distribution gap and further improve results.

## Design

- **voronoi_forward**: Forward diffusion noises Voronoi fill, target is clean x0
- **prediction_target: x0** (required — eps would recover Voronoi, not GT)
- **Random masks each sample**: coverage fracs [0.1%, 0.5%, 1%, 2%, 5%]
- **On-the-fly Voronoi**: cKDTree NN fill, ~1ms per sample, no precomputation
- **Same architecture/loss as topo_aware_training**: standard_attn, topology_aware loss
- **Added augment: true** for velocity-aware flips (more diversity helps with random masks)

## Training Log

(entries below)
