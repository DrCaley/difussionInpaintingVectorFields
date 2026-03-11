# Experiment 10: Topological Similarity Metrics for Vector Field Reconstruction

## Research Question

MSE measures pixel-wise accuracy but is blind to the topological structure of
ocean velocity fields (eddies, jets, convergence zones, stagnation). Can
persistent homology on derived scalar fields provide metrics that better capture
whether a reconstruction preserves the flow topology? And if so, can those
metrics be used as training losses to produce topology-aware DDPMs?

## Approach

Compute 0-dimensional persistent homology (connected component birth/death) on
four derived scalar fields, each capturing a different family of topological
features:

| Scalar | Formula | Topological features captured |
|--------|---------|-------------------------------|
| Vorticity ω | ∂v/∂x − ∂u/∂y | Eddies (rotational extrema) |
| Divergence δ | ∂u/∂x + ∂v/∂y | Convergence/divergence zones |
| Speed \|v\| | √(u² + v²) | Jets (ridges), stagnation (minima) |
| Okubo-Weiss W | S² − ω² | Strain- vs rotation-dominated regions |

For each scalar, sublevel-set persistence is computed in both directions
(+field and −field) to capture both maxima and minima. Diagrams are compared
via Wasserstein-2 distance.

## Sub-experiments

### `eval_persistence/` — Evaluation metric only (no training change)

Run multi-scalar persistence on existing saved reconstructions from GP and
GP-Diff. Determine whether persistence Wasserstein distances discriminate
between methods better than MSE, and whether they correlate with existing
Γ₁ eddy detection results.

**Must complete before moving to `topo_aware_training/`.**

### `topo_aware_training/` — Topology-aware training loss

Train a new DDPM with auxiliary vorticity MSE and divergence MSE losses
alongside standard pixel MSE. Compare against the existing MSE-only model
on both pixel-level and topological metrics.

Loss: L = L_MSE + λ_vort · L_vort + λ_div · L_div

## Controlled Variables

- Same COAWST/ROMS Ram's Head domain, 64×128 grid (44×94 ocean)
- Same train/val split (9180 / 1965)
- Same observation mask (row 22, ~2.4% coverage)
- Same architecture (23.2M param attention U-Net)
- Same noise schedule (linear, β₁=1e-4, βT=0.02, T=250)
- Same optimizer (Adam, lr=1e-3), batch size 80

## Varied Variables

- **eval_persistence**: nothing varied — pure measurement
- **topo_aware_training**: loss function (MSE vs MSE + vort + div),
  loss weights (λ_vort, λ_div)

## File Structure

```
experiments/10_topology_metrics/
├── README.md                        # This file
├── persistence_metrics.py           # Evaluation: multi-scalar persistence + Wasserstein
├── topology_loss.py                 # Training: differentiable auxiliary losses
├── sanity_check.py                  # Verify gudhi + loss gradients before real runs
├── eval_persistence/                # Sub-experiment A: evaluation only
│   ├── NOTES.md                     # Lab notebook
│   └── results/                     # (auto-created by scripts)
└── topo_aware_training/             # Sub-experiment B: modified training
    ├── config.yaml                  # Override config for experiment launcher
    ├── NOTES.md                     # Lab notebook
    └── results/                     # (auto-created by training)
```

## Dependencies

```
pip install gudhi
```
