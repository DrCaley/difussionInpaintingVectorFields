# Helmholtz Baseline — Experiment Notes

## 2026-03-07 — Setup

- Created `MyUNet_Helmholtz` in `ddpm/neural_networks/unets/unet_helmholtz.py`
- Architecture: shares full encoder+bottleneck+decoder with `MyUNet_Attn`
  (~18M params), two lightweight output heads (ψ for streamfunction, φ for
  velocity potential) that map to velocity via physics operators
- ψ head: predicts scalar ψ on H×W grid, zero-pads to (H+1)×(W+1), applies
  forward-diff curl → exactly divergence-free solenoidal velocity
- φ head: predicts scalar φ on H×W grid, applies central-diff gradient
  → exactly curl-free irrotational velocity
- φ head initialised to zero (model starts solenoidal-biased, learns to add
  divergent component as needed)
- v_total = curl(ψ) + grad(φ)
- Training from scratch (no init_from_weights) since architecture differs
- Voronoi warm-start, single-stage, topology-aware loss
