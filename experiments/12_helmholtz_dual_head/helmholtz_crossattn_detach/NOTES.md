# Helmholtz Cross-Attention with Detached Heads

## Hypothesis
The cancellation pathology (φ/ψ≈2, cos≈-0.99, cancel≈13x) in all conditioned
Helmholtz-split models is caused by the base reconstruction MSE gradient
coupling the two heads. Even with per-head decomposition supervision
(λ_decomp=0.1), the reconstruction gradient overwhelms the decomp signal and
drives the heads to a cancelling solution.

## Approach
**Detach the sum**: `return v_sol.detach() + v_irr.detach()` in the UNet.
The base MSE still computes correctly (for monitoring), but its gradients
cannot reach the ψ/φ decoder branches. Only the decomposition loss
(`MSE(v_sol, GT_sol) + MSE(v_irr, GT_irr)`) and orthogonality penalty
train the heads. Since decomp loss is now the primary signal, λ_decomp
is increased from 0.1 → 1.0.

## Key insight
`decomp_loss = MSE(v_sol, GT_sol) + MSE(v_irr, GT_irr)` implicitly ensures
good reconstruction: if both heads match their GT decompositions, then
`v_sol + v_irr ≈ GT_sol + GT_irr = GT`.

## Controlled variables (same as helmholtz_film_crossattn)
- Architecture: cross-attention FiLM, split decoder
- Noise: helmholtz_matched
- Data: voronoi_forward, same known fracs
- Training: lr=0.0005, cosine, 800ep, EMA

## Varied variables
- `detach_heads: true` (was false/absent)
- `lambda_decomp: 1.0` (was 0.1)

## Log
