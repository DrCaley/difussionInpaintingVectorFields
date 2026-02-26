# NOTES — DPS Guided Inpainting (Div-Free Model, No Attention)

## Purpose
Test whether DPS guidance avoids the copy-paste problem that made the div-free
model fail with RePaint. Hypothesis: RePaint injects Gaussian-noised known
values into a field the model expects to be divergence-free, corrupting the
latent manifold. DPS boundary guidance never modifies x_t directly.

## Key Results (Feb 23, 2026)
5 samples, boundary=2.0, div=0.0, GP warm start t=75:

| Model | Wins vs GP | Avg Ratio |
|-------|-----------|-----------|
| Gaussian-attn DPS | 5/5 | 0.822x |
| Div-free DPS (no attn) | 2/5 | 1.061x |

The div-free model IS modifying the field (~10% of signal magnitude),
but changes are uncorrelated with GP errors:
- Gaussian model: Corr(change, GP_error) = -0.59 (strongly corrective)
- Div-free model: Corr(change, GP_error) = -0.04 (random)

## Conclusion
Architecture (no attention = can't capture long-range dependencies) is the
bottleneck, not the inference algorithm. DPS successfully avoids copy-paste,
but the model itself is too weak. → Led to training fwd_divfree_attn.
