# FiLM Helmholtz Split — No mask_xt

## Hypothesis
The FiLM model with `mask_xt: true` (v4 on Server 1) performs 3.5x worse than
the unconditional weakfullsup model at inference. We hypothesize that `mask_xt`
destroys the strongest signal source: at 0.5% coverage, 19 clean observation
pixels in x_t carry more spatial information than the crude Voronoi-fill
conditioning fed through the FiLM bottleneck.

## What changed vs helmholtz_film_fullsup
- `mask_xt: false` — known region stays visible in x_t (not replaced with noise)
- `lambda_decomp: 0.1` (was 0.5) — match weakfullsup penalty strength
- `lambda_orth: 0.01` (was 0.1) — match weakfullsup penalty strength
- Loss is now full-field (when mask_xt=false, the else branch computes full MSE)

## Training Log
