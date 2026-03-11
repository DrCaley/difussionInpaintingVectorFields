# 06 — GP-Forward Training

## Research Question

Can we close the training–inference distribution gap by noising the GP
posterior (instead of ground truth) during training, so the model learns
to refine GP reconstruction errors rather than generic Gaussian noise?

## Background

The FiLM+Attn model (experiment 03) was trained in the standard DDPM way:
`x_t = √ᾱ_t · GT + √(1-ᾱ_t) · ε`, then predict GT from x_t.  At
inference, however, we noise the *GP posterior mean* (not GT) and ask the
model to recover GT.  The GP posterior has structured, spatially-correlated
errors that look nothing like Gaussian noise.  Result: the model achieves
only 1–2% MSE improvement over the GP baseline (vs Voronoi-CNN's 73%).

## Approach

**GP-forward training**: during training, replace the standard forward
process with:

    x_t = √ᾱ_t · GP(GT, mask) + √(1-ᾱ_t) · ε

The target remains GT (x0-prediction).  The model learns:
"given a noised GP field + mask + known observations → recover ground truth"

This directly matches the inference distribution.

## Controlled Variables
- Architecture: MyUNet_FiLM_Attn (same as 03/film_attn_divfree)
- Prediction target: x0
- Noise: forward_diff_div_free, T=250
- mask_xt: true
- Training masks: same distribution as 03

## Varied Variables
- `gp_forward: true` (the key change)
- Fine-tuned from existing epoch-161 weights (not from scratch)
- Reduced epoch count (model already knows ocean velocity structure)

## Expected Outcome
The model should learn to correct structured GP errors, producing
reconstructions closer to GT than either pure GP or the original
DDPM.  Target: match or beat Voronoi-CNN (MSE ratio ~0.27× vs GP).
