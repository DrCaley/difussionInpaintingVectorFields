# Experiment Group 02 — Inpainting Algorithm

## Research Question

**Does RePaint + per-step CG div-free projection produce lower divergence
at the known/unknown boundary than FiLM-conditioned x₀-prediction
inpainting?**

The standard RePaint algorithm copy-pastes forward-noised known values
into the reverse chain at every timestep.  This creates a divergence
discontinuity at the mask boundary — the denoised unknown region and
the forward-noised known region come from different distributions.

When working with divergence-free vector fields, this boundary break
is physically unacceptable.  Applying a conjugate-gradient (CG)
streamfunction projection after each paste step should eliminate
boundary divergence by construction.

## Controlled Variables (held constant)

| Variable | Value |
|----------|-------|
| Noise function | `forward_diff_div_free` |
| Noise steps | 250 |
| Beta schedule | `min_beta=0.0001`, `max_beta=0.02` |
| Dataset | rams_head |
| Standardizer | `zscore_unified` (auto) |
| Batch size | 80 |
| Learning rate | 0.001 |
| Epochs | 1000 |

> **Note (2026-02-19):** Beta schedule was corrected from `min_beta=0.0004, max_beta=0.08`
> to `min_beta=0.0001, max_beta=0.02`. The original schedule was 4× too aggressive
> (ᾱ₂₄₉ = 0.000033 vs 0.0797), destroying all signal by t≈166 and causing
> magnitude blow-up in generated samples. See AI_RESEARCH_JOURNAL.md for full diagnosis.
> Models trained before this date need retraining.

## Varied Variables

| Experiment | UNet | Prediction Target | Noise | Inpainting Method |
|------------|------|--------------------|-------|-------------------|
| `repaint_cg` | `standard` (2ch unconditional) | `eps` | `forward_diff_div_free` | `repaint_standard` with `project_div_free=True` |
| `repaint_gaussian` | `standard` (2ch unconditional) | `eps` | `gaussian` | `repaint_standard` (no projection) |
| `repaint_gaussian_attn` | `standard_attn` (2ch unconditional + self-attention) | `eps` | `gaussian` | `repaint_standard` (no projection) |

## Key Hypothesis

An unconditional eps-prediction model + RePaint + CG projection should:
1. Produce exactly div-free outputs (by construction, via CG streamfunction)
2. Maintain boundary coherence (projection smooths the paste discontinuity)
3. Trade off some MSE accuracy for physical consistency

## Inference

Use `repaint_standard()` from `ddpm/utils/inpainting_utils.py` with:
- `prediction_target="eps"`
- `project_div_free=True`
- `resample_steps=5` (standard RePaint resampling)
- Optionally `project_final_steps=3` for extra end-of-chain cleanup

## Comparison Baseline

Compare against the FiLM `x₀`-prediction model from
`experiments/01_noise_strategy/fwd_divfree/` which uses
`x0_full_reverse_inpaint()` with `project_steps`.
