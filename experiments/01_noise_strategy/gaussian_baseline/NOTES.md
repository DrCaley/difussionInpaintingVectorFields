# Gaussian Noise Baseline — Experiment Notes

## Architecture

| Property | Value |
|----------|-------|
| UNet | FiLM-conditioned (5ch input: x_t + mask + known_values) |
| Prediction target | x₀ |
| Noise strategy | `gaussian` |
| Noise steps | 250 |
| Conditioning | FiLM (mask-aware) |
| `mask_xt` | true |
| Standardizer | `zscore` (auto-resolved — per-component OK for Gaussian) |

**Note:** This experiment inherits FiLM/x₀ defaults from `base_inpaint.yaml`.
It is a **conditioned** model. Gaussian noise does not require unified z-score
because ∂u/∂x + ∂v/∂y = 0 is not a constraint for Gaussian noise.

---

## 2026-02-17 — Config created

- Baseline comparison for noise strategy experiments
- Config validated via dry-run
- Not yet trained as a standalone experiment within the experiment framework

### Prior Gaussian models (trained outside experiment framework)

Several Gaussian-trained models pre-date the experiment framework:

- `weekend_ddpm_ocean_model.pt` — n_steps=100, Gaussian noise. Used in early
  div-free vs Gaussian comparisons (Jan 5, 2026). This model was the primary
  reference during early development.
- `ddpm_ocean_model_gaussian_t250.pt` — n_steps=250, Gaussian noise, unconditional
  eps-prediction. Trained as part of `02_inpaint_algorithm/repaint_gaussian/`.

### Historical comparison data (Jan 5, 2026)

The "weekend model" (Gaussian, n_steps=100) was compared against div-free models:

| Configuration | MSE | Angular Error | Mag Ratio |
|---------------|-----|---------------|-----------|
| Gaussian model + Gaussian noise | **0.0106** | 77.7° | **1.55×** |
| Gaussian model + Div-free noise | 0.0320 | 139.7° | 2.58× |
| Div-free model + Div-free noise | 0.0818 | 89.4° | 5.30× |

Gaussian noise consistently outperformed div-free on most metrics, though the div-free
model's poor performance was later traced to (a) aggressive beta schedule, (b) spectral
gap in div-free noise, and (c) boundary discontinuity in RePaint stitching.

## Status

**Not trained** as a standalone experiment. Existing Gaussian models serve as
the baseline. This config exists for completeness and potential future retraining
with the experiment framework.
