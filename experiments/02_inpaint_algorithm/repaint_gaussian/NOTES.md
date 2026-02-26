# Gaussian RePaint Baseline — Experiment Notes

## Architecture

| Property | Value |
|----------|-------|
| UNet | `standard` — unconditional (2ch input: u, v only) |
| Prediction target | eps |
| Noise strategy | `gaussian` |
| Noise steps | 250 |
| Conditioning | NONE — model never sees mask or known values |
| `mask_xt` | false |
| Inpainting algorithm | `repaint_standard()` — no div-free projection |
| Standardizer | `zscore` (per-component, fine for Gaussian) |
| Beta schedule | `min_beta=0.0001, max_beta=0.02` |

**Note:** This is an **unconditional** model. Inpainting is done via RePaint
(copy-paste at each step). The model only ever sees 2-channel (u, v) inputs.
This is the **vanilla RePaint baseline** for comparing inpainting algorithms.

---

## 2026-02-18 — Training completed

- Unconditional eps-prediction DDPM with standard Gaussian noise
- Beta schedule: `min_beta=0.0001, max_beta=0.02`
- Model checkpoint: `results/ddpm_ocean_model_gaussian_t250.pt`

## 2026-02-18 — Inference runs

- Bulk evaluation: `results/repaint_gaussian_bulk.csv`
- Visualization runs: `results/repaint_gaussian_run*.png`
- Comparison plot: `results/repaint_gaussian_comparison.png`
- **Magnitude ratio ~2.1×** — overshoots without div-free projection.
  This is a known issue with RePaint: the copy-paste stitching creates
  boundary artefacts that compound over denoising steps. Without projection
  to constrain magnitudes, the result overshoots.

### Boundary divergence analysis (Jan 8, 2026)

Boundary stitching creates measurable divergence at the mask boundary:

| Method | Bnd/Away Div Ratio |
|--------|-------------------|
| Gaussian naive stitch | 1.95× |
| Gaussian + Per-Step CombNet | 1.20× |
| Div-free naive stitch | 2.93× |
| Div-free + Per-Step CombNet | 1.65× |

Gaussian has inherently better boundary behavior than div-free because
Gaussian noise is spatially uncorrelated — splicing two independently
denoised regions creates no spatial correlation conflict.

## Status

**Complete.** This experiment serves as the baseline for comparing inpainting
algorithms. Key finding: Gaussian RePaint produces 2.1× magnitude overshoot
without projection, but has better boundary behavior than div-free approaches.
