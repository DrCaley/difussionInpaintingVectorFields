# Gaussian RePaint Baseline — Experiment Notes

## 2026-02-18 — Training completed

- Unconditional eps-prediction DDPM with standard Gaussian noise
- Beta schedule: `min_beta=0.0001, max_beta=0.02`
- Model checkpoint: `results/ddpm_ocean_model_gaussian_t250.pt`
- This is the vanilla RePaint baseline — no div-free projection

## 2026-02-18 — Inference runs

- Ran bulk evaluation: `results/repaint_gaussian_bulk.csv`
- Visualization runs: `results/repaint_gaussian_run14.png`, `results/repaint_gaussian_run19.png`
- Comparison plot: `results/repaint_gaussian_comparison.png`
- Magnitude ratio ~2.1× (overshoots without projection)
