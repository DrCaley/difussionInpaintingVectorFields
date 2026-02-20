# RePaint + CG Div-Free Projection — Experiment Notes

## 2026-02-19 — Initial training (bad beta schedule)

- Trained unconditional eps-prediction DDPM with `forward_diff_div_free` noise
- Beta schedule: `min_beta=0.0004, max_beta=0.08` (inherited from base template)
- Training converged: best test loss 0.0023 at epoch 172
- Inference results showed severe magnitude blow-up:
  - RePaint (no projection): median magnitude 1.45 vs GT 0.19 (7.5×)
  - RePaint+CG: median magnitude 0.31 vs GT 0.19 (1.6×)
  - GP baseline: MSE 0.0007 (much better)

### Diagnosis

- CG projection is NOT the culprit — energy_ratio = 1.0000 on div-free noise
- Unconditional generation (no inpainting at all) also produces 2.32× too-large samples
- Root cause: beta schedule too aggressive. ᾱ₂₄₉ = 0.000033 (should be ~0.08)
- By t=166, ᾱ < 0.01 — all signal destroyed. Last 84/250 steps are noise→noise

### Diagnostic scripts

- `scripts/diagnose_repaint_magnitudes.py` — 5-test suite
- `scripts/diagnose_uncond_generation.py` — unconditional generation sanity check

## 2026-02-19 — Retraining with corrected beta schedule

- Fixed `base_inpaint.yaml`: `min_beta=0.0001, max_beta=0.02`
- Killed old training, relaunched
- Early progress: loss dropped from 0.267 → 0.022 in ~200 epochs
- Best test loss so far: 0.0022 at epoch 203 (still improving slowly)
- Training in progress — waiting for convergence before re-running inference
