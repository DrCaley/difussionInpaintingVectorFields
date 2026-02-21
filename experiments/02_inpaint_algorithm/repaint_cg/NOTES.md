# RePaint + CG Div-Free Projection — Experiment Notes

## Architecture

| Property | Value |
|----------|-------|
| UNet | `standard` — unconditional (2ch input: u, v only) |
| Prediction target | eps |
| Noise strategy | `forward_diff_div_free` |
| Noise steps | 250 |
| Conditioning | NONE — model never sees mask or known values |
| `mask_xt` | false |
| Inpainting algorithm | `repaint_standard()` with per-step CG div-free projection |
| Standardizer | `zscore_unified` (auto-resolved) |
| Beta schedule | `min_beta=0.0001, max_beta=0.02` (corrected) |

**Note:** This is an **unconditional** model. Inpainting is done via RePaint
(copy-paste at each step). The model only ever sees 2-channel (u, v) inputs.
CG projection enforces the divergence-free constraint at each denoising step.

---

## 2026-02-19 — Initial training (bad beta schedule)

- Trained unconditional eps-prediction DDPM with `forward_diff_div_free` noise
- Beta schedule: `min_beta=0.0004, max_beta=0.08` (inherited from old base template)
- Training converged: best test loss 0.0023 at epoch 203 (of 206)
- Inference results showed severe magnitude blow-up:

| Method | Median Magnitude | GT Magnitude | MSE |
|--------|-----------------|--------------|-----|
| RePaint (no projection) | 1.45 | 0.19 | 34.48 |
| RePaint+CG | 0.31 | 0.19 | 3.20 |
| GP baseline | ~0.19 | 0.19 | 0.0007 |

CG projection helped (1.45→0.31) but magnitudes were still 1.6× too large.

### Diagnosis (5-test diagnostic suite)

Created `scripts/diagnose_repaint_magnitudes.py` with five targeted tests:

1. **CG on pure div-free noise** → energy_ratio = 1.0000 at all timesteps.
   CG is a perfect no-op on div-free noise — NOT the culprit.
2. **CG on x_t (signal+noise)** → 2–12% energy loss (removes irrotational
   component of signal). Expected and acceptable.
3. **Plain RePaint step trace** → x₀_pred RMS 25–37× too large; final
   magnitude 7.5× GT.
4. **RePaint+CG step trace** → final magnitude 1.6× GT. CG helps but
   doesn't fix root cause.
5. **Repeated project→re-stamp cycle** → stable convergence. No runaway growth.

### Key finding: unconditional generation also broken

`scripts/diagnose_uncond_generation.py` tested the model with NO inpainting
(pure reverse diffusion from noise). Result: generated samples were **2.32×
too large**. This proved the problem is in the model/schedule, not in RePaint.

### Root cause: overly aggressive beta schedule

| Schedule | min_beta | max_beta | ᾱ₂₄₉ |
|----------|----------|----------|-------|
| Old (broken) | 0.0004 | 0.08 | 0.000033 |
| Corrected | 0.0001 | 0.02 | 0.0797 |

The old schedule destroyed all signal by t≈166 (ᾱ < 0.01). The last 84/250
steps were pure noise→noise denoising. The near-zero ᾱ_T amplified prediction
errors in the eps→x₀ conversion.

### Diagnostic scripts

- `scripts/diagnose_repaint_magnitudes.py` — 5-test magnitude diagnostic
- `scripts/diagnose_uncond_generation.py` — unconditional generation sanity check

---

## 2026-02-19 — Retraining with corrected beta schedule

- Fixed `base_inpaint.yaml`: `min_beta=0.0001, max_beta=0.02`
- Killed old training, relaunched
- Training log: `results/training_log_Feb19_1648.csv` (357 epochs)
- Best test loss: 0.003686 at epoch 348 (converged but slightly higher than
  the old model's 0.0023 — expected since old model had more aggressive noise
  which may have been easier to denoise in the low-noise regime)

---

## Lessons learned

1. **CG projection is well-behaved:** energy_ratio = 1.0 on div-free noise;
   it only removes irrotational components.
2. **Beta schedule must preserve residual signal:** ᾱ_T should be > ~0.05.
3. **Test unconditional generation first:** If pure generation is broken,
   no inpainting algorithm can fix it.
4. **Don't blame training duration when loss has plateaued.** The old model
   was fully converged at 0.0023 — the problem was the schedule, not training.

## Status

**Retrained** with corrected beta schedule (357 epochs, best test loss 0.003686).
Inference with corrected model not yet run. This experiment shares the same
spectral gap in `forward_diff_div_free` noise identified in the fwd_divfree
experiment — the equalized noise variant (`fwd_divfree_equalized`) may perform
better for this reason.
