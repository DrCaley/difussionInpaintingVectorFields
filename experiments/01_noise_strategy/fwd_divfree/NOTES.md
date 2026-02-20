# Forward-Diff Div-Free Noise (FiLM x₀-prediction) — Experiment Notes

## 2026-02-17 — Training

- FiLM-conditioned UNet (5ch input, 9.8M params), x₀ prediction
- `forward_diff_div_free` noise, 250 steps
- **Note:** Trained with old beta schedule (`min_beta=0.0004, max_beta=0.08`)
  which was later found to be too aggressive. May need retraining with
  corrected schedule (`min_beta=0.0001, max_beta=0.02`).
- Best test loss: 0.0441
- Model used for early inference testing with `x0_full_reverse_inpaint()`

## 2026-02-19 — Beta schedule issue identified

- The aggressive beta schedule (ᾱ₂₄₉ = 0.000033) was identified as root cause
  of magnitude blow-up in the repaint_cg experiment
- This model may also be affected, though x₀ prediction is potentially less
  sensitive to schedule issues than eps prediction
- Retraining with corrected schedule is recommended
