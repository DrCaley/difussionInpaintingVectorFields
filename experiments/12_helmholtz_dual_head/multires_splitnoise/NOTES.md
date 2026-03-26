# multires_splitnoise — Experiment Notes

## Hypothesis
Combining the multires FiLM architecture (`helmholtz_split_film_multires`)
with dual independent noise schedules (solenoidal normal-speed, irrotational
2× faster) will improve head specialization. The irrotational component is
typically lower-energy and smoother; destroying it faster forces the φ-head
to denoise a harder signal at each timestep, sharpening the decomposition.

## What's Varied vs multires_maskxt
- `helmholtz_split_noise: true` (was false)
- `irr_speed: 2.0` (new)
- `split_head_loss: true` (new) — primary loss is per-head MSE against
  Helmholtz-decomposed GT: `MSE(v_sol, GT_sol) + MSE(v_irr, GT_irr)`.
  No combined MSE on `v_sol + v_irr`; each head is evaluated independently.
- `noise_function: gaussian` (was `helmholtz_matched` — split schedule
  handles subspace projection internally via FFT Helmholtz decomposition)

## What's Held Constant
- UNet: `helmholtz_split_film_multires`
- `mask_xt: true`
- `voronoi_forward: false`
- `helmholtz_loss: {lambda_decomp: 0.1, lambda_orth: 0.01}`
- All optimizer/scheduler settings identical to multires_maskxt

## Log

### 2026-03-19 — Created
- Config created, deploying to server 1 (RTX 3060, 12GB).
