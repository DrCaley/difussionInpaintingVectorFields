# helmholtz_matched_fullsup — NOTES

## What's being tested
Matched noise + **full-time decomposition supervision** (decomp_t_max_frac=1.0).

Previous matched noise model had decomp supervision gated to t < 25 (10% of
training). The FFT Helmholtz decomposition of GT x₀ is always clean regardless
of noise level, so there's no reason to gate it. This experiment applies
per-head decomp loss at ALL timesteps.

## Key changes vs helmholtz_split_matched_noise
- `decomp_t_max_frac: 1.0` (was 0.1 by default — only 10% of steps)
- `lambda_decomp: 0.5` (was 0.1 — stronger signal)
- `lambda_orth: 0.1` (was 0.01 — stronger anti-cancellation)

## Controlled variables
- Same architecture (MyUNet_Helmholtz_Split, 24.49M params)
- Same noise (helmholtz_matched)
- Same standardizer (zscore_unified, auto-detected)
- Same LR schedule, batch size, epochs

## Observations

### 2026-03-10 — Smoke test + full training launch
- Smoke test (3 epochs) completed successfully on Server 1 (RTX 5070 Ti 16GB).
  - Epoch 1: test=0.5660, Epoch 2: test=0.1383, Epoch 3: test=0.1035
  - `decomp_t_max_frac=1.0` working — no errors with full-time decomp supervision.
  - ~109s/epoch with batch_size=16.
- Full 400-epoch training launched on Server 1.
  - Log: `/workspace/train_fullsup.log`
  - First 2 epochs confirm smooth training (test loss dropping normally).
- **Key question**: Will full-time decomp supervision keep heads physically
  meaningful (cos(ψ-head, GT_sol) >> 0.17) while maintaining reconstruction
  quality (target: < 1.0x V-CNN)?
