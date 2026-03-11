# Helmholtz Supervised Decomposition

**Hypothesis**: Per-head supervision (v_sol → GT_sol, v_irr → GT_irr via FFT)
plus orthogonality penalty will break the degenerate cancellation solution
observed in the baseline Helmholtz model.

**Diagnostic findings (baseline)**:
- v_sol and v_irr RMS nearly equal (~1.25 each), massive cancellation
- Model routes solenoidal content through grad(φ) head (27% leakage)
- Per-component MSE terrible (~1.4) despite good total MSE (0.056)
- GT split: 74% solenoidal, 12% irrotational

**What's new**:
- `helmholtz_supervised` loss = topology_aware + decomp supervision + orthogonality
- λ_decomp = 0.1, λ_orth = 0.01
- GT decomposition via FFT Helmholtz (same as helmholtz_split.py)

**Controlled variables** (identical to helmholtz_baseline):
- Architecture (MyUNet_Helmholtz), lr, batch, schedule, EMA, epochs
- Only change: loss function

## Log
