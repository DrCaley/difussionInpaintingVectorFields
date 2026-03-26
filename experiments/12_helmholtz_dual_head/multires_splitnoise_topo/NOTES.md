# multires_splitnoise_topo — Experiment Notes

## Hypothesis
Adding topology penalties (vorticity λ=0.01, divergence λ=0.005) on the
combined prediction (v_sol + v_irr) will improve physical consistency of
inpainted fields without hurting per-head decomposition quality.

## What changed
- **Code change**: `ddpm/training/train_inpaint.py` `_prediction_loss()` —
  the `split_head_loss=True` path previously early-returned with raw per-head
  MSE, completely bypassing topology penalties. Now it computes vorticity and
  divergence penalties on `v_combined = v_sol + v_irr`, gated by warmup_epochs
  and t_max_frac (only low-noise timesteps, after warmup).
- **Config**: Identical to `multires_splitnoise` except model_name and
  explicit topology section (lambda_vort=0.01, lambda_div=0.005).

## Baseline comparison
- `multires_splitnoise`: split_head_loss WITHOUT topology penalties
- V-CNN: MSE=0.000719 on 20 GTs × 3 masks at 0.5% coverage

## Training log

### 2025-03-23 — Launched training
- Server: RTX 5060 Ti (server 2), PID 2157
- 800 epochs, loss converging (~0.029 at epoch 274)

### 2025-03-24 — Intermediate eval at epoch 274 (best@241)

Ran `scripts/eval_helmholtz_split.py` with 5 samples at 4 coverages.
Compared against multires_splitnoise (non-topo, fully trained on prior server).

**MSE — best method per coverage:**

| Coverage | Non-topo best    | MSE     | vs V-CNN | Topo best     | MSE     | vs V-CNN | Ratio    |
|----------|-----------------|---------|----------|---------------|---------|----------|----------|
| 0.5%     | 1S-DivFree      | 0.00396 | 6.86x    | 1S-DivFree    | 0.01010 | 17.49x   | 2.55x ↓  |
| 1.0%     | Ens10           | 0.00128 | 3.21x    | Ens10         | 0.00532 | 13.30x   | 4.15x ↓  |
| 2.0%     | Ens10           | 0.00071 | 2.93x    | 1S-DivFree    | 0.00462 | 19.00x   | 6.48x ↓  |
| 5.0%     | Ens10           | 0.00047 | 3.33x    | 1S-DivFree    | 0.00375 | 26.65x   | 7.98x ↓  |

**Per-head alignment** (cosine similarities nearly identical):
- cos(ψ-head, GT_sol): topo 0.854 vs non-topo 0.865
- cos(φ-head, GT_irr): topo 0.713 vs non-topo 0.702
- Head MSE much worse: sol 0.649 vs 0.169, irr 0.156 vs 0.061

**⚠️ CAVEAT**: Topo is only 1/3 through training (best@epoch 241/800).
Non-topo was fully trained. Not a fair comparison yet.
Re-evaluate after topo training completes.
