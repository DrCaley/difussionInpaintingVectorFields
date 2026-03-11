# Topology-Aware Training — Lab Notebook

## Goal

Train DDPM with auxiliary vorticity/divergence MSE losses so the model
learns to preserve differential structure, not just pixel-level accuracy.

## Status: Training v3 in progress on vast.ai (stable through epoch 21)

Baseline persistence metrics done:
- GP: W_total=0.652  |  GP-Diff: W_total=0.508  |  V-CNN: W_total=0.385
- V-CNN beats GP-Diff on all topology metrics → room to improve

---

## Design Decisions

- **eps-parameterization + x̂₀ reconstruction**: User requested unconditional
  multi-step diffusion (standard_attn UNet, eps prediction). We reconstruct
  x̂₀ = (x_t − √(1−ᾱ_t)·ε̂) / √ᾱ_t inside the loss and compute topology
  terms on the reconstruction. This is gated to low timesteps where ᾱ_t is
  large enough for the reconstruction to be meaningful.
- **t_max_frac = 0.25**: Only apply auxiliary loss at low noise levels
  (t/T < 0.25). At high noise the signal is buried and differential operators
  amplify noise, making the aux loss unreliable.
- **Starting weights**: λ_vort=0.1, λ_div=0.05, λ_speed=0.0. Conservative;
  increase if eval shows topology improvements without hurting pixel MSE.
- **Ocean mask baked in**: TopologyAwareLossStrategy has a registered buffer
  for the 44×94 ocean mask — no need to pass it through the data pipeline.
- **Tier 3 future**: Once Tier 2 works, consider adding Wasserstein persistence
  loss (expensive, needs gudhi in training loop, possibly every N batches only).

## Integration Steps

1. [x] Register `TopologyAwareLossStrategy` in `LOSS_REGISTRY` in
       `ddpm/helper_functions/loss_functions.py`
2. [x] Ocean mask built into loss (register_buffer) — no pipeline changes needed
3. [x] Pass timestep `t`, `x0`, `ddpm`, `prediction_target` via **kwargs
       from `train_inpaint.py` → loss strategy
4. [x] Update `DDInitializer._setup_loss_strategy()` to pass topology_loss config
5. [x] Config updated: standard_attn + eps + gaussian + topology_aware
6. [x] Smoke test: 1 epoch (batch_size=16, MPS), gradients flow, loss=0.056
7. [ ] Full training on GPU server
8. [ ] Evaluate trained model with persistence_metrics.py

## Entries

*(Add dated entries below as work progresses)*

### 2025-03-04 — Implementation complete + smoke test

**Done:**
- `TopologyAwareLossStrategy` added to `ddpm/helper_functions/loss_functions.py`
  with `register_buffer` for ocean mask and central-difference kernels
- Training loop passes `x0`, `t`, `ddpm`, `prediction_target` via `**kwargs`
- `DDInitializer._setup_loss_strategy()` forwards `topology_loss:` config block
- Config: `standard_attn` + `eps` prediction + `gaussian` noise + `topology_aware` loss
- Unit test confirms: low-t topology loss active (adds ~0.31), high-t gated off,
  fallback to pure MSE when kwargs missing
- Smoke test (1 epoch, batch_size=16, MPS): epoch_loss=0.056, test=0.026
- Fixed MPS device mismatch: `self.loss_strategy = dd.get_loss_strategy().to(self.device)`

**Next:** Full training on GPU server (1000 epochs, batch_size=80)

### 2025-03-05 — Training v1 collapsed at epoch 7 (lr too high)

**Problem:** Launched full training on vast.ai (RTX 3060, ssh -p 58777
root@1.208.108.242). Loss decreased normally through epoch 5 (test=0.018)
then spiked at epoch 7 → train/test ≈ 1.0 (predicting constant zero).

**Root cause diagnosed as wrong training recipe:**
The real issue was NOT the topology loss — it was using `lr=0.001` (default)
for the `standard_attn` UNet (23.2M params). Experiment 08 proved this
architecture requires `lr=3e-4` with cosine schedule + EMA to be stable.

**v1 config (collapsed):** lr=0.001, constant, no EMA, no weight_decay
**v2 config (also collapsed at epoch 5):** Added warmup/clamp for topo loss
but kept lr=0.001 — proved the collapse was NOT from topology terms since
it happened during MSE-only warmup phase.

### 2025-03-05 — Training v3 launched with correct recipe

**Fix:** Adopted proven training recipe from `exp08/repaint_gaussian_attn`:
- `lr: 0.0003` (3e-4, down from 1e-3)
- `lr_schedule: cosine` (with 10-epoch warmup)
- `use_ema: true` (decay=0.9999)
- `weight_decay: 0.0001` (AdamW regularization)
- `max_grad_norm: 1.0` (gradient clipping)
Plus topology loss stability features from v2:
- `warmup_epochs: 10` (MSE-only for first 10 epochs)
- `x0_clamp: 6.0` (clamp reconstructed x̂₀)
- `t_max_frac: 0.1` (only apply topo at very low noise)
- `lambda_vort: 0.01`, `lambda_div: 0.005` (conservative weights)

**Results through epoch 21 — STABLE:**

| Epoch | test | ema_test | lr | Notes |
|-------|------|----------|-----|-------|
| 1 | 0.836 | 1.061 | 3e-5 | LR warmup, barely learning |
| 5 | 0.023 | 0.721 | 1.5e-4 | **No collapse** (v1/v2 died here) |
| 10 | 0.018 | 0.257 | 3e-4 | Topo terms activate — smooth |
| 15 | 0.016 | 0.102 | 3e-4 | EMA catching up |
| 18 | 0.016 | 0.069 | 3e-4 | Best test so far |
| 21 | 0.016 | 0.051 | 3e-4 | Still improving |

**Key insight:** The `standard_attn` UNet collapse was entirely a training
recipe issue, not related to topology loss. The proven exp08 recipe (lr=3e-4,
cosine, EMA) prevents the instability that occurred at lr=1e-3.

**Next:** Monitor through ~epoch 50+ (EMA convergence), then evaluate with
persistence_metrics.py vs baselines.

### 2025-03-06 — Training stopped at epoch 184, inference comparison run

**Training:** Ran to epoch 184/1000. Best EMA checkpoint at epoch 69
(test_loss=0.0143). Loss had clearly plateaued by ~epoch 100.

**Inference: Topo-aware (EMA) vs Baseline (exp08) — 10 samples, ~95% missing, RePaint resample=5**

Script: `tmp_topo_inference.py` — loads both models (same architecture:
standard_attn/MyUNet_Attn, eps-prediction, T=250, gaussian noise), runs
`repaint_standard()` on identical validation samples & masks.

| Metric | Baseline (exp08) | Topo-aware (EMA) | Change | Winner |
|--------|------------------|------------------|--------|--------|
| MSE | 0.4113 | 0.4071 | −1.0% | TOPO |
| MAE | 0.3164 | 0.3197 | +1.0% | BASE |
| Mean \|div\| | 0.003335 | 0.003539 | +6.1% | BASE |
| Vorticity MSE | 0.000261 | 0.000254 | −2.8% | TOPO |

Per-sample wins (topo/total): MSE=6/10, |div|=2/10, vort=4/10

**Observations:**
- Results are essentially **within noise** — no statistically significant
  difference on 10 samples at this mask coverage. The topo model neither
  clearly helps nor hurts.
- MSE is marginally better (−1%), vorticity MSE marginally better (−2.8%),
  but divergence is slightly worse (+6.1%) and MAE is a wash.
- The topology loss terms (λ_vort=0.01, λ_div=0.005) at t_max_frac=0.1
  may be too conservative to produce a detectable effect — they only act
  on the lowest 10% of timesteps.
- Training only reached epoch 184 (best at 69), vs exp08 which fully
  converged over 1000 epochs. However, the loss plateau suggests further
  training is unlikely to change the outcome significantly.
- Both models produce mean |div| ≈ 0.003, well below the GT |div| ≈ 0.009.
  The inpainting process inherently smooths, reducing divergence regardless
  of the loss.

**Next steps (potential):**
- Run with more samples (50–100) for statistical significance
- Try stronger topology weights (λ_vort=0.1, λ_div=0.05)
- Try larger t_max_frac (0.25 or 0.5) so topo loss sees more timesteps
- Run persistence diagram comparison (Wasserstein distance) to see if
  topological features differ despite similar pixel-level metrics
- Consider: the unconditional model's inference is dominated by the RePaint
  algorithm, which overwrites known regions regardless of what the model
  learned — the topology loss may only help in the unknown region's structure
