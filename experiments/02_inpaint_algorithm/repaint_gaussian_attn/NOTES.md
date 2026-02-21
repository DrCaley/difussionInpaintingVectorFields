# Gaussian RePaint + Attention UNet — Experiment Notes

## Architecture

| Property | Value |
|----------|-------|
| UNet | `standard_attn` — unconditional with attention (2ch input: u, v only) |
| Prediction target | eps |
| Noise strategy | `gaussian` |
| Noise steps | 250 |
| Conditioning | NONE — model never sees mask or known values |
| `mask_xt` | false |
| Inpainting algorithm | `repaint_standard()` — no div-free projection |
| Standardizer | `zscore` (per-component, fine for Gaussian) |
| Beta schedule | `min_beta=0.0001, max_beta=0.02` |
| Batch size | 16 (reduced from default 80 due to larger model) |
| Learning rate | 0.0003 (reduced from default 0.001) |
| Gradient clipping | `max_grad_norm=1.0` |

**UNet enhancements over `standard`:**
- Residual blocks (skip connections within each block)
- Multi-head self-attention at 16×32 and 8×16 resolutions
- GroupNorm + AdaGN time conditioning

**Note:** This is an **unconditional** model. Inpainting is done via RePaint
(copy-paste at each step). The model only ever sees 2-channel (u, v) inputs.
Tests whether long-range spatial awareness (via attention) improves RePaint
inpainting quality compared to the vanilla `repaint_gaussian` baseline.

---

## 2026-02-20 — Local training

### Run 1 (Feb20 07:22): 86 epochs

- Training log: `results/training_log_Feb20_0722.csv`
- Loss curve:
  - Epoch 1: test loss 0.0183
  - Epoch 4: test loss 0.0125
  - Epoch 43: test loss 0.0094
  - Epoch 57: test loss **0.009294** (best)
  - Epoch 86: test loss 0.009528 (plateau reached)
- Model plateaued around test loss ~0.0093–0.0095 from epoch 57 onward
- Checkpoint: `results/inpaint_gaussian_t250_Feb20_0722.pt`

### Run 2 (Feb20 20:37): 8 epochs (resume, epochs 87–94)

- Training log: `results/training_log_Feb20_2037.csv`
- Resumed from epoch 86 checkpoint
- Best test loss 0.009574 at epoch 94 — no improvement
- Confirms model has converged locally at ~0.0093

### Best local checkpoint

- `results/inpaint_gaussian_t250_best_checkpoint.pt` — epoch 56 (0-indexed),
  test loss 0.009336

---

## 2026-02-20 — Colab training

- Trained independently on Google Colab (T4 GPU) to **314 epochs**
- Best test loss: **0.012747** (worse than local — likely different random seed
  or data ordering; Colab run may not have used the same data split or
  hyperparameters as the local run)
- Checkpoint: `inpaint_gaussian_t250_Feb20_2305.pt` (stored in experiment root,
  not in results/)

---

## 2026-02-20 — Inpainting inference

Ran `scripts/test_attn_inpainting.py -n 10` using the **Colab checkpoint**
(epoch 314, test loss 0.0127). 10 validation samples, resample_steps=5.

| Sample | DDPM MSE | GP MSE | DDPM better? |
|--------|----------|--------|-------------|
| 0 | 0.000806 | 0.000717 | No |
| 1 | 0.002849 | 0.002156 | No |
| 2 | 0.000889 | 0.000350 | No |
| 3 | 0.004734 | 0.004058 | No |
| 4 | 0.004866 | 0.002622 | No |
| 5 | 0.006113 | 0.000893 | No |
| 6 | 0.002410 | 0.001026 | No |
| 7 | 0.000307 | 0.000429 | **Yes** |
| 8 | 0.001698 | 0.002963 | **Yes** |
| 9 | 0.006581 | 0.008975 | **Yes** |

| Metric | Value |
|--------|-------|
| Mean DDPM MSE | 0.003125 |
| Mean GP MSE | 0.002419 |
| DDPM/GP ratio | 1.29× (DDPM 29% worse) |
| DDPM wins | 3/10 samples |

**Result:** Attention UNet with Gaussian RePaint is competitive with GP on some
samples but averages 29% worse overall. The model was run from the Colab
checkpoint (test loss 0.0127) rather than the better local checkpoint (test
loss 0.0093) — **re-running inference with the local checkpoint may improve
results.**

Output files: `results/repaint_attn_results_{00..09}.pt` (per-sample),
`results/repaint_attn_results.pt` (combined), `results/repaint_attn_comparison.png`.

### Comparison to repaint_gaussian baseline

The attention model reached local test loss 0.0093 vs the vanilla model's
converged test loss (similar magnitude). The attention model's lower learning
rate (0.0003 vs 0.001) and smaller batch size (16 vs 80) slow convergence but
the final denoising quality is comparable.

**Key question:** Does the attention mechanism improve *inpainting* quality
despite similar denoising loss? The inference results above suggest modest
benefit — DDPM beats GP on 3/10 samples, but loses on average. Re-running
with the local checkpoint (lower test loss) is the natural next step.

---

## 2026-02-21 — Training improvements & restart

### Diagnosis

Model plateaued at epoch ~57 (best test loss 0.009294). Investigation showed
this is an **Adam optimizer plateau**, not overfitting (test loss < train loss
throughout). The optimizer's momentum estimates settle into a trajectory that
can't escape the current basin.

### Four improvements added

All are domain-agnostic optimizer/evaluation techniques:

1. **Cosine LR schedule with warmup** — 10-epoch linear warmup from
   `lr×0.001` → `lr`, then cosine decay to 0. Helps escape plateaus by
   varying the learning rate.
2. **AdamW with weight decay** (`weight_decay: 0.0001`) — mild L2
   regularization, decoupled from gradient updates.
3. **EMA** (`ema_decay: 0.9999`) — exponential moving average of model
   weights. Checkpoint saves the EMA weights for inference.
4. **Full test-set evaluation** — removed the old 20-batch cap on eval loop;
   all 1,965 test samples are now used every epoch.

### Augmentation decision

Velocity-field augmentation (H-flip negating u, V-flip negating v) was
initially implemented but **disabled** after analysis. The model learns
site-specific ocean current patterns at Ram's Head, St. John. Flipping the
field creates non-physical patterns (wrong coastline geometry, reversed
prevailing currents, wrong bathymetry influence). The augmentation
infrastructure remains in code (`augment: false` in config) for potential
future use with synthetic/multi-site datasets.

### Run 3 (Feb21): resumed from epoch 126 with improvements

- Config: `augment: false`, `weight_decay: 0.0001`, `lr_schedule: cosine`,
  `warmup_epochs: 10`, `use_ema: true`, `ema_decay: 0.9999`, `reset_best: true`
- Resuming from checkpoint `inpaint_gaussian_t250_Feb20_2037.pt` (epoch 124,
  best test 0.009294 pre-improvements)
- Two epochs (125–126) ran briefly with `augment: true` before the fix;
  training was killed and restarted with `augment: false`
- Training in progress

---

## Status

**Training in progress** with all 4 improvements (cosine LR, AdamW, EMA, full
eval). Previous best test loss was 0.009294 at epoch 57. Goal: break the
plateau and improve denoising quality for downstream inpainting.
