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

## 2026-02-20 — Training run

- Trained for 43 epochs (killed before completion)
- Training log: `results/training_log_Feb20_0722.csv`
- Loss curve was still slowly improving when killed:
  - Epoch 1: test loss 0.0183
  - Epoch 4: test loss 0.0125
  - Epoch 43: test loss 0.0094 (best)
- Model checkpoint saved: `results/inpaint_gaussian_t250_best_checkpoint.pt`
- Training was terminated early — reason unclear from logs

### Comparison to repaint_gaussian

The attention model reached test loss 0.0094 at epoch 43 vs the vanilla model
which was trained to completion. The attention model's lower learning rate
and smaller batch size mean it needs more epochs to converge fully.

## Status

**Partially trained.** 43 epochs completed, best test loss 0.0094. The model
was learning and had not plateaued when training was killed. Needs to be
resumed or restarted to reach full convergence before comparing against
the `repaint_gaussian` baseline.

**To resume:**
```yaml
retrain_mode: true
model_to_retrain: experiments/02_inpaint_algorithm/repaint_gaussian_attn/results/inpaint_gaussian_t250_best_checkpoint.pt
reset_best: false
```
