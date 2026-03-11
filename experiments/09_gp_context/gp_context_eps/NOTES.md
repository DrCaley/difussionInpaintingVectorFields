# GP-Context Dense Conditioning — eps prediction

## Experiment Log

### Setup
- **Architecture**: MyUNet_Attn, in_channels=8
- **8 channels**: x_t(2) + mask(1) + GP_mean(2) + GP_var(1) + distance(1) + ocean_mask(1)
- **Prediction target**: eps (standard noise prediction)
- **Noise**: Gaussian, T=250
- **Training**: mask_xt=true, augmentation=true, loss on missing region only
- **Dataset**: rams_head, fixed sensor layout, precomputed GP fields

### Rationale
- FiLM conditioning failed at multi-step inference (2.18× GP MSE) due to sparse
  conditioning signal that compounds errors across 250 diffusion steps
- V-CNN succeeds with 5 dense channels — GP-context adapts this idea for DDPM
- All conditioning channels are dense (GP fills every pixel), no sparsity problem
- Standard eps-prediction enables multi-step iterative denoising (RePaint compatible)

### 2026-03-04: Initial training (v1) — DIVERGED at epoch 115
- **Config**: lr=0.001 (default), batch_size=80, no lr_schedule, no warmup, no
  weight_decay, no grad clipping, no EMA
- **Remote**: vast.ai RTX 5070 Ti, 96 CPU cores
- **GP precompute**: 95 workers, ~7 min, 1.4GB cache
- **Progress**: Loss decreased steadily from 0.170 (ep1) → 0.053 (ep112)
- **Divergence**: Epoch 115, train loss jumped 0.055 → 1.606, locked at ~2.0 for 60+ epochs
- **Root cause**: lr=0.001 too aggressive for 8ch attention UNet with batch_size=80.
  All other successful attention models use lr=0.0003 with cosine scheduling.
- Best checkpoint saved at epoch ~112 (test_loss=0.053)

### 2026-03-04: Resumed training (v2) — stabilized hyperparameters
- **Changes from v1**:
  - lr: 0.001 → 0.0003
  - lr_schedule: constant → cosine (min_lr=3e-6)
  - warmup_epochs: 0 → 10
  - weight_decay: 0 → 0.0001 (AdamW)
  - max_grad_norm: 0 → 1.0
  - use_ema: false → true (decay=0.9999)
- **Resumed** from best checkpoint at epoch 112 (test_loss=0.053)
- Training for 1000 more epochs (total budget: 1112 effective epochs)
- Log: `/root/train_gp_context_v2.log`
- Also fixed bug: warmup_epochs was gated by use_ema instead of being independent

### 2026-03-04: v2 converged — best at epoch 338
- **EMA test loss plateaued at ~0.049** (comparable to multi-mask v3)
- Best checkpoint saved at epoch 338
- Downloaded locally to `experiments/09_gp_context/gp_context_eps/results/`
- **STATUS: CONVERGED.** The 1000 epoch budget is a conservative ceiling;
  the model plateaued well before that. Do not treat this as "mid-training."

### 2026-03-04: RePaint + resample inference breakthrough
- 10/10 wins vs GP baseline, 0.649× GP MSE ratio
- This is the single-mask (fixed row-22) model
