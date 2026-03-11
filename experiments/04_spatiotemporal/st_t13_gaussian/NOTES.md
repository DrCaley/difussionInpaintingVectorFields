# st_t13_gaussian — Experiment Notes

## 2026-02-25: Initial setup
- Created spatiotemporal UNet (`MyUNet_ST`) extending `MyUNet_Attn`
- T=13 frames = one M2 tidal cycle at hourly resolution
- 23.2M spatial + 2.1M temporal = 25.3M total parameters (8.9% overhead)
- Temporal layers zero-initialized → model starts identical to T independent
  copies of the pretrained spatial model
- Two-phase training: freeze spatial (50 epochs) → unfreeze (remaining)
- Dataset: OceanSequenceDataset, 7,598 training sequences from 131 chunks × 58 offsets
- Pretrained from: `02_inpaint_algorithm/repaint_gaussian_attn/results/inpaint_gaussian_t250_best_checkpoint.pt`

## 2026-02-25 – 2026-02-27: Training runs (full mode, ~2.06M temporal params)
- **Run 1** (freeze_spatial_epochs=15, lr=0.001): best test loss at epoch 2, severe overfitting
- **Run 2** (TODO settings): similarly overfit
- **Run 3** (freeze-forever, temporal_dropout=0.3, lr=1e-4): best at epoch ~4 of 76 trained.
  Still overfits despite aggressive regularization.
- **Inference comparison** (Run 3 best checkpoint): GP wins 3/4 test samples.
  ST model not competitive overall. Only beats GP on samples with strong temporal evolution.
- **Conclusion**: 2.06M temporal params for 7,598 sequences (~271 params/sample) is too much.
  Even with dropout=0.3 + frozen spatial + low LR, model memorizes within a few epochs.

## 2026-02-27: Architectural redesign — "lite" temporal mode
- **Root cause**: Temporal attention blocks (5 of them) contribute ~1.13M params (55%
  of temporal budget) with full QKV projections at 256-dim. With T=13, the attention
  can trivially memorize frame-to-frame relationships.
- **Fix**: Added `temporal_mode: lite` to `MyUNet_ST`:
  1. Replace all `TemporalConvBlock` (C→C) with `TemporalBottleneckConv` (C→C/4→C)
  2. Remove ALL temporal attention blocks
  3. Net result: 2,062,016 → **216,352 temporal params** (9.5× reduction)
  4. 28.5 params per training sample — within sustainable range
- **Verification**: Forward pass matches, zero-init verified (max diff 3.5e-6 vs T×2D)
- **New config**: `config_lite.yaml` — higher LR (0.001), lower dropout (0.05),
  freeze_spatial_epochs=100 then unfreeze with spatial_lr_factor=0.01
- **Next**: Train on Vast.ai RTX 5070 Ti with `config_lite.yaml`
