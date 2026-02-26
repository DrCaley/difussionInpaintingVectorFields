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
