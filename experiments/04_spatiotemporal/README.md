# Spatiotemporal Experiments (04)
#
# Research question: Does jointly processing T=13 consecutive hourly
# frames improve inpainting quality compared to per-frame independent
# denoising?  Can the model learn to predict future frames?
#
# Controlled variables:
#   - Spatial architecture: MyUNet_Attn (same as best 2D model)
#   - Noise function: gaussian
#   - Prediction target: eps
#   - Noise steps: 250
#   - Base learning rate: 0.001
#
# Varied variables:
#   - T (number of temporal frames): 13
#   - freeze_spatial_epochs: varies per experiment
#   - Pretrained spatial checkpoint: from 02_inpaint_algorithm/repaint_gaussian_attn

## Experiments

### st_t13_gaussian
First spatiotemporal experiment. T=13 hourly frames (one M2 tidal cycle).
Initialize spatial weights from best 2D MyUNet_Attn checkpoint.
Phase 1 (50 epochs): freeze spatial, train temporal layers only.
Phase 2 (remaining): unfreeze and fine-tune everything with cosine LR.
