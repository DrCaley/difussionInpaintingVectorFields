# multires_maskxt — Experiment Notes

## 2025-03-19: Created experiment

**Motivation**: `multires_standard_diffusion` (mask_xt=false) failed fundamentally
at inpainting. Even oracle denoising (GT-noised x_t, exact training distribution)
was 40x worse than V-CNN. Root cause: with mask_xt=false, the model learns
unconditional whole-image denoising. Loss is computed over entire image, and x_t
contains noised GT everywhere — no incentive to use FiLM conditioning from sparse
observations.

**Fix**: `mask_xt: true` changes two things:
1. Known region in x_t replaced with `randn()` noise → forces model to read
   from conditioning channels (sparse obs via FPN encoder), not x_t spatial structure
2. Loss computed ONLY on missing region (`diff * mask_2ch`) → model optimizes
   specifically for reconstruction of unobserved cells

**Config**: Identical to multires_standard_diffusion except `mask_xt: true`.

**Training**: Running on both Server 2 (RTX 3060) and Server 1 (RTX 5060 Ti).
Epoch 1 loss: ~2.1 (expected — much higher than mask_xt=false since now predicting
missing regions from noise). By epoch 2: train~0.41, test~0.13.

**Note**: `weakfullsup_detach_stddiff` was already training with mask_xt=true
(inherited from base_inpaint.yaml default). It converged at epoch 260 with
test loss 0.0071. That checkpoint may already work for inpainting and should
be evaluated.
