# Experiment Group 01: Noise Strategy Comparison
#
# Research question: Which noise construction method produces the best
# inpainting results for divergence-free ocean vector fields?
#
# Controlled variables:
#   - UNet architecture: FiLM-conditioned (5ch)
#   - Prediction target: x0 (direct clean-image prediction)
#   - Noise schedule: 250 steps, β ∈ [0.0004, 0.08]
#   - Loss: MSE
#   - mask_xt: true
#   - Dataset: rams_head ocean currents (64×128)
#
# Varied variable:
#   - noise_function: forward_diff_div_free | spectral_div_free | gaussian
#
# Expected outcome:
#   - forward_diff_div_free should produce lowest divergence in inpainted region
#   - gaussian baseline should have higher MSE in masked region (no physics prior)
#
# Status: fwd_divfree trained to epoch 380 (best test loss 0.0441 @ epoch 365)

