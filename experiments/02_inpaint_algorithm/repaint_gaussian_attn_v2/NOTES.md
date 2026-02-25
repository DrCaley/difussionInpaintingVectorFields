# NOTES — Gaussian Attention v2 (Regularized)

## Purpose
Test whether regularization (AdamW, cosine LR, EMA) improves on the original
attention model (repaint_gaussian_attn) which peaked at test_loss=0.0093
at epoch 57, then diverged.

## Training history
- Attempt 1: lr=0.001, batch_size=16 → collapsed at epoch 15
- Attempt 2: lr=0.0005, batch_size=16 → stable but slow
- Attempt 3: Added gradient_accumulation_steps=5 (eff batch=80), resumed epoch 31

## Result
Model converged but did not beat the original 0.0093 test loss.
The original model's checkpoint remains best for inference.
