# NOTES — fwd_divfree_attn

## Purpose
Train the same attention UNet (MyUNet_Attn) that dominates with Gaussian noise,
but using forward-diff div-free noise instead.  The original div-free model
(repaint_cg) used MyUNet without attention and had 8.8M params — this tests
whether the architectural gap was the bottleneck.

## What's controlled
- Same architecture as repaint_gaussian_attn (standard_attn)
- Same hyperparameters (lr=3e-4, cosine schedule, warmup=10, EMA, AdamW)
- Same prediction target (eps)

## What varies
- noise_function: forward_diff_div_free (vs gaussian)
- standardizer: unified z-score (auto-resolved from noise)

## Context
- DPS guidance with Gaussian-attn model: 0.725x GP (5/5 wins, boundary=2.0)
- DPS guidance with div-free MyUNet: 1.06x GP (architecture too weak)
- If this model converges well, test with DPS guidance (boundary only, no div penalty)
