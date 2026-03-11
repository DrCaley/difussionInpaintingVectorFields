# Helmholtz Split Decoder — Experiment Notes

## What
Option C architecture: split the decoder after dec3 (16×32), giving each head
(ψ=solenoidal, φ=irrotational) its own independent dec2+dec1 operating at
32×64 and 64×128 resolution. This breaks the shared high-res features that
enable cancellation degeneracy, while keeping the expensive attention layers
(dec4, dec3) shared.

+6% params over baseline (24.43M vs 23.15M). Only 1.28M extra parameters —
all in cheap non-attention ResBlocks.

Combined with helmholtz_supervised loss (λ_decomp=0.1, λ_orth=0.01) for
per-head supervision.

## Why
All 3 previous Helmholtz models (baseline, supervised, split_noise) exhibited
severe cancellation degeneracy: v_sol and v_irr were 4-40× the total energy,
cancelling each other. Root cause: shared decoder features allow trivial
negation. The supervision loss alone couldn't fix it because the shared
features provide too strong a coordination channel.

## 2026-03-09 — Created
- Architecture implemented in `unet_helmholtz_split.py`
- Using helmholtz_supervised loss to combine architecture + loss fixes
- Same training recipe as baseline/supervised for fair comparison
