# NOTES — DPS Guided Inpainting (Div-Free Attention Model)

## Purpose
Test whether adding attention to the div-free model closes the gap with the
Gaussian attention model. The prior experiment (dps_divfree, no attention)
showed the model made random corrections (Corr = -0.04), suggesting
architecture was the bottleneck.

## Model
- MyUNet_Attn (same architecture as Gaussian attention model)
- Trained with fwd_diff_eq_divfree noise (spectrally equalized)
- Converged at epoch ~43, test loss 0.009 (same plateau as Gaussian)

## Key Results (Feb 24, 2026)
5 samples, boundary=2.0, div=0.0, GP warm start t=75:

| Model | Wins vs GP | Avg Ratio |
|-------|-----------|-----------|
| Gaussian-attn DPS | 5/5 | 0.725x |
| Div-free ATTN DPS | **0/5** | **2.553x** |

Individual samples: 2.66x, 2.94x, 1.36x, 2.98x, 2.83x GP — uniformly bad.

## Conclusion
**Architecture was NOT the bottleneck — the div-free noise itself is.**
Adding attention didn't help; the model is actually *worse* than the
no-attention version (2.55x vs 1.06x GP). The problem is fundamental:

- Real ocean data has ~16% divergent energy (div/curl ratio = 0.44)
- Div-free noise confines the model to a manifold that excludes this energy
- The model literally cannot represent the divergent component of the signal
- Larger network = more capacity to overfit to the div-free manifold
  (paradoxically making it worse at reconstructing real divergent fields)

**Verdict:** Div-free noise is a dead end for this dataset. Gaussian noise
with DPS guidance is the winning approach.
