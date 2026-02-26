# NOTES — DPS Guided Inpainting (Gaussian Attention Model)

## Purpose
Test Diffusion Posterior Sampling (Chung et al. 2022, arxiv 2209.14687) as an
alternative inference algorithm to RePaint. DPS replaces hard copy-paste with
soft gradient guidance, avoiding the manifold-violation problem.

## Algorithm
At each reverse step, compute denoised estimate x̂₀ from model prediction,
then apply gradient correction:
  x_{t-1} = x_{t-1}^pred − ζ_b · ∇||y − M⊙x̂₀||² / ||y − M⊙x̂₀||

Key innovations over our initial attempt:
- **DPS-style normalization**: divide gradient by residual norm (not MSE)
- **Sum-based loss** (not mean): preserves proper gradient magnitude
- **GP warm start**: forward-diffuse GP output to t_start, begin reverse there

## Key Results

### 5-sample sweep (Feb 23, 2026)
| Boundary | Div | Wins | Avg Ratio |
|----------|-----|------|-----------|
| 0.5 | 0.0 | 4/5 | 0.851x |
| 1.0 | 0.0 | 4/5 | 0.793x |
| 2.0 | 0.0 | **5/5** | **0.725x** |
| 0.5 | 0.1 | 3/5 | 0.896x |
| 1.0 | 0.1 | 4/5 | 0.829x |
| 2.0 | 0.1 | 4/5 | 0.765x |
| * | 0.5 | — | all ≥ 1.0x (divergence penalty hurts) |

→ Divergence penalty harmful (data has 16% divergent energy).
→ Higher boundary guidance consistently better.

### 100-sample evaluation (Feb 24, 2026)
Config: boundary=2.0, div=0.0, GP warm start t=75

| Metric | Value |
|--------|-------|
| Wins vs GP | 88/100 |
| Avg ratio | 0.800x |
| Median ratio | 0.865x |
| Min ratio | 0.284x |
| Max ratio | 1.094x |

### Comparison with RePaint (100 samples each)
| Method | Wins | Avg Ratio |
|--------|------|-----------|
| DPS Guided (b=2.0) | **88/100** | 0.800x |
| RePaint Adaptive | 85/100 | **0.744x** |

DPS wins more often but RePaint gets bigger improvements when it wins.

## Divergence penalty analysis
Training data has ~16% divergent energy (RMS div/curl ratio = 0.44).
Setting div > 0 penalizes real physics → hurts reconstruction.
