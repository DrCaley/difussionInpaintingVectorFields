# helmholtz_matched_weakfullsup

## 2026-03-10: Experiment Created

**Motivation**: Ablation to isolate the effect of full-time decomposition
supervision from the effect of increased supervision weights.

The fullsup run (λ_decomp=0.5, λ_orth=0.1, decomp_t_max_frac=1.0) completely
fixed head degeneracy (cos_sol=0.95, cancel=1.3x) but cost ~10% reconstruction
quality (1.142x V-CNN). The original matched_noise run (λ_decomp=0.1, λ_orth=0.01,
decomp_t_max_frac≈0.1) had great reconstruction (1.038x V-CNN) but degenerate
heads (cos_sol=0.28, cancel=5.8x).

This run tests: **Can full-time supervision at the original weak weights fix
the heads without the reconstruction penalty?**

**Only change vs original matched_noise**: `decomp_t_max_frac: 1.0`
(was ~0.1 hardcoded). Weights unchanged: λ_decomp=0.1, λ_orth=0.01.

**Controlled comparison**:
| Experiment | λ_decomp | λ_orth | decomp_t_max_frac | epochs |
|------------|----------|--------|-------------------|--------|
| matched_noise (original) | 0.1 | 0.01 | ~0.1 | 400 |
| fullsup_800 (Server 1) | 0.5 | 0.1 | 1.0 | 800 |
| **weakfullsup (this, Server 2)** | **0.1** | **0.01** | **1.0** | **800** |

**Server**: Server 2 (1.208.108.242:58777, RTX 3060)
