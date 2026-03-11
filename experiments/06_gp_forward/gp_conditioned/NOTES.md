# GP-Conditioned Training Experiment

## What changed from GP-Forward
GP-forward noised the GP field and predicted GT — creating a distribution
mismatch because during reverse inference the model sees its own predictions
(approaching GT), not noised GP.

GP-conditioned fixes this by noising GT normally (standard DDPM) and feeding
the full GP field as conditioning via the known_values channels:
- Training:  input = [noised_GT, mask, GP_field], target = GT
- Inference: input = [x_t_reverse, mask, GP_field], target = GT

The GP is computed identically in both paths → zero distribution mismatch.

## Config
- Same FiLM+Attn UNet (28.6M params, 5ch input)
- EMA re-enabled (was off for GP-forward experiment)
- x0-prediction, mask_xt=true, cosine LR

## Log

### 2026-03-01 — v1 Training & Diagnosis

**Training v1** (gaussian noise, no augmentation, ema_decay=0.9999, weight_decay=0.0001):
- Ran ~187 epochs on vast.ai RTX 5070 Ti
- Best non-EMA test loss: ~0.036 at epoch ~9
- Severe overfitting: test loss increased from 0.036 to 0.060 while train→0.004
- Training overfit extremely fast (by epoch 9!) despite 9,180 training samples
- Root cause: fixed mask (no mask diversity) + no data augmentation

**EMA Bug**: EMA (decay=0.9999) hadn't converged when best checkpoint saved at
epoch ~9. EMA needs ~10K steps to average one window; at 574 steps/epoch, only
got ~5K steps. EMA weights were worse than non-EMA.

**Inference Results (v1 weights, 10 samples)**:
- Full 250-step reverse: **1.77-1.90x GP** (WORSE than doing nothing)
- Single-step from noised GP: **1.02x GP** (barely an improvement)
- More chain steps = worse results (chain error amplification)

**Critical Diagnostic — Per-Timestep x0 Prediction Quality**:
Single-step x0 predictions from noised GT are excellent:

| Timestep | ᾱ_t   | Ratio vs GP |
|----------|--------|-------------|
| t=1      | 0.9997 | 0.07-0.13x  |
| t=50     | 0.8984 | 0.07-0.13x  |
| t=100    | 0.6605 | 0.08-0.19x  |
| t=200    | 0.1949 | 0.25-0.45x  |
| t=249    | 0.0797 | 0.45-1.01x  |

The model's per-step predictions are GREAT (5-15x better than GP at low t!)
but the 250-step chain amplifies errors and destroys quality.

**Chain Amplification Root Cause**: With mask_xt=True, the known region of
x_t is replaced with independent noise at every step. This creates:
1. Incoherent known-region signal through attention layers at each step
2. No spatial anchoring — model relies entirely on GP conditioning + noised missing region
3. Small per-step errors in x0_pred compound through the DDPM posterior

**Alternative Inference Strategies (5 samples)**:

| Strategy                | vs GP  |
|------------------------|--------|
| Full reverse (250)      | 1.77x  |
| RePaste known (250)     | 1.67x  |
| No mask_xt (250)        | 1.15x  |
| 1-step from noise t=249 | 1.81x  |
| 1-step from GP t=25-100 | 1.02x  |
| 5-step DDIM from GP     | 1.21x  |

**Conclusion v1**: Model learns good per-step denoising but chain diverges.
More training needed to develop a stronger generative prior. Overfitting must
be addressed.

### 2026-03-01 — v2 Training Launch

Changes from v1:
- **augment: true** — velocity-field-aware H/V flips (4x effective data diversity)
- **weight_decay: 0.001** — 10x increase to fight overfitting
- **ema_decay: 0.999** — 10x faster EMA convergence (was 0.9999)

Added augmentation support to `OceanGPForwardDataset` (flips both GT and GP
fields consistently to preserve divergence structure).

Training launched as PID 12582 on vast.ai remote.
Log: `/workspace/training_gp_cond_v2.log`

### 2026-03-01 — v2 Inference Results (epoch ~29-32)

**Training v2 summary**:
- Epoch 32: Train=0.028, Test=0.028, EMA_Test=0.027
- **Zero overfitting** throughout (train ≈ test at all epochs)
- Test loss plateauing around 0.027-0.028 (was 0.036 best in v1)
- 24% lower test loss than v1's best

**Critical finding: Full reverse chain still broken (~2.18x GP).**
More training helped per-step quality but the 250-step chain error
amplification is structural with mask_xt=True, not a convergence issue.

**MAJOR BREAKTHROUGH — Single-step GP refinement works!**

The key insight: instead of running the full 250-step reverse chain from noise,
noise the GP field at a HIGH timestep (t=200-240) and predict x0 in a single
model forward pass. This avoids chain error accumulation entirely.

**Best strategies (10-sample evaluation, v2 EMA weights):**

| Strategy                 | vs GP | Win Rate |
|--------------------------|-------|----------|
| Ensemble 20x at t=200   | 0.85x | 10/10    |
| Ensemble 10x at t=200   | 0.87x | 10/10    |
| Ensemble 3x at t=200    | 0.87x | 10/10    |
| 1-step at t=240          | 0.91x | 9/10     |
| 2-stage ens3@200->ens3@50| 0.88x | 10/10    |
| Full reverse (250 steps) | 2.18x | 1/10     |

**Why this works**:
- At t=200-240, alpha_bar ≈ 0.08-0.19, so x_t ≈ 30-40% field + 90%+ noise
- Whether that 30% comes from GP or GT barely matters (small difference)
- Model relies heavily on GP conditioning at high t -> good predictions
- Single step = no error accumulation
- Ensemble averaging reduces prediction variance -> 0.85x with 20 samples

**Comparison with other methods:**

| Method          | vs GP  | Notes |
|-----------------|--------|-------|
| GP              | 1.000x | baseline |
| **GP-Cond ens10** | **0.87x** | **this work (10 fwd passes)** |
| GP-Diff (paper) | 0.74x  | 250-step chain, unconditional |
| FiLM (sparse)   | 0.98x  | sparse obs conditioning |
| Voronoi-CNN     | 0.27x  | deterministic DL baseline |
