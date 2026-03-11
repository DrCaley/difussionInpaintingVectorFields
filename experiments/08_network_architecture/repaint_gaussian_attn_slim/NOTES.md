# Slim Attention UNet — Experiment Notes

## Motivation

The full-size `MyUNet_Attn` (23.2M params) showed signs of overfitting on the
9,180-sample training set:
- Best test loss 0.009294 at epoch 57, then test loss slowly rose
- On 10 training samples: DDPM wins 6/10, but average MSE ratio 1.37× (GP
  better on average — even on *memorized* training data)
- DDPM collapses at mask coverage >92%, suggesting the problem is partly
  the RePaint inference algorithm, not just model capacity

The params-per-sample ratio was 2,523 — roughly 6× higher than successful
DDPM papers (e.g., Guided Diffusion at ~390 params/sample on ImageNet).

## Architecture Changes

| Property | Original (`standard_attn`) | Slim (`standard_attn_slim`) |
|----------|---------------------------|---------------------------|
| Channels | [64, 128, 256, 256] | [32, 64, 128, 128] |
| Attention levels | 16×32, 8×16, bottleneck | Bottleneck only (4×8) |
| Dropout | None | p=0.1 in every ResBlock |
| Total params | 23.2M | 6.1M |
| Trainable params | 23.1M | 6.1M |
| Params/sample | 2,523 | 666 |
| Batch size | 16 | 64 |
| Learning rate | 0.0003 | 0.001 |

### Why these changes

1. **Halved channels [32,64,128,128]**: Conv params scale as C_in × C_out, so
   halving channels quarters the weight count. Brings params/sample in line
   with Nichol & Dhariwal (2021) "Improved DDPM" recommendations.

2. **Dropout p=0.1**: Dhariwal & Nichol (2021) "Diffusion Models Beat GANs"
   used dropout 0.1–0.3 in ResBlocks and found it critical for preventing
   overfitting on smaller datasets (Table 2).

3. **Attention only at bottleneck**: The original DDPM paper (Ho et al., 2020)
   used attention at a single resolution level. Multi-level attention was a
   Guided Diffusion addition for 256×256+ images with millions of samples.
   For our 64×128 images, the conv receptive field covers most of the image
   by the third downsampling level. Bottleneck-only attention gives global
   communication (32 spatial positions) without quadratic cost at 512+ positions.

4. **Higher batch size (64) and LR (0.001)**: The smaller model fits easily in
   MPS memory, so we can use larger batches (closer to the base template's
   default of 80). Higher LR is appropriate for a model training from scratch
   with cosine schedule.

## Training config

- All improvements from the previous experiment carry forward: cosine LR
  with 10-epoch warmup, AdamW (weight_decay=0.0001), EMA (decay=0.9999),
  full test-set evaluation.
- Training from scratch (no retrain_mode).

---

## Run 1 — Feb 21 (no EMA warmup)

Training started but the epoch_loss dropped rapidly (1.08 → 0.02 by epoch 3)
while train/test loss stayed near 1.0. Root cause: **EMA warmup was missing**.
With `ema_decay=0.9999` and ~143 batches/epoch, the EMA shadow weights are
~83% random initialization after 13 epochs. Since `evaluate()` uses EMA weights
(`ema.apply()` before eval), the train/test metrics were scoring against
mostly-random weights.

Added `ema_warmup_steps: 2000` to config (~14 epochs). This linearly ramps
decay from 0 → 0.9999, so early EMA copies model weights almost exactly
before gradually switching to smoothing. Killed and restarted.

## Run 2 — Feb 21 (with EMA warmup, fresh start)

With warmup, train/test loss now tracks epoch_loss correctly:

| Epoch | Epoch Loss | Train Loss | Test Loss |
|-------|-----------|------------|-----------|
| 1 | 1.077 | 1.028 | 1.027 |
| 2 | 0.270 | 0.091 | 0.084 |
| 3 | 0.080 | 0.046 | 0.043 |

### Related finding: Big model regression explained

While investigating the EMA issue, discovered why the **big model**
(`repaint_gaussian_attn`) regressed from test_loss=0.010 → 0.014 after
adding training improvements via resume:

1. **Adam → AdamW optimizer state mismatch**: Checkpoint saved with plain Adam
   (`weight_decay=0`). Resumed with AdamW (`weight_decay=0.0001`).
   `load_state_dict()` loaded Adam momentum buffers into AdamW, then AdamW
   applied decoupled weight decay on top of momentum estimates that were
   tuned without it, actively damaging converged weights.

2. **Cosine LR fast-forward**: Scheduler was fast-forwarded 125 steps, but
   the model had been trained with constant LR. The cosine phase didn't
   match optimizer state expectations.

3. **EMA without warmup**: Minor factor — shadow was initialized from
   converged weights (not random), so eval was OK, but still suboptimal.

**Lesson learned**: When introducing new optimizer/scheduler settings, do NOT
resume with `load_state_dict` for the optimizer. Either:
- Start fresh with the new settings, or
- Resume but discard the optimizer state (fresh optimizer, load only model weights)

---

## Future Options (if this doesn't resolve overfitting)

### Option C: Switch to Conditional Architecture (FiLM or Palette)

**Priority: HIGH.** The training-data experiment showed DDPM collapses above
~92% mask coverage even on memorized data. This is a fundamental RePaint
limitation — at 93%+ coverage, the "paste known pixels + re-noise" step
covers so little area that the denoiser gets almost no conditioning signal.

A conditional model (FiLM or Palette concat) sees the mask and observed
values directly during training. It learns to inpaint *given* partial
information, bypassing RePaint's fragile re-noising loop.

- **FiLM** (Perez et al., 2018): Existing `MyUNet_FiLM` in codebase. Modulates
  intermediate features based on conditioning. Lightweight.
- **Palette concat** (Saharia et al., 2022): Existing `MyUNet_Inpaint`. Concatenates
  masked input as extra channels. State-of-the-art on image inpainting.
- Effort: ~1 day (models exist, just need config + training)

### Option D: Efficient Attention Variants

Replace standard O(n²) attention with:
- **Linear attention** (Katharopoulos et al., 2020): O(n) complexity via kernel
  approximation. Good for larger spatial resolutions.
- **Neighborhood/windowed attention** (Hassani et al., 2023, NAT): Local
  windows combine conv inductive bias with attention expressiveness.
- **Channel attention** (Zamir et al., 2022, Restormer): Attend across channels
  O(C²) instead of spatial positions O(HW²). Since C ≪ HW, much cheaper.
  Lets you put attention at higher resolutions without quadratic blowup.
- Effort: half day per variant

### Option E: U-ViT / DiT (Vision Transformer backbone)

- **U-ViT** (Bao et al., 2023): Treat all inputs as tokens, process with ViT +
  long skip connections. Matched UNet-based DDPMs on CIFAR-10/CelebA.
- **DiT** (Peebles & Xie, 2023): Pure transformer, no convolutions. State-of-
  the-art ImageNet generation.
- Risk: ViTs are more data-hungry than ConvNets. With 9K samples, the lack of
  inductive bias (translation equivariance, locality) could hurt.
- Effort: 2-3 days

### Option F: Hybrid — Slim ConvUNet + Conditional Inference

Combine the slim model with a conditional training objective. Train the slim
UNet as a Palette-style 5-channel model (x_t, mask, known_values → ε).
This addresses both problems simultaneously: reduced overfitting (fewer params
+ dropout) AND better high-mask-coverage performance (conditional model doesn't
need RePaint).

---
