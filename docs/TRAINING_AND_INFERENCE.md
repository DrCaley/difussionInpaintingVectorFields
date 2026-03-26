# Training & Inference Guide

This guide covers how to train DDPM inpainting models using the experiment
framework and how to run inference with trained checkpoints.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Training](#training)
   - [Creating an Experiment](#creating-an-experiment)
   - [Config Overrides](#config-overrides)
   - [Running Training](#running-training)
   - [Resuming Training](#resuming-training)
   - [Monitoring Progress](#monitoring-progress)
3. [Inference](#inference)
   - [Reading a Checkpoint's Settings](#reading-a-checkpoints-settings)
   - [Choosing the Right Inpainting Algorithm](#choosing-the-right-inpainting-algorithm)
   - [Running an Eval Script](#running-an-eval-script)
   - [Inference Parameters](#inference-parameters)
4. [Reference Tables](#reference-tables)
   - [UNet Types](#unet-types)
   - [Noise Functions](#noise-functions)
   - [Loss Functions](#loss-functions)
   - [Full Config Key Reference](#full-config-key-reference)

---

## Prerequisites

```bash
# Activate the environment
source env/bin/activate

# All commands assume PYTHONPATH includes the project root
export PYTHONPATH=.
```

---

## Training

### Creating an Experiment

Experiments live under `experiments/`, grouped by research question. Each
experiment has a small **override config** that changes only what differs
from the base template (`experiments/templates/base_inpaint.yaml`).

```
experiments/
  templates/
    base_inpaint.yaml           ← shared defaults (DO NOT edit per-experiment)
  run_experiment.py             ← launcher script
  01_noise_strategy/            ← research question group
    README.md                   ← what's being tested
    fwd_divfree/
      config.yaml               ← overrides only (2–5 lines)
      NOTES.md                  ← observations and results log
      results/                  ← auto-created at training time
        resolved_config.yaml    ← full merged config (for reproducibility)
        training_log_*.csv      ← epoch-by-epoch metrics
        *_best_checkpoint.pt    ← best model (full state)
        *_best_weights.pt       ← best model (weights only)
```

**Steps:**

1. Pick or create a group folder (`experiments/NN_description/`).

2. Create `config.yaml` with only the keys you want to override:
   ```yaml
   # experiments/01_noise_strategy/my_experiment/config.yaml
   model_name: my_experiment
   noise_function: gaussian
   prediction_target: eps
   ```

3. Create a `NOTES.md` to log observations as work progresses.

### Config Overrides

The override config is merged on top of `base_inpaint.yaml`. You only need
to specify keys that differ. Common overrides:

```yaml
# Minimal example — FiLM UNet with x0 prediction
model_name: my_film_x0
unet_type: film
prediction_target: x0
noise_function: forward_diff_div_free
```

```yaml
# Concat UNet with eps prediction (Palette-style)
model_name: my_concat_eps
unet_type: concat
prediction_target: eps
noise_function: gaussian
```

```yaml
# With topology penalties
model_name: my_topo_experiment
loss_function: helmholtz_supervised
helmholtz_loss:
  lambda_decomp: 0.1
  lambda_orth: 0.01
topology:
  lambda_vort: 0.01
  lambda_div: 0.005
  t_max_frac: 0.1
```

### Running Training

**Validate first** (dry-run — prints resolved config, checks compatibility):
```bash
PYTHONPATH=. python experiments/run_experiment.py --dry-run \
    experiments/01_noise_strategy/my_experiment/config.yaml
```

**Smoke test** (3 epochs, checks that everything wires up):
```bash
PYTHONPATH=. python experiments/run_experiment.py --smoke \
    experiments/01_noise_strategy/my_experiment/config.yaml
```

**Full training:**
```bash
PYTHONPATH=. python experiments/run_experiment.py \
    experiments/01_noise_strategy/my_experiment/config.yaml
```

The launcher:
1. Deep-merges your override config on top of `base_inpaint.yaml`
2. Auto-resolves `standardizer_type` and `enable_divergence_projection`
   based on your `noise_function`
3. Validates component compatibility (see `ddpm/protocols.py`)
4. Writes `results/resolved_config.yaml` (the complete config)
5. Launches `ddpm/training/train_inpaint.py`

### Resuming Training

Add these keys to your `config.yaml`:

```yaml
retrain_mode: true
model_to_retrain: experiments/.../results/inpaint_*_best_checkpoint.pt
reset_best: false   # set true if you changed the loss function
```

The checkpoint restores model weights, optimizer state, epoch counter, and
loss history. The LR scheduler is fast-forwarded to the correct step.

### Monitoring Progress

Training logs are written to `results/training_log_<date>.csv` with columns:

| Column | Description |
|--------|-------------|
| Epoch | Epoch number |
| Epoch Loss | Raw epoch loss |
| Train Loss | Training loss (may differ if using advanced loss) |
| Test Loss | Validation loss |
| MSE | MSE component of loss |
| Decomp | Helmholtz decomposition penalty (if applicable) |
| Orth | Orthogonality penalty (if applicable) |
| Topo | Topology penalty (if applicable) |
| EMA Train/Test Loss | EMA model losses (if `use_ema: true`) |

**Checkpoints saved when test loss improves:**
- `*_best_checkpoint.pt` — full state (model + optimizer + history), for resuming
- `*_best_weights.pt` — weights only, lighter, for inference
- `*_best_ema_weights.pt` — EMA weights (if enabled)

---

## Inference

### Reading a Checkpoint's Settings

Every trained model has a `resolved_config.yaml` in its `results/` folder.
These fields determine which inference algorithm and settings to use:

```yaml
# The three critical fields:
prediction_target: x0          # x0 or eps
unet_type: film                # which UNet architecture
noise_function: forward_diff_div_free   # noise strategy used in training

# Also important:
mask_xt: true                  # must match at inference
p_uncond: 0.0                  # if >0, model supports classifier-free guidance
noise_steps: 250               # number of diffusion timesteps
```

If you only have a `.pt` file without the resolved config, you can inspect
the checkpoint to determine the architecture:

```python
import torch
ck = torch.load("checkpoint.pt", map_location="cpu", weights_only=False)
# Check keys for architecture clues:
for k in ck["model_state_dict"]:
    print(k)
# FiLM models have "film_scale" / "film_shift" keys
# Helmholtz models have "psi_head" / "phi_head" keys
# Concat models have large first-layer weights (5 input channels)
```

### Choosing the Right Inpainting Algorithm

The inpainting algorithm **must match** the model's `prediction_target` and
`unet_type`. Using the wrong algorithm will produce garbage.

| `prediction_target` | `unet_type` | Algorithm | Function |
|---------------------|-------------|-----------|----------|
| **x0** | film | Full-reverse x₀ | `x0_full_reverse_inpaint()` |
| **x0** | concat | Full-reverse x₀ | `x0_full_reverse_inpaint()` |
| **x0** | helmholtz_split_film* | Full-reverse x₀ | `x0_full_reverse_inpaint()` |
| **eps** | concat | Mask-aware (Palette) | `mask_aware_inpaint()` |
| **eps** | film | Mask-aware | `mask_aware_inpaint()` |
| **eps** | standard | RePaint | `repaint_standard()` |
| **eps** | standard, `p_uncond>0` | CFG inpainting | `mask_aware_inpaint_cfg()` |
| **eps** | standard | Gradient-guided | `guided_inpaint()` |

**Key rules:**
- `x0_full_reverse_inpaint()` **requires** `prediction_target: x0` — it
  reads the model's x₀ predictions directly.
- `repaint_standard()` works with any prediction target but is designed for
  unconditional (standard) UNets using copy-paste.
- `mask_xt` **must match** what the model was trained with. FiLM models
  are always trained with `mask_xt: true`.

All inpainting functions live in `ddpm/utils/inpainting_utils.py`.

### Running an Eval Script

Evaluation scripts follow a standard pattern. Here's a typical invocation:

```bash
PYTHONPATH=. python scripts/eval_subframes_baseline.py \
    --weights path/to/model_best_weights.pt \
    --coverage 0.5 1.0 2.0 5.0 \
    --n-samples 10 \
    --t-starts 25 50 \
    --resample 3 \
    --n-ensemble 10
```

**Common arguments across eval scripts:**

| Argument | Description | Typical values |
|----------|-------------|----------------|
| `--weights` | Path to model checkpoint | Required |
| `--coverage` | Observation coverage percentages | `0.5 1.0 2.0 5.0` |
| `--n-samples` | Number of test samples to evaluate | `1` (quick), `10–20` (paper) |
| `--t-starts` | Starting timestep(s) for reverse chain | `25 50 100` |
| `--resample` | RePaint resample steps per timestep | `1` (none), `3–10` |
| `--n-ensemble` | Ensemble chains to average | `1` (quick), `10` (paper) |
| `--seed` | Random seed | `42` |

**Output:** Results are saved as `.pt` files first (tensors for every method),
then plots are generated from the saved data. This ensures expensive GPU
inference is never wasted if plotting fails.

### Inference Parameters

**`t_start` — Starting Timestep**

Controls how much of the reverse chain to run. The model was trained with
`noise_steps` timesteps (default 250). At inference, you start from timestep
`t_start` and denoise down to 0.

| t_start | Speed | Quality | Notes |
|---------|-------|---------|-------|
| 25 | Fast | Good for easy masks | Starts near clean data |
| 50 | Medium | Good default | Balanced |
| 100 | Slow | Best for sparse data | More room for the model to work |

Higher `t_start` gives the model more steps to shape the output but costs
proportionally more compute.

**`n_ensemble` — Ensemble Averaging**

Each reverse chain is stochastic (random noise at each step). Averaging
multiple independent chains reduces variance and produces smoother
predictions.

| n_ensemble | Use case |
|------------|----------|
| 1 | Quick testing |
| 5 | Reasonable quality |
| 10 | Paper-quality results |

Compute scales linearly with `n_ensemble`.

**`resample` — RePaint Resampling**

For RePaint-style algorithms only. At each reverse step, re-noise and
re-denoise `resample` times to improve boundary coherence between known
and unknown regions.

| resample | Effect |
|----------|--------|
| 1 | No resampling (fastest) |
| 3 | Mild boundary smoothing |
| 5–10 | Stronger harmonization (slower) |

**`project_steps` — Divergence-Free Projection**

Post-hoc CG projection onto the divergence-free manifold via streamfunction
solve. Applied after the reverse chain completes.

| project_steps | Effect |
|---------------|--------|
| 0 | No projection |
| 3–5 | Typical; removes boundary divergence artifacts |

---

## Reference Tables

### UNet Types

| `unet_type` | Class | Input Ch | Conditioning | Prediction |
|-------------|-------|----------|-------------|------------|
| `film` | MyUNet_FiLM | 5 | FiLM layers | x0 or eps |
| `concat` | MyUNet_Inpaint | 5 | Channel concat | x0 or eps |
| `standard` | MyUNet | 2 | None (unconditional) | x0 or eps |
| `standard_attn` | MyUNet_Attn | 2 | None + self-attention | x0 or eps |
| `standard_attn_slim` | MyUNet_Attn_Slim | 2 | None + slim attention | x0 or eps |
| `standard_attn_mid` | MyUNet_Attn_Mid | 2 | None + mid attention | x0 or eps |
| `helmholtz` | MyUNet_Helmholtz | 2 | None, dual-head (ψ,φ) | x0 |
| `helmholtz_split` | MyUNet_Helmholtz_Split | 2 | Split decoders (ψ,φ) | x0 |
| `helmholtz_split_film` | MyUNet_Helmholtz_Split_FiLM | 5 | FiLM + split decoders | x0 |
| `helmholtz_split_film_multires` | MyUNet_Helmholtz_Split_FiLM_MultiRes | 5+ | FiLM + FPN encoder + split decoders | x0 |
| `helmholtz_split_film_crossattn` | MyUNet_Helmholtz_Split_FiLM_CrossAttn | 5 | Cross-attention + split decoders | x0 |
| `spatiotemporal` | MyUNet_ST | 2T | Temporal modules | x0 or eps |

**Unconditional types** (`standard*`, `helmholtz`, `helmholtz_split`,
`spatiotemporal`) automatically disable `mask_xt` and `p_uncond`.

### Noise Functions

| `noise_function` | Standardizer | Projection | Description |
|-------------------|-------------|------------|-------------|
| `gaussian` | Per-component z-score | None | Standard Gaussian noise |
| `forward_diff_div_free` | Unified z-score | Forward-diff CG | Div-free noise via forward-difference curl (default) |
| `spectral_div_free` | Unified z-score | FFT Helmholtz | Div-free noise via spectral projection |
| `div_free` | Unified z-score | Jacobi Poisson | Original div-free noise |

Standardizer and projection are auto-resolved from `noise_function` — you
don't need to set them manually.

### Loss Functions

| `loss_function` | Description |
|-----------------|-------------|
| `mse` | Standard MSE (default) |
| `physical` | MSE + divergence penalty (weighted by `w1`, `w2`) |
| `best_loss` | Adaptive best-of-N loss |
| `helmholtz_supervised` | MSE + Helmholtz decomposition + topology penalties |

For `helmholtz_supervised`, configure via nested keys:
```yaml
loss_function: helmholtz_supervised
helmholtz_loss:
  lambda_decomp: 0.1      # decomposition supervision weight
  lambda_orth: 0.01        # orthogonality penalty weight
topology:
  lambda_vort: 0.01        # vorticity penalty weight
  lambda_div: 0.005        # divergence penalty weight
  t_max_frac: 0.1          # only apply topo loss for t < t_max_frac * n_steps
  warmup_epochs: 0         # delay topo loss activation
```

### Full Config Key Reference

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| **Architecture** | | | |
| `unet_type` | string | `film` | UNet variant (see table above) |
| `mask_xt` | bool | `true` | Replace known region of x_t with noise during training |
| `prediction_target` | string | `x0` | What the model predicts: `x0` or `eps` |
| `p_uncond` | float | `0.0` | Classifier-free guidance dropout (0 = disabled) |
| `gp_forward` | bool | `false` | Noise GP posterior instead of GT |
| `use_bathymetry` | bool | `false` | Include bathymetry channel (subframes only) |
| **Training** | | | |
| `epochs` | int | `1000` | Training epochs |
| `batch_size` | int | `80` | Batch size |
| `lr` | float | `0.001` | Learning rate |
| `lr_schedule` | string | `constant` | `constant` or `cosine` |
| `warmup_epochs` | int | `0` | Linear warmup for cosine schedule |
| `weight_decay` | float | `0` | AdamW weight decay (0 = plain Adam) |
| `max_grad_norm` | float | `0` | Gradient clipping (0 = disabled) |
| `augment` | bool | `false` | Velocity-field-aware flip augmentation |
| `use_ema` | bool | `false` | Exponential moving average of weights |
| `ema_decay` | float | `0.9999` | EMA decay rate |
| **Noise** | | | |
| `noise_function` | string | `forward_diff_div_free` | Noise strategy |
| `noise_steps` | int | `250` | Diffusion timesteps |
| `min_beta` | float | `0.0001` | Minimum noise schedule value |
| `max_beta` | float | `0.02` | Maximum noise schedule value |
| **Loss** | | | |
| `loss_function` | string | `mse` | Loss strategy |
| `w1` | float | `0.6` | Physical loss MSE weight |
| `w2` | float | `0.4` | Physical loss divergence weight |
| **Resume** | | | |
| `retrain_mode` | bool | `false` | Resume from checkpoint |
| `model_to_retrain` | string | `null` | Path to checkpoint file |
| `reset_best` | bool | `false` | Reset best loss tracker on resume |
| **Misc** | | | |
| `model_name` | string | auto | Experiment name (used for output folder) |
| `gpu_to_use` | int | `0` | GPU device index |
| `num_workers` | int | `0` | DataLoader workers |
