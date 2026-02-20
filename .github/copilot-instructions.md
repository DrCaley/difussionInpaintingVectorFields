# Project Copilot Instructions

## Research Context

This project develops DDPM-based inpainting for ocean velocity (vector) fields
with divergence-free physical constraints. **The core research goal is
inpainting under very high mask coverage (70–95%+ missing data).** This is the
realistic operating regime for autonomous ocean robots that have only observed
a small fraction of the domain. High mask percentages are NOT a problem — they
are the primary research challenge and the reason this work exists. Never flag
high mask coverage as an issue or suggest reducing it.

Training epoch counts (e.g. 1000) are chosen conservatively to ensure
convergence; models typically plateau well before the full run completes.
When a model's loss has clearly flattened, treat it as converged — do NOT
blame "insufficient training" or "only N/1000 epochs" for poor inference
results. If results are bad despite a converged model, the issue is
architectural, algorithmic, or in the inference pipeline, not training
duration.

## How to Run an Experiment

### Structure

Experiments live in `experiments/` organized by research question:

```
experiments/
  templates/
    base_inpaint.yaml        ← shared defaults (DO NOT edit per-experiment)
  run_experiment.py          ← launcher: merge + validate + train
  01_noise_strategy/         ← group folder (one research question)
    README.md                ← what's being tested, controlled/varied variables
    fwd_divfree/
      config.yaml            ← ONLY the overrides (2-5 lines typically)
      NOTES.md               ← per-experiment log: observations, issues, results
      results/               ← auto-created: resolved config, checkpoints, logs
    spectral_divfree/
      config.yaml
      NOTES.md
      results/
    gaussian_baseline/
      config.yaml
      NOTES.md
      results/
  02_prediction_target/      ← next research question
    README.md
    ...
```

### Creating a New Experiment

1. **Decide which group** it belongs to. If testing a new research question, create a new numbered folder (`experiments/NN_description/`) with a `README.md` explaining what's being varied and what's held constant.

2. **Create the override config** — only include keys that differ from `experiments/templates/base_inpaint.yaml`:
   ```yaml
   # experiments/01_noise_strategy/my_new_noise/config.yaml
   model_name: my_new_noise_experiment
   noise_function: forward_diff_div_free
   ```

3. **Create a NOTES.md** in the experiment folder to log observations,
   issues, and results as work progresses. Update this file whenever you
   run training, inference, or diagnostics for this experiment. Each entry
   should be dated and include what was done and what was observed.

4. **Validate before training** (dry run):
   ```bash
   PYTHONPATH=. python experiments/run_experiment.py --dry-run experiments/01_noise_strategy/my_new_noise/config.yaml
   ```
   This prints the fully resolved config and checks component compatibility using `ddpm/protocols.py`.

5. **Smoke test** (3 epochs):
   ```bash
   PYTHONPATH=. python experiments/run_experiment.py --smoke experiments/01_noise_strategy/my_new_noise/config.yaml
   ```

6. **Full training**:
   ```bash
   PYTHONPATH=. python experiments/run_experiment.py experiments/01_noise_strategy/my_new_noise/config.yaml
   ```

### What the Launcher Does

`experiments/run_experiment.py`:
- Deep-merges `base_inpaint.yaml` + your `config.yaml` (overrides win)
- Validates noise↔standardizer, noise↔projection, prediction_target↔inpaint compatibility (via `ddpm.protocols`)
- Writes the fully resolved config to `results/resolved_config.yaml`
- Launches `ddpm/training/train_inpaint.py --training_cfg <resolved_config>`

### Component Compatibility Rules

These are enforced by the launcher and defined in `ddpm/protocols.py`:

| Rule | Why |
|------|-----|
| Div-free noise → unified standardizer | Per-component z-score breaks ∇·v=0 |
| forward_diff noise → forward_diff projection | Discrete operators must match |
| x0 prediction → x0 inpainting algorithms | eps inpainting can't use x0 predictions |
| UNet type must match network class | film→MyUNet_FiLM, concat→MyUNet_Inpaint |

### Useful Config Keys to Override

| Key | Values | Default | Notes |
|-----|--------|---------|-------|
| `noise_function` | `gaussian`, `forward_diff_div_free`, `spectral_div_free`, `div_free` | `forward_diff_div_free` | |
| `unet_type` | `film`, `concat`, `standard` | `film` | `standard` = unconditional (2ch) |
| `prediction_target` | `x0`, `eps` | `x0` | |
| `mask_xt` | `true`, `false` | `true` | Replace known region in x_t with noise |
| `noise_steps` | int | `250` | Diffusion timesteps |
| `epochs` | int | `1000` | |
| `lr` | float | `0.001` | |
| `batch_size` | int | `80` | |
| `loss_function` | `mse`, `physical`, `best_loss` | `mse` | |
| `p_uncond` | float 0-1 | `0.0` | Classifier-free guidance dropout |
| `model_name` | string | auto | Name for output folder |

### Resuming Training

Set these in your override config:
```yaml
retrain_mode: true
model_to_retrain: experiments/01_.../results/checkpoint.pt
reset_best: false   # set true if loss metric changed
```

## DDPM Building Blocks

See `ddpm/protocols.py` for formal protocol definitions and the ASCII architecture diagram. Key building blocks:

- **NoiseStrategy** (`ddpm/utils/noise_utils.py`): Generates ε for forward process
- **LossStrategy** (`ddpm/helper_functions/loss_functions.py`): Training loss
- **Standardizer** (`ddpm/helper_functions/standardize_data.py`): Data normalization
- **MaskGenerator** (`ddpm/helper_functions/masks/`): Inpainting mask shapes
- **UNet** (`ddpm/neural_networks/unets/`): Denoiser network
- **GaussianDDPM** (`ddpm/neural_networks/ddpm.py`): Forward/backward wrapper

## Algorithm & Variant Documentation

Two reference documents describe every paper-based technique and our variants:

- **`design/algorithm-catalog.md`** — Human-readable prose with paper references,
  math, our modifications, code pointers, and a combination matrix.
- **`design/algorithms.yaml`** — Machine-readable catalog of conditioning methods,
  inpainting algorithms, noise strategies, projections, losses, and standardizers
  with compatibility constraints and experiment status.

### Quick Reference: Key Algorithms

| Category | Name | Paper | Config / Function |
|----------|------|-------|-------------------|
| **Conditioning** | FiLM | Perez et al. 2018 | `unet_type: film` |
| | Palette concat | Saharia et al. 2022 | `unet_type: concat` |
| | Unconditional | Ho et al. 2020 | `unet_type: standard` |
| **Inpainting** | RePaint | Lugmayr et al. 2022 | `repaint_standard()` |
| | RePaint + CG | Our variant | `repaint_standard(project_div_free=True)` |
| | Full-reverse x₀ | Our adaptation | `x0_full_reverse_inpaint()` |
| | Palette mask-aware | Saharia et al. 2022 | `mask_aware_inpaint()` |
| | CFG inpainting | Ho & Salimans 2021 | `mask_aware_inpaint_cfg()` |
| | Gradient-guided | Dhariwal & Nichol 2021 | `guided_inpaint()` |
| **Noise** | Gaussian | Standard | `noise_function: gaussian` |
| | Fwd-diff div-free | Our construction | `noise_function: forward_diff_div_free` |
| | Spectral div-free | Our construction | `noise_function: spectral_div_free` |
| **Projection** | CG streamfunction | Our implementation | `forward_diff_project_div_free()` |
| | FFT Helmholtz | Standard | `spectral_project_div_free()` |
| | Jacobi Poisson | Standard | `project_div_free_2d()` |

## Running Tests

```bash
PYTHONPATH=. python -m pytest unit_tests/test_protocols.py -v
```
