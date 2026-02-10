# AI Agent Guide: Diffusion Inpainting Vector Fields

> **IMPORTANT:** This project maintains a research journal at [AI_RESEARCH_JOURNAL.md](AI_RESEARCH_JOURNAL.md). As an AI agent, you MUST add entries to this journal documenting:
> - Any experiments you run and their results
> - Bugs discovered and their fixes
> - Configuration changes and their effects
> - New findings or insights about the codebase
>
> This creates a persistent record of investigations for future reference.

---

## Quick Start Commands

```bash
# Activate environment
cd /path/to/diffusionInpaintingVectorFields
source env/bin/activate

# Run inpainting test (primary workflow)
python -m ddpm.Testing.model_inpainter

# Run training
python scripts/mega_trainer.py
```

---

## Configuration: `data.yaml`

The `data.yaml` file in the project root controls ALL behavior. Modify this file to switch between modes.

### Data Normalization Statistics (DO NOT MODIFY)
```yaml
u_training_mean: -0.06929559429949586
u_training_std: 0.1358005549716049
v_training_mean: -0.0323937796117541
v_training_std: 0.08899177232117582
mag_mean: 0.1446947511640557
```

### GPU Selection
```yaml
gpu_to_use: 0  # Set GPU index, or use cpu if unavailable
```

---

## Training Configuration

### Basic Training Settings
```yaml
training_mode: True       # Must be True to enable training
epochs: 1                 # Number of training epochs
batch_size: 100           # Batch size (~80 for servers with limited memory)
lr: 0.001                 # Learning rate
testSeed: 0               # Random seed for reproducibility
num_workers: 0            # DataLoader workers (0 for debugging)
```

### Retraining an Existing Model
```yaml
retrain_mode: True                    # Enable retraining
model_to_retrain: "path/to/model.pt"  # Path to checkpoint
```

### Noise Function Options
```yaml
noise_function: gaussian
# Options:
#   gaussian           - Standard Gaussian noise (default, most tested)
#   div_free           - Divergence-free noise for physics-informed inpainting
#   div_gaussian       - Combined divergence-free and Gaussian
#   hh_decomp_div_free - Helmholtz-Hodge decomposition based
#   cached_div         - Cached divergence-free noise
```

### Loss Function Options
```yaml
loss_function: mse
# Options:
#   mse       - Mean Squared Error (standard)
#   physical  - Physics-informed loss
#   best_loss - Custom combination loss

w1: 0.6  # Weight for MSE component
w2: 0.4  # Weight for divergence/physics component
```

### Data Standardization
```yaml
standardizer_type: zscore
# Options:
#   zscore - Z-score normalization (recommended)
#   maxmag - Max magnitude normalization
#   units  - Unit normalization
```

### Diffusion Process Parameters
```yaml
noise_steps: 100    # Number of diffusion steps (must match trained model!)
min_beta: 0.0001    # Minimum noise schedule
max_beta: 0.02      # Maximum noise schedule
```

---

## Inpainting Configuration

### Basic Inpainting Settings
```yaml
inpainting_batch_size: 1   # Usually 1 for inpainting
num_images_to_process: 1   # Number of test images
n_samples: 1               # Samples per image
resample_nums: [5]         # Resampling steps (higher = better quality, slower)
save_pt_fields: False      # Save tensor outputs
```

### Model Paths

**Automatic Model Selection (Recommended):**

The system automatically selects the correct model based on `noise_function`:

```yaml
noise_function: gaussian  # Change this to switch modes

# Models are auto-selected based on noise_function
model_by_noise_type:
  gaussian: "ddpm/Trained_Models/weekend_ddpm_ocean_model.pt"
  div_free: "ddpm/Trained_Models/div_free_model.pt"
```

**To switch between Gaussian and Divergence-Free mode, you only need to change `noise_function`:**
- `noise_function: gaussian` → uses `weekend_ddpm_ocean_model.pt`
- `noise_function: div_free` → uses `div_free_model.pt`

**Legacy Manual Model Paths (fallback):**
```yaml
model_paths:
  - "ddpm/Trained_Models/some_model.pt"
```

**IMPORTANT:** Model paths are relative to the project root directory.

### Divergence-Free Inpainting (Advanced)
```yaml
use_comb_net: auto  # yes/no/auto - Use combination network
comb_training_steps: 40
fidelity_weight: 0.01
physics_weight: 2
smooth_weight: 0
```

---

## Available Trained Models

Located in `ddpm/Trained_Models/`:

| Model | Description | Noise Type |
|-------|-------------|------------|
| `weekend_ddpm_ocean_model.pt` | Standard Gaussian DDPM | gaussian |
| `ddpm_ocean_good_normalized.pt` | Normalized model (n_steps=1000) | gaussian |
| `div_free_model.pt` | Divergence-free model | div_free |

**CRITICAL:** The `noise_steps` in `data.yaml` must match the model's training configuration. The weekend model uses `noise_steps: 100`.

---

## Running Inpainting Tests

### Method 1: Using ModelInpainter (Recommended)

```bash
python -m ddpm.Testing.model_inpainter
```

This reads from `data.yaml` and uses models specified in `model_paths`.

### Method 2: Direct Script

```bash
python scripts/test_gaussian_inpainting.py
```

---

## Running Training

### Single Model Training

Edit `data.yaml` with desired configuration, then:
```bash
python scripts/mega_trainer.py
```

### Batch Training

Place YAML configs in `models_to_train/` directory:
```bash
python scripts/mega_trainer.py
```
The script automatically trains all `.yaml` files in that directory.

---

## Mask Types for Inpainting

Available mask generators in `ddpm/helper_functions/masks/`:

| Mask | Import | Description |
|------|--------|-------------|
| `StraightLineMaskGenerator(n)` | `straigth_line` | n straight lines |
| `CoverageMaskGenerator(ratio)` | `n_coverage_mask` | percentage coverage |
| `RobotOceanMaskGenerator` | `robot_ocean_mask` | Simulated robot path |
| `RandomMaskGenerator` | `random_mask` | Random scattered mask |
| `GaussianMaskGenerator` | `gaussian_mask` | Gaussian-shaped mask |
| `BorderMaskGenerator` | `border_mask` | Border region mask |

---

## Common Workflows

### 1. Switch Between Gaussian and Divergence-Free Mode

**Single change in `data.yaml`:**
```yaml
# For Gaussian noise (fast, ~12 sec):
noise_function: gaussian

# For Divergence-free noise (slow, ~19 min):
noise_function: div_free
```

The correct model is automatically selected based on `model_by_noise_type` mapping.

Then run:
```bash
python -m ddpm.Testing.model_inpainter
```

### 2. Train a New Model with Different Loss Weights

```yaml
# data.yaml
training_mode: True
epochs: 50
noise_function: gaussian
loss_function: mse
w1: 0.8  # 80% MSE
w2: 0.2  # 20% divergence
```

### 3. Compare Multiple Models

Use legacy `model_paths` to test multiple models in one run:
```yaml
model_paths:
  - "ddpm/Trained_Models/model_a.pt"
  - "ddpm/Trained_Models/model_b.pt"
```

---

## Configuration Snapshot

### Current Working Configuration
```yaml
noise_function: gaussian  # SINGLE CHANGE to switch: gaussian or div_free
loss_function: mse
w1: 0.6
w2: 0.4
noise_steps: 100
min_beta: 0.0001
max_beta: 0.02
resample_nums: [5]

# Auto-selects model based on noise_function
model_by_noise_type:
  gaussian: "ddpm/Trained_Models/weekend_ddpm_ocean_model.pt"
  div_free: "ddpm/Trained_Models/div_free_model.pt"
```

---

## Critical Notes for AI Agents

### 1. Land Mask Handling
When creating custom inpainting scripts, ALWAYS apply the land mask:
```python
land_mask = (input_image_original.abs() > 1e-5).float().to(device)
raw_mask = mask_generator.generate_mask(input_image.shape).to(device)
mask = raw_mask * land_mask  # Critical!
```

### 2. Model Parameter Matching
Always check that `noise_steps`, `min_beta`, and `max_beta` match the trained model. Mismatches cause shape errors.

### 3. Standardization
Data must be standardized before model inference and unstandardized after:
```python
standardizer = dd.get_standardizer()
# Before model: data is already standardized from dataloader
# After model:
output = standardizer.unstandardize(model_output)
```

### 4. Cropping for Evaluation
The ocean data has land boundaries. Always crop to valid region for metrics:
```python
from ddpm.utils.inpainting_utils import top_left_crop
cropped = top_left_crop(tensor, 44, 94)
```

### 5. Environment Activation
Always activate the virtual environment:
```bash
source env/bin/activate
```

---

## Project Structure Reference

```
diffusionInpaintingVectorFields/
├── data.yaml                 # Main configuration file
├── data/                     # Ocean velocity data
├── ddpm/
│   ├── neural_networks/      # DDPM and UNet implementations
│   ├── Testing/              # Inpainting test scripts
│   │   └── model_inpainter.py
│   ├── Trained_Models/       # Saved model checkpoints
│   ├── training/             # Training scripts
│   ├── helper_functions/     # Utilities, masks, metrics
│   └── utils/                # Inpainting utilities
├── scripts/                  # Entry point scripts
│   ├── mega_trainer.py       # Main training script
│   └── test_gaussian_inpainting.py
├── models_to_train/          # YAML configs for batch training
└── env/                      # Python virtual environment
```

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `size mismatch for time_embed` | `noise_steps` doesn't match model | Check model's n_steps and update yaml |
| `no models in model_paths` | Model path doesn't exist | Verify path is correct relative to `ddpm/Testing/` |
| High MSE (~0.8) | Missing land mask | Apply `land_mask * raw_mask` |
| `CUDA out of memory` | Batch size too large | Reduce `batch_size` |
