# DDPM Ocean Current Inpainting - Logical Design

**Version**: 1.0  
**Date**: 2026-01-05  
**Author**: Jeff Caley (PLU)  
**Status**: Draft  
**Phase**: A - Architecture

## Document History

| Version | Date       | Author     | Changes       |
| ------- | ---------- | ---------- | ------------- |
| 1.0     | 2026-01-05 | Jeff Caley | Initial draft |

---

## 1. System Overview

The system consists of three primary components that form a pipeline:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           DDPM OCEAN INPAINTING SYSTEM                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐                     │
│  │             │      │             │      │             │                     │
│  │  COMPONENT  │      │  COMPONENT  │      │  COMPONENT  │                     │
│  │     1       │ ───▶ │     2       │ ───▶ │     3       │                     │
│  │             │      │             │      │             │                     │
│  │  Training   │      │  Inpainting │      │  Evaluation │                     │
│  │   System    │      │   System    │      │   System    │                     │
│  │             │      │             │      │             │                     │
│  └─────────────┘      └─────────────┘      └─────────────┘                     │
│        │                    │                    │                             │
│        ▼                    ▼                    ▼                             │
│   Trained Model        Inpainted           Comparison                          │
│   Checkpoint           Fields              vs GP Baseline                      │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Component 1: Training System

### 2.1 Purpose
Train a neural network to predict noise added to ocean velocity fields during the forward diffusion process (DDPM).

### 2.2 Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TRAINING SYSTEM                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                 │
│  │   Config     │     │    Data      │     │   Model      │                 │
│  │  (YAML)      │────▶│ Initializer  │────▶│  Factory     │                 │
│  │              │     │              │     │              │                 │
│  └──────────────┘     └──────────────┘     └──────────────┘                 │
│                              │                    │                          │
│                              ▼                    ▼                          │
│                       ┌──────────────┐     ┌──────────────┐                 │
│                       │   Dataset    │     │  GaussianDDPM│                 │
│                       │   + Loader   │     │   + UNet     │                 │
│                       └──────────────┘     └──────────────┘                 │
│                              │                    │                          │
│                              └────────┬──────────┘                          │
│                                       ▼                                      │
│                              ┌──────────────┐                               │
│                              │   Trainer    │                               │
│                              │ (xl_ocean_   │                               │
│                              │  trainer.py) │                               │
│                              └──────────────┘                               │
│                                       │                                      │
│                    ┌──────────────────┼──────────────────┐                  │
│                    ▼                  ▼                  ▼                  │
│             ┌──────────┐      ┌──────────────┐    ┌──────────┐             │
│             │ Noise    │      │    Loss      │    │  Model   │             │
│             │ Strategy │      │   Strategy   │    │ Checkpoint│            │
│             └──────────┘      └──────────────┘    └──────────┘             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.3 Key Classes

| Class | File | Responsibility |
|-------|------|----------------|
| `TrainOceanXL` | `ddpm/training/xl_ocean_trainer.py` | Orchestrates training loop, checkpointing, logging |
| `DDInitializer` | `data_prep/data_initializer.py` | Singleton that loads config, data, and creates strategies |
| `GaussianDDPM` | `ddpm/neural_networks/ddpm.py` | DDPM forward/backward process implementation |
| `MyUNet` | `ddpm/neural_networks/unets/unet_xl.py` | Noise prediction network (encoder-decoder with time embedding) |
| `OceanImageDataset` | `data_prep/ocean_image_dataset.py` | PyTorch Dataset for velocity fields |

### 2.4 Training Flow

```
1. Load Configuration (data.yaml or custom YAML)
        │
        ▼
2. DDInitializer creates:
   - NoiseStrategy (gaussian, div_free, hh_decomp_div_free)
   - LossStrategy (mse, physical, best_loss)  
   - Standardizer (zscore, maxmag, units)
   - Train/Val/Test DataLoaders
        │
        ▼
3. For each epoch:
   │
   ├─▶ For each batch (x0, t, noise):
   │       │
   │       ├─▶ Forward: x_t = √(ᾱ_t) * x0 + √(1-ᾱ_t) * ε   (add noise)
   │       │
   │       ├─▶ Backward: ε̂ = UNet(x_t, t)                   (predict noise)
   │       │
   │       └─▶ Loss: L(ε, ε̂) + λ * divergence_penalty
   │
   └─▶ Evaluate on validation set
        │
        ▼
4. Save best checkpoint with:
   - model_state_dict
   - optimizer_state_dict
   - noise_strategy
   - standardizer_strategy
   - training config
```

### 2.5 Entry Points

| Script | Purpose |
|--------|---------|
| `scripts/mega_trainer.py` | Batch training - trains all models defined in `models_to_train/*.yaml` |
| `TrainOceanXL` direct | Single model training with custom config |

### 2.6 Outputs

| Output | Location | Description |
|--------|----------|-------------|
| Best checkpoint | `training_output/{config_name}/ddpm_ocean_model_best_checkpoint.pt` | Full model state for inference |
| Training log | `training_output/{config_name}/training_log_{timestamp}.csv` | Epoch, train loss, test loss |
| Loss plot | `training_output/{config_name}/training_test_loss_xl_{timestamp}.png` | Visual training curve |
| Config used | `training_output/{config_name}/config_used.yaml` | Saved configuration for reproducibility |

---

## 3. Component 2: Inpainting System

### 3.1 Purpose
Use a trained DDPM to reconstruct complete velocity fields from sparse observations (inpainting).

### 3.2 Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         INPAINTING SYSTEM                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                 │
│  │   Trained    │     │    Input     │     │    Mask      │                 │
│  │   Model      │     │    Image     │     │  Generator   │                 │
│  │  Checkpoint  │     │   (x0)       │     │              │                 │
│  └──────────────┘     └──────────────┘     └──────────────┘                 │
│         │                    │                    │                          │
│         └────────────────────┼────────────────────┘                          │
│                              ▼                                               │
│                    ┌──────────────────┐                                     │
│                    │  inpaint_generate│                                     │
│                    │  _new_images()   │                                     │
│                    └──────────────────┘                                     │
│                              │                                               │
│                              ▼                                               │
│         ┌────────────────────────────────────────────┐                      │
│         │          REVERSE DIFFUSION LOOP            │                      │
│         │                                            │                      │
│         │  for t = T → 0:                           │                      │
│         │    1. Denoise: x̂_{t-1} = denoise(x_t)     │                      │
│         │    2. Preserve known: combine with mask    │                      │
│         │    3. Optionally resample (RePaint)        │                      │
│         │    4. Apply vector combination network     │                      │
│         │                                            │                      │
│         └────────────────────────────────────────────┘                      │
│                              │                                               │
│                              ▼                                               │
│                    ┌──────────────────┐                                     │
│                    │  Inpainted       │                                     │
│                    │  Velocity Field  │                                     │
│                    └──────────────────┘                                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Key Classes

| Class | File | Responsibility |
|-------|------|----------------|
| `ModelInpainter` | `ddpm/Testing/model_inpainter.py` | Orchestrates inpainting tests, comparison with GP |
| `MaskGenerator` (abstract) | `ddpm/helper_functions/masks/abstract_mask.py` | Interface for creating observation masks |
| `inpaint_generate_new_images` | `ddpm/utils/inpainting_utils.py` | Core inpainting algorithm |
| `combine_fields` | `ddpm/vector_combination/vector_combiner.py` | Blends known/predicted regions |

### 3.4 Mask Types Available

| Mask Class | File | Description |
|------------|------|-------------|
| `CoverageMaskGenerator` | `n_coverage_mask.py` | Target coverage percentage |
| `StraightLineMaskGenerator` | `straigth_line.py` | Simulates AUV transects |
| `RandomMaskGenerator` | `random_mask.py` | Random pixel sampling |
| `RobotPathMaskGenerator` | `robot_path.py` | Simulated robot trajectory |
| `GaussianNoiseBinaryMaskGenerator` | `gaussian_mask.py` | Gaussian-weighted sampling |
| `BorderMaskGenerator` | `border_mask.py` | Border observations only |
| `SquigglyLineMaskGenerator` | `squiggly_line.py` | Curved transect paths |

### 3.5 Inpainting Algorithm (RePaint-style)

```python
# Pseudocode for inpaint_generate_new_images()

1. Forward noise input image to all timesteps:
   noised_images[0] = x0
   for t in 0..T:
       noised_images[t+1] = noise_one_step(noised_images[t], t)

2. Initialize: x = noise * mask + noised_images[T] * (1-mask)

3. Reverse diffusion with resampling:
   for t = T-1 → 0:
       for r = 1 → resample_steps:
           # Denoise
           inpainted = denoise_one_step(x, t)
           
           # Get known region at this noise level
           known = noised_images[t]
           
           # Combine known and predicted
           combined = combine_fields(known, inpainted, mask)
           
           # If not last resample, re-noise for next iteration
           if r < resample_steps:
               x = noise_one_step(combined, t)

4. Final result: x0 * (1-mask) + combined * mask
```

### 3.6 Vector Combination Network

For physics-aware blending at mask boundaries:

| Class | File | Purpose |
|-------|------|---------|
| `VectorCombiner` | `vector_combination/vector_combiner.py` | Orchestrates combination |
| `CombinerUNet` | `vector_combination/combiner_unet.py` | Learned boundary blending |
| `CombinationLoss` | `vector_combination/combination_loss.py` | Fidelity + physics + smoothness |

---

## 4. Component 3: Evaluation System

### 4.1 Purpose
Systematically compare DDPM inpainting quality against Gaussian Process (GP) baseline across different mask configurations.

### 4.2 Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         EVALUATION SYSTEM                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                      ModelInpainter                                   │   │
│  │                                                                       │   │
│  │  Inputs:                        Outputs:                             │   │
│  │  - Trained model checkpoint     - CSV: metrics per test case         │   │
│  │  - List of MaskGenerators       - Plots: MSE vs coverage/distance    │   │
│  │  - Validation dataset           - Saved tensors (.pt files)          │   │
│  │                                  - Heatmaps (optional)               │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                              │                                               │
│                              ▼                                               │
│         ┌────────────────────────────────────────────┐                      │
│         │           FOR EACH TEST IMAGE              │                      │
│         │                                            │                      │
│         │  1. Generate mask from MaskGenerator       │                      │
│         │  2. Run DDPM inpainting                    │                      │
│         │  3. Run GP fill (baseline)                 │                      │
│         │  4. Compute metrics for both               │                      │
│         │  5. Log to CSV                             │                      │
│         │                                            │                      │
│         └────────────────────────────────────────────┘                      │
│                              │                                               │
│                              ▼                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                         METRICS COMPUTED                              │   │
│  │                                                                       │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │   │
│  │  │ Normalized  │  │  Percent    │  │  Angular    │  │  Magnitude  │ │   │
│  │  │    MSE      │  │   Error     │  │   Error     │  │   Diff      │ │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.3 Key Functions

| Function | File | Purpose |
|----------|------|---------|
| `calculate_mse()` | `ddpm/utils/inpainting_utils.py` | Normalized MSE over masked region |
| `calculate_percent_error()` | `ddpm/utils/inpainting_utils.py` | Relative error per pixel |
| `gp_fill()` | `ddpm/helper_functions/interpolation_tool.py` | Gaussian Process baseline |
| `save_angular_error_heatmap()` | `plots/visualization_tools/error_visualization.py` | Direction error visualization |
| `save_magnitude_difference_heatmap()` | `plots/visualization_tools/error_visualization.py` | Speed error visualization |

### 4.4 CSV Output Schema

```
model, image_num, mask, num_lines, resample_steps, ddpm_mse, gp_mse, mask_percent, average_pixel_distance
```

### 4.5 Entry Points

| Script | Purpose |
|--------|---------|
| `scripts/mega_inpainter.py` | Batch evaluation - tests all models in `ddpm/trained_models/` |
| `ModelInpainter` direct | Single model evaluation with custom masks |

### 4.6 Evaluation Dimensions

| Dimension | Purpose | Plot |
|-----------|---------|------|
| MSE vs Mask Coverage % | How does error scale with prediction area? | Scatter plot |
| MSE vs Avg Distance to Observed | How far can we extrapolate? | Scatter plot |
| Per-pixel heatmaps | Where are errors concentrated? | 2D heatmap |

---

## 5. Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA FLOW                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ROMS Model Output                                                          │
│  (stjohn_hourly_5m_velocity_ramhead_v2.mat)                                │
│         │                                                                    │
│         ▼                                                                    │
│  ┌──────────────┐                                                           │
│  │ spliting_    │  Creates train/val/test split                             │
│  │ data_sets.py │  Saves to data.pickle                                     │
│  └──────────────┘                                                           │
│         │                                                                    │
│         ▼                                                                    │
│  ┌──────────────┐                                                           │
│  │ data.pickle  │  Serialized numpy arrays                                  │
│  │              │  (training_data, validation_data, test_data)              │
│  └──────────────┘                                                           │
│         │                                                                    │
│         ▼                                                                    │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐              │
│  │DDInitializer │ ───▶ │Standardizer  │ ───▶ │OceanImage    │              │
│  │              │      │(z-score)     │      │Dataset       │              │
│  └──────────────┘      └──────────────┘      └──────────────┘              │
│         │                                                                    │
│         ├───────────────────────┬───────────────────────┐                   │
│         ▼                       ▼                       ▼                   │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐              │
│  │ Training     │      │ Validation   │      │    Test      │              │
│  │ DataLoader   │      │ DataLoader   │      │  DataLoader  │              │
│  └──────────────┘      └──────────────┘      └──────────────┘              │
│         │                       │                       │                   │
│         ▼                       │                       │                   │
│  ┌──────────────┐               │                       │                   │
│  │   TRAINING   │               │                       │                   │
│  │              │               │                       │                   │
│  └──────────────┘               │                       │                   │
│         │                       │                       │                   │
│         ▼                       ▼                       ▼                   │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐              │
│  │   Trained    │      │  INPAINTING  │      │  EVALUATION  │              │
│  │   Model      │─────▶│              │─────▶│              │              │
│  │  (.pt file)  │      │              │      │              │              │
│  └──────────────┘      └──────────────┘      └──────────────┘              │
│                                                      │                      │
│                                                      ▼                      │
│                                             ┌──────────────┐               │
│                                             │  Results CSV │               │
│                                             │  + Plots     │               │
│                                             └──────────────┘               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Configuration System

### 6.1 Primary Config: `data.yaml`

```yaml
# Data statistics (for standardization)
u_training_mean: -0.0693
u_training_std: 0.1358
v_training_mean: -0.0324
v_training_std: 0.0890

# Training parameters
epochs: 100
batch_size: 80
lr: 0.001

# Strategy selection
noise_function: div_free      # gaussian | div_free | hh_decomp_div_free
loss_function: mse            # mse | physical | best_loss
standardizer_type: zscore     # zscore | maxmag | units

# Loss weights (for physical loss)
w1: 0.6  # MSE weight
w2: 0.4  # Divergence weight

# Diffusion parameters
noise_steps: 100
min_beta: 0.0001
max_beta: 0.02

# Inpainting parameters
resample_nums: [5]
use_comb_net: auto
```

### 6.2 Model-Specific Configs: `models_to_train/*.yaml`

Override specific parameters for different experiments (e.g., loss weight ablation).

---

## 7. Neural Network Architecture

### 7.1 UNet (MyUNet)

```
Input: (B, 2, 64, 128) + time embedding
       │
       ▼
┌─────────────────────────────────────┐
│          ENCODER                     │
│                                      │
│  Block1: 2→16 channels (64×128)     │
│     ↓ Conv(4,2,1)                   │
│  Block2: 16→32 channels (32×64)     │
│     ↓ Conv(4,2,1)                   │
│  Block3: 32→64 channels (16×32)     │
│     ↓ Conv(4,2,1)                   │
│  Block4: 64→128 channels (8×16)     │
│     ↓ Conv(4,2,1) × 2               │
│                                      │
└─────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│          BOTTLENECK                  │
│  128→256→128 channels (2×4)         │
└─────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│          DECODER                     │
│  (mirror of encoder with skip       │
│   connections)                       │
│                                      │
│  Block5: 256→64 channels (4×8)      │
│     ↑ ConvT(4,2,1)                  │
│  Block6: 128→32 channels (16×32)    │
│     ↑ ConvT(4,2,1)                  │
│  Block7: 64→16 channels (32×64)     │
│     ↑ ConvT(4,2,1)                  │
│  Block8: 32→16 channels (64×128)    │
│                                      │
└─────────────────────────────────────┘
       │
       ▼
Output: Conv(16→2) → (B, 2, 64, 128)
        (predicted noise ε̂)
```

### 7.2 Time Embedding

Sinusoidal positional embedding (100-dim) added to each block via learned linear projection.

---

## 8. Strategy Patterns

### 8.1 Noise Strategies

| Strategy | Class | Physics | Use Case |
|----------|-------|---------|----------|
| `gaussian` | `GaussianNoise` | None | Standard DDPM baseline |
| `div_free` | `DivergenceFreeNoise` | ∇·ε = 0 | Physics-informed diffusion |
| `hh_decomp_div_free` | `HH_Decomp_Div_Free` | Helmholtz-Hodge | Strongest physics constraint |

### 8.2 Loss Strategies

| Strategy | Class | Formula | Use Case |
|----------|-------|---------|----------|
| `mse` | `MSELossStrategy` | MSE(ε, ε̂) | Standard training |
| `physical` | `PhysicalLossStrategy` | w₁·MSE + w₂·‖∇·ε̂‖² | Divergence penalty |
| `best_loss` | `HotGarbage` | MSE + divergence on unstandardized | Experimental |

### 8.3 Standardization Strategies

| Strategy | Class | Transform |
|----------|-------|-----------|
| `zscore` | `ZScoreStandardizer` | (x - μ) / σ per channel |
| `maxmag` | `MaxMagStandardizer` | x / max_magnitude |
| `units` | `UnitStandardizer` | No transform |

---

## 9. Architecture Decision Records (ADRs)

### ADR-001: Divergence-Free Noise for Physics Consistency

**Context**: Ocean currents are approximately incompressible (∇·v ≈ 0).

**Decision**: Implement noise strategies that produce divergence-free noise fields, constraining the diffusion process to physically plausible states.

**Consequences**: 
- (+) Reduces hallucination of non-physical flows
- (+) Aligns with H2.2 hypothesis
- (-) More complex noise generation
- (-) May reduce model flexibility

### ADR-002: RePaint-Style Resampling for Inpainting

**Context**: Standard DDPM conditioning doesn't preserve known observations well.

**Decision**: Implement resampling at each diffusion step (RePaint algorithm) to repeatedly enforce known constraints.

**Consequences**:
- (+) Better preservation of observed values
- (+) Configurable via `resample_nums` parameter
- (-) Increases inference time linearly with resample count

### ADR-003: GP Baseline for Fair Comparison

**Context**: Need a meaningful baseline to demonstrate DDPM value.

**Decision**: Use Gaussian Process interpolation (`gp_fill`) as the state-of-the-art baseline for all comparisons.

**Consequences**:
- (+) GP is a well-established method in oceanography
- (+) Fair comparison on identical masked inputs
- (-) GP can be slow for large grids

### ADR-004: Singleton DDInitializer for Configuration

**Context**: Many components need access to shared configuration and data.

**Decision**: Use singleton pattern for `DDInitializer` to provide centralized access.

**Consequences**:
- (+) Consistent configuration across all components
- (+) Easy access without passing config everywhere
- (-) Global state can complicate testing
- (-) Must call `reset_instance()` between different configs

---

## 10. File Structure Summary

```
ddpm/
├── neural_networks/
│   ├── ddpm.py                 # GaussianDDPM - forward/backward diffusion
│   ├── interpolation_ddpm.py   # Partial convolution variant
│   └── unets/
│       ├── unet_xl.py          # Primary UNet architecture
│       ├── unet_xxl.py         # Larger variant
│       └── pconv_*.py          # Partial convolution variants
│
├── training/
│   ├── xl_ocean_trainer.py     # TrainOceanXL - training orchestration
│   └── training_output/        # Saved models and logs
│
├── Testing/
│   ├── model_inpainter.py      # ModelInpainter - evaluation orchestration
│   └── results/                # Evaluation outputs
│
├── utils/
│   ├── noise_utils.py          # NoiseStrategy implementations
│   └── inpainting_utils.py     # Inpainting algorithm + metrics
│
├── helper_functions/
│   ├── loss_functions.py       # LossStrategy implementations
│   ├── compute_divergence.py   # Divergence calculation
│   ├── HH_decomp.py            # Helmholtz-Hodge decomposition
│   ├── interpolation_tool.py   # GP fill baseline
│   ├── standardize_data.py     # Standardizer implementations
│   └── masks/                  # MaskGenerator implementations
│
└── vector_combination/
    ├── vector_combiner.py      # Field combination orchestration
    ├── combiner_unet.py        # Combination network
    └── combination_loss.py     # Physics-aware combination loss

data_prep/
├── data_initializer.py         # DDInitializer singleton
├── ocean_image_dataset.py      # PyTorch Dataset
└── spliting_data_sets.py       # Train/val/test split creation

scripts/
├── mega_trainer.py             # Batch training entry point
└── mega_inpainter.py           # Batch evaluation entry point

plots/
└── visualization_tools/
    ├── error_visualization.py  # Error heatmaps
    └── pt_visualizer_plus.py   # Tensor visualization
```

---

## 11. Verification Checklist

### Component 1: Training System
- [ ] `mega_trainer.py` successfully trains a model from YAML config
- [ ] Model checkpoint saves all necessary state (model, optimizer, strategies)
- [ ] Training logs (CSV + plot) are generated correctly
- [ ] Validation loss is computed and used for best model selection
- [ ] Different noise strategies can be selected via config
- [ ] Different loss strategies can be selected via config

### Component 2: Inpainting System
- [ ] Trained model loads correctly from checkpoint
- [ ] Inpainting produces output of correct shape
- [ ] Known observations are preserved in output
- [ ] Resample parameter affects output quality
- [ ] Different mask types can be used

### Component 3: Evaluation System
- [ ] DDPM and GP metrics are computed on identical inputs
- [ ] CSV output contains all required columns
- [ ] MSE vs coverage plot is generated
- [ ] Results are reproducible with same random seed

---

## 12. Approvals

| Role | Name | Date | Status |
|------|------|------|--------|
| Lead Researcher | Jeff Caley | 2026-01-05 | Draft |
| Technical Review | | | Pending |

---

*Document follows RAPID Method Phase 2 (Architecture) template*
