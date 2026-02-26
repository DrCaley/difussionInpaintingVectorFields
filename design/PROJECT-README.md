# Diffusion Inpainting Vector Fields - RAPID Project Documentation

> **RAPID Method Applied**: This project follows the RAPID AI-assisted software development methodology.
> 
> **Current Phase**: R (Rationale) - Problem definition and requirements gathering

---

## Project Overview

**Purpose**: Predict ocean currents in a region from sparse in-situ point measurements by treating this as an inpainting problem solved with Denoising Diffusion Probabilistic Models (DDPM).

**Domain**: Oceanography / Scientific Machine Learning / Inverse Problems

**Problem Statement**: 
Ocean current measurements are expensive and sparse - sensors can only sample velocity at discrete points. Given a small set of in-situ measurements, we want to reconstruct the full 2D velocity vector field for the surrounding region. This is framed as an **inpainting problem**: the sparse measurements are "known pixels" and the DDPM fills in the rest while respecting physical constraints (e.g., approximate incompressibility/divergence-free flow).

**Key Goal**: Create a model that accurately reconstructs complete ocean velocity fields from sparse point samples, enabling better understanding of ocean dynamics with minimal sensor deployment.

---

## RAPID Method Integration

This project uses the [RAPID AI Software Engineering Method](../../RAPID/cp-rapid-ai-1/README.md) for structured development.

### Configuration Files

| File | Location | Purpose |
|------|----------|---------|
| `rapid-config.json` | `.rapid/` | Project configuration |
| `rapid-status.json` | `design/` | Project and iteration state |

### Phase Status

- **Phase R (Rationale)**: 🔄 In Progress - Documenting existing system and requirements
- **Phase A (Architecture)**: ⏳ Pending
- **Phase P (Planning)**: ⏳ Pending  
- **Phase I/D (Implementation)**: 🔧 Existing code available (needs RAPID documentation)

---

## Repository Structure

```
diffusionInpaintingVectorFields/
├── .rapid/                    # RAPID configuration
│   └── rapid-config.json      # Project configuration
├── design/                    # RAPID design artifacts
│   └── rapid-status.json      # Project status tracking
│
├── data/                      # Input data
│   └── rams_head/             # Ram's Head ocean current data
│       ├── boundaries.yaml    # Geographic boundaries
│       └── *.mat              # MATLAB velocity data files
│
├── data_prep/                 # Data preprocessing modules
│   ├── data_initializer.py    # Data initialization utilities
│   ├── minimal_dataloader.py  # Lightweight data loading
│   ├── ocean_image_dataset.py # PyTorch dataset for ocean images
│   ├── polar_dataset_splitter.py
│   └── spliting_data_sets.py  # Train/test splitting
│
├── ddpm/                      # Core DDPM implementation
│   ├── helper_functions/      # Utility functions
│   │   ├── calculator.py
│   │   ├── compute_divergence.py
│   │   ├── HH_decomp.py       # Helmholtz-Hodge decomposition
│   │   ├── interpolation_tool.py
│   │   ├── loss_functions.py
│   │   ├── model_evaluation.py
│   │   ├── standardize_data.py
│   │   └── view_tensor.py
│   │
│   ├── neural_networks/       # Neural network architectures
│   │   ├── ddpm.py            # Base DDPM implementation
│   │   ├── interpolation_ddpm.py
│   │   └── unets/             # UNet architectures
│   │
│   ├── Testing/               # Model testing and evaluation
│   │   ├── model_inpainter.py # Inpainting execution
│   │   └── results/           # Test results
│   │
│   ├── Trained_Models/        # Saved model checkpoints
│   │   └── ddpm_ocean_good_normalized.pt
│   │
│   ├── training/              # Training infrastructure
│   │   ├── xl_ocean_trainer.py
│   │   └── training_output/
│   │
│   ├── utils/                 # General utilities
│   │   ├── inpainting_utils.py
│   │   └── noise_utils.py
│   │
│   └── vector_combination/    # Vector field combination
│       ├── combination_loss.py
│       ├── combiner_unet.py
│       └── vector_combiner.py
│
├── noising_process/           # Noise generation
│   ├── incompressible_gp/     # Incompressible Gaussian process
│   └── simple_gp/             # Simple Gaussian process
│
├── models_to_train/           # Training configurations
│   ├── div_free_comb_net_initial.yaml
│   └── gaussian_mse_*.yaml    # Various loss weight configurations
│
├── scripts/                   # Execution scripts
│   ├── batch_training.sh      # Batch training script
│   ├── mega_inpainter.py      # Large-scale inpainting
│   └── mega_trainer.py        # Large-scale training
│
├── unit_tests/                # Test suites
│   ├── ddpm_tests/
│   ├── helper_function_tests/
│   └── vector_combination_tests/
│
├── paper/                     # Research paper materials
│   └── template.tex           # LaTeX paper template
│
├── plots/                     # Visualization
│   ├── outputs/
│   └── visualization_tools/
│
├── data.yaml                  # Main configuration file
├── env/                       # Python virtual environment
└── README.md                  # This file
```

---

## Configuration (data.yaml)

The main configuration file controls:

### Training Parameters
- `epochs`: Number of training epochs
- `batch_size`: Training batch size
- `lr`: Learning rate
- `standardizer_type`: Data normalization (`zscore`, `maxmag`, `units`)

### Noise Functions
- `gaussian`: Standard Gaussian noise
- `div_free`: Divergence-free noise (physics-informed)
- `hh_decomp_div_free`: Helmholtz-Hodge decomposition based

### Loss Functions
- `mse`: Mean squared error
- `physical`: Physics-informed loss
- `w1`, `w2`: Loss component weights

### Inpainting Settings
- `noise_steps`: Diffusion steps
- `resample_nums`: Resampling iterations
- `use_comb_net`: Vector combination network mode

---

## Quick Start

1. **Activate environment**:
   ```bash
   source env/bin/activate
   ```

2. **Configure training** in `data.yaml`

3. **Train a model**:
   ```bash
   python scripts/mega_trainer.py
   ```

4. **Run inpainting**:
   ```bash
   python scripts/mega_inpainter.py
   ```

---

## Key Concepts

### The Inpainting Problem
Given sparse point measurements of ocean velocity (u, v components), reconstruct the full 2D velocity field. The "mask" represents known measurement locations; the model must predict velocities everywhere else.

### Diffusion Models (DDPM)
Denoising Diffusion Probabilistic Models progressively add noise to data and learn to reverse this process. For inpainting, the known regions are preserved while the model iteratively denoises the unknown regions, conditioned on the sparse measurements.

### Physics-Informed Constraints

**Divergence-Free Flow**: Ocean currents are approximately incompressible (∇·v ≈ 0). The project implements physics-informed noise and loss functions that encourage divergence-free predictions.

**Helmholtz-Hodge Decomposition**: Decomposes vector fields into:
- Divergence-free (rotational) component - captures vortices, eddies
- Curl-free (irrotational) component - captures sources/sinks

Used to project predictions onto physically plausible flow fields.

---

## RAPID Next Steps

### Phase R - Rationale (Current)
- [ ] Create Domain Specification document
- [ ] Document stakeholder requirements
- [ ] Define success criteria and metrics
- [ ] Identify technical risks

### Phase A - Architecture
- [ ] Create Logical Design document
- [ ] Document component relationships
- [ ] Create Architecture Decision Records (ADRs)

### Phase P - Planning
- [ ] Create Implementation Design documents
- [ ] Define testing strategy
- [ ] Plan iteration milestones

### Phase I/D - Implementation
- [ ] Validate existing code against specifications
- [ ] Implement missing features
- [ ] Complete testing and documentation

---

## References

- RAPID Method: `../../RAPID/cp-rapid-ai-1/README.md`
- Quick Reference: `../../RAPID/cp-rapid-ai-1/method/00-Quick-Reference.md`
- Getting Started: `../../RAPID/cp-rapid-ai-1/guides/getting-started.md`

---

*Last Updated: January 5, 2026*
