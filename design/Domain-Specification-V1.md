# DDPM Ocean Current Inpainting - Domain Specification

**Version**: 1.0  
**Date**: 2026-01-05  
**Author**: Jeff Caley (PLU)  
**Status**: Draft  
**Project**: NSF CAIG - Collaborative Research: Characterization of 3D Flow Structure in Transient Headland Eddies

## Document History

| Version | Date       | Author     | Changes       |
| ------- | ---------- | ---------- | ------------- |
| 1.0     | 2026-01-05 | Jeff Caley | Initial draft |

---

## 1. Executive Summary

This project develops a learning-based digital twin using Denoising Diffusion Probabilistic Models (DDPMs) to predict ocean currents from sparse in-situ measurements collected by Autonomous Underwater Vehicles (AUVs). Traditional hydrodynamic models (COAWST/ROMS) require 3 hours on 288 CPUs to simulate 6 days of ocean dynamics—far too slow for real-time adaptive sampling. By treating current prediction as an **inpainting problem**, where sparse AUV measurements serve as known constraints and the DDPM reconstructs the complete 3D velocity field, we enable real-time decision-making for coordinated multi-AUV sampling of transient headland eddies. The physics-constrained DDPM will maintain physical consistency (approximate incompressibility) while running efficiently on embedded AUV hardware, enabling unprecedented characterization of small-scale flow structures critical for understanding coral larvae dispersal.

---

## 2. Problem Statement

### 2.1 Current State

**Hydrodynamic Modeling Approach:**
- High-resolution coastal circulation models (COAWST/ROMS) simulate 3D velocity fields at 50m horizontal resolution with 25 vertical layers
- St. John model covers 130km × 100km around the Virgin Islands
- Computational cost: ~3 hours on 288 CPUs for 6-day simulation
- Models run in hindcast mode, updated once or twice daily at best
- Cannot provide real-time guidance for adaptive AUV sampling

**Observational Limitations:**
- AUVs collect sparse point measurements using Nortek DVL-1000 Water Track
- Stationary moorings and satellites cannot achieve 4D coverage at required resolution
- No existing system can predict full flow field from sparse real-time observations
- Gap between model update frequency (hours) and eddy evolution timescale (minutes to hours)

### 2.2 Research Problems

**Scientific Problems:**
- 3D structure of tidally-driven headland eddies is poorly characterized
- Convergence/divergence streaks within eddies have horizontal scales of 100-200m (at or below model resolution)
- Mechanisms forming and dissipating these streaks remain unknown
- Impact of streaks on coral larvae dispersal not quantified

**Technical Problems:**
- Hydrodynamic models too computationally expensive for real-time use
- No compact predictive model exists for onboard AUV deployment
- Existing ML approaches lack physical consistency guarantees
- Generative models prone to hallucination in unstructured environments

### 2.3 Target State

A trained DDPM digital twin that:
1. Accepts sparse velocity measurements from AUV sensors as input (known "pixels")
2. Inpaints the complete 3D velocity field in real-time (seconds to minutes)
3. Maintains physical consistency (divergence-free flow, ∇·v ≈ 0)
4. Runs efficiently on embedded AUV hardware
5. Provides uncertainty estimates to guide adaptive sampling decisions
6. Enables coordinated multi-AUV observation of transient headland eddies

### 2.4 Gap Analysis

| Current | Target | Gap |
|---------|--------|-----|
| 3-hour model runtime | Seconds inference | ~10,000× speedup needed |
| Shore-based computation | Onboard AUV | Compact model architecture |
| Dense model output | Sparse observations → dense prediction | Inpainting capability |
| No physical guarantees | Divergence-free output | Physics-constrained training |
| Single prediction | Uncertainty quantification | Ensemble or probabilistic output |

---

## 3. Domain Context

### 3.1 System Ecosystem

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         SHORE-BASED SYSTEMS                              │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐  │
│  │   COAWST/ROMS    │    │  Training Data   │    │   Mission        │  │
│  │ Hydrodynamic     │───▶│   Repository     │───▶│   Planning       │  │
│  │    Model         │    │  (17k+ samples)  │    │                  │  │
│  └──────────────────┘    └──────────────────┘    └──────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    Trained Model   │   Acoustic Modem
                    Deployment      ▼   Communication
┌─────────────────────────────────────────────────────────────────────────┐
│                           AUV SYSTEMS                                    │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐  │
│  │  Nortek DVL-1000 │    │   DDPM Digital   │    │   Adaptive       │  │
│  │  Water Track     │───▶│      Twin        │───▶│   Sampling       │  │
│  │  (sparse obs)    │    │   (inpainting)   │    │   Planner        │  │
│  └──────────────────┘    └──────────────────┘    └──────────────────┘  │
│                                                            │            │
│                          ┌─────────────────────────────────┘            │
│                          ▼                                              │
│                    Navigation to                                        │
│                    Areas of Interest                                    │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Stakeholders

| Stakeholder | Role | Interest |
|-------------|------|----------|
| Jeff Caley (PLU) | PI - ML/DDPM Component | Physics-constrained inpainting, embedded deployment |
| MIT-WHOI Collaborators | PIs - Oceanography & AUV | Hydrodynamic modeling, AUV integration, adaptive sampling |
| NSF | Funder | Novel AI-oceanography integration, broader impacts |
| PhD Student (MIT-WHOI) | Researcher | System integration, field experiments |
| Summer Students (WHOI/PLU) | Researchers | Component development, experiments |
| Coral Reef Conservation | End Users | Larvae dispersal prediction, reef restoration |
| Oceanography Community | End Users | Methods for studying dynamic coastal processes |

### 3.3 Key Entities

| Entity | Description | Relationships |
|--------|-------------|---------------|
| **Headland Eddy** | Tidally-driven rotating flow feature formed at coastal headlands | Contains Convergence Streaks; affects Particle Dispersal |
| **Convergence Streak** | Small-scale (100-200m) bands of flow convergence/divergence within eddies | Part of Headland Eddy; traps Coral Larvae |
| **Velocity Field** | 3D vector field (u, v, w) describing ocean currents | Sampled by AUV; predicted by DDPM; simulated by ROMS |
| **Sparse Observation** | Point measurement of velocity from DVL | Input to DDPM; collected by AUV |
| **Inpainted Field** | Complete velocity field reconstructed from sparse observations | Output of DDPM; guides Adaptive Sampling |
| **AUV** | Autonomous Underwater Vehicle with DVL and onboard compute | Collects Observations; runs DDPM; executes Sampling Plan |
| **Digital Twin** | Compact ML model approximating full hydrodynamic simulation | Trained on ROMS output; deployed on AUV |

### 3.4 Terminology

| Term | Definition |
|------|------------|
| **Inpainting** | Reconstructing missing regions of an image/field given known regions |
| **DDPM** | Denoising Diffusion Probabilistic Model - generative model that iteratively denoises samples |
| **Divergence-free** | Vector field with ∇·v = 0; implies incompressible flow (mass conservation) |
| **Helmholtz-Hodge Decomposition** | Decomposition of vector field into curl-free and divergence-free components |
| **DVL** | Doppler Velocity Log - acoustic sensor measuring water velocity |
| **Water Track** | DVL mode measuring velocity of water relative to vehicle |
| **ROMS** | Regional Ocean Modeling System - numerical ocean circulation model |
| **COAWST** | Coupled Ocean-Atmosphere-Wave-Sediment Transport modeling system |
| **Okubo-Weiss Parameter** | Scalar field W identifying eddies where W < threshold |
| **Hallucination** | ML model producing plausible-looking but physically impossible output |

### 3.5 Study Site

**Location**: Ram Head, St. John, U.S. Virgin Islands

**Characteristics**:
- Tidally-driven headland eddies form on alternating sides during tidal cycles
- Shallow coral reef environment with complex bathymetry
- Convergence streaks with 100-200m horizontal scale
- Pronounced 3D structure with vertical tilting
- Eddy evolution timescale: hours
- Streak evolution timescale: minutes to hours

---

## 4. Research Objectives

### 4.1 Primary Objectives (This Component)

| ID | Objective | Success Metric |
|----|-----------|----------------|
| **O1** | Train DDPM to inpaint 3D velocity fields from sparse observations | Reconstruction error < threshold on held-out test set |
| **O2** | Enforce physical consistency through physics-constrained training | Output divergence ‖∇·v‖ within acceptable bounds |
| **O3** | Achieve real-time inference for embedded deployment | Inference time < decision timescale (target: seconds) |
| **O4** | Provide uncertainty quantification | Calibrated uncertainty correlates with actual error |

### 4.2 Secondary Objectives

| ID | Objective | Success Metric |
|----|-----------|----------------|
| **O5** | Generalize to unseen tidal conditions | Performance on out-of-distribution test cases |
| **O6** | Support variable observation sparsity | Graceful degradation with fewer measurements |
| **O7** | Enable temporal forecasting | Short-horizon prediction accuracy |

### 4.3 Related Project Hypotheses

From the NSF proposal, this component directly addresses:

- **H2.1**: Generative AI models can be effectively trained to predict 3D fluid flows with a combination of synthetic data and hydrodynamic model output.
- **H2.2**: Encoding physically-guided constraints into the generative model architecture, model training, and loss functions will mitigate hallucination and guarantee physical consistency.
- **H2.3**: Learned digital twins can be used to augment hydrodynamic modeling for the purposes of characterizing transient flow dynamics.

### 4.4 Out of Scope

- Full hydrodynamic model development (provided by collaborators)
- AUV hardware/navigation systems
- Acoustic communication protocols
- Multi-AUV coordination algorithms (separate component)
- Self-Organizing Map (SOM) approach (alternative being evaluated by collaborators)
- Real-time data assimilation with ROMS

---

## 5. Functional Requirements

### Priority Levels
- **P1**: Must have for initial demonstration
- **P2**: Should have for field deployment
- **P3**: Nice to have, future enhancement

### Requirements

| ID | Requirement | Priority | Rationale |
|----|-------------|----------|-----------|
| **FR-001** | Accept sparse 2D velocity observations (u, v) as input mask | P1 | Core inpainting input |
| **FR-002** | Output complete 2D velocity field at model resolution | P1 | Core inpainting output |
| **FR-003** | Support configurable observation sparsity patterns | P1 | Real observations are irregularly spaced |
| **FR-004** | Train on ROMS hydrodynamic model output | P1 | Primary training data source |
| **FR-005** | Implement divergence-free noise during diffusion | P1 | H2.2 - physical consistency |
| **FR-006** | Implement physics-informed loss function | P1 | H2.2 - physical consistency |
| **FR-007** | Extend to 3D velocity fields (u, v, w) with depth layers | P2 | Full 3D characterization |
| **FR-008** | Provide per-pixel uncertainty estimates | P2 | Guide adaptive sampling |
| **FR-009** | Support weak prior from coarse model prediction | P2 | Constrain inpainting |
| **FR-010** | Export model for embedded deployment (ONNX/TorchScript) | P2 | AUV deployment |
| **FR-011** | Support temporal conditioning (previous timestep) | P3 | Temporal consistency |
| **FR-012** | Support multi-resolution output | P3 | Computational efficiency |

---

## 6. Non-Functional Requirements

| ID | Category | Requirement | Target |
|----|----------|-------------|--------|
| **NFR-001** | Performance | Inference time for single 2D field | < 1 second on GPU, < 10 seconds on CPU |
| **NFR-002** | Performance | Training time for full model | < 24 hours on single GPU |
| **NFR-003** | Accuracy | Velocity reconstruction RMSE | < 0.05 m/s (TBD based on data) |
| **NFR-004** | Physics | Output divergence magnitude | ‖∇·v‖ < 10⁻⁵ s⁻¹ (TBD) |
| **NFR-005** | Scalability | Support grid sizes up to | 256 × 256 (higher with tiling) |
| **NFR-006** | Memory | Model size for deployment | < 100 MB |
| **NFR-007** | Memory | Inference memory footprint | < 2 GB GPU RAM |
| **NFR-008** | Robustness | Handle missing observations gracefully | No NaN/Inf outputs |
| **NFR-009** | Reproducibility | Deterministic inference option | Seed-controlled sampling |

---

## 7. Use Cases

### UC-001: Train DDPM on Hydrodynamic Model Output

**Actor**: Researcher  
**Preconditions**: ROMS model output available as training data  
**Trigger**: Initiate training run with configuration

**Main Flow**:
1. Load ROMS velocity field snapshots from training dataset
2. Apply standardization/normalization
3. For each training epoch:
   a. Sample batch of complete velocity fields
   b. Generate random observation masks (sparse sampling)
   c. Apply forward diffusion with divergence-free noise
   d. Train UNet to denoise while preserving known observations
   e. Compute physics-informed loss (MSE + divergence penalty)
4. Save model checkpoint
5. Evaluate on validation set

**Postconditions**: Trained model checkpoint saved; validation metrics logged

**Requirements Traced**: FR-001, FR-002, FR-004, FR-005, FR-006, FR-007

---

### UC-002: Inpaint Velocity Field from Sparse Observations

**Actor**: AUV System / Researcher  
**Preconditions**: Trained DDPM model loaded; sparse observations available  
**Trigger**: New observations collected or inpainting requested

**Main Flow**:
1. Receive sparse velocity observations (u, v at known locations)
2. Construct observation mask indicating known/unknown pixels
3. Initialize noisy field (pure noise or weak prior)
4. For each diffusion timestep (reverse process):
   a. Predict denoised field using UNet
   b. Resample known observations to preserve constraints
   c. Add appropriate noise for next step
5. Apply final divergence-free projection (optional)
6. Return complete velocity field

**Alternative Flows**:
- If uncertainty requested: Run multiple samples, compute statistics
- If prior available: Initialize from prior instead of pure noise

**Postconditions**: Complete velocity field returned; matches observations at known locations

**Requirements Traced**: FR-001, FR-002, FR-003, FR-009

---

### UC-003: Evaluate Physical Consistency

**Actor**: Researcher  
**Preconditions**: Inpainted velocity field available  
**Trigger**: Post-inpainting validation

**Main Flow**:
1. Compute divergence field: ∇·v = ∂u/∂x + ∂v/∂y
2. Compute divergence statistics (mean, max, RMS)
3. Compare against physically acceptable thresholds
4. Flag regions of high divergence
5. Optionally apply Helmholtz-Hodge decomposition to extract divergence-free component

**Postconditions**: Physical consistency metrics computed and logged

**Requirements Traced**: FR-005, FR-006, NFR-004

---

### UC-004: Deploy Model for Real-Time Inference

**Actor**: AUV Integration Engineer  
**Preconditions**: Trained model validated; target hardware specified  
**Trigger**: Deployment preparation

**Main Flow**:
1. Export model to deployment format (TorchScript/ONNX)
2. Optimize for target hardware (quantization if needed)
3. Benchmark inference time on target
4. Validate output matches PyTorch reference
5. Package with inference wrapper

**Postconditions**: Deployable model artifact ready for AUV integration

**Requirements Traced**: FR-011, NFR-001, NFR-006, NFR-007

---

## 8. Integration Points

| System | Direction | Data/Events | Format | Status |
|--------|-----------|-------------|--------|--------|
| ROMS Hydrodynamic Model | Inbound | Training velocity fields | NetCDF / MAT | Existing |
| DVL Observations | Inbound | Sparse velocity measurements | Real-time stream | New |
| Adaptive Sampling Planner | Outbound | Inpainted field + uncertainty | Tensor | New |
| Model Evaluation Pipeline | Bidirectional | Metrics, visualizations | JSON, PNG | New |
| AUV Onboard Computer | Deployment | Inference requests/responses | TorchScript | New |

---

## 9. Constraints & Assumptions

### 9.1 Technical Constraints

| Constraint | Reason | Impact |
|------------|--------|--------|
| Python/PyTorch ecosystem | Existing codebase, team expertise | Architecture choices |
| Single GPU training | Available hardware | Model size limits |
| 2D fields initially | Complexity management | 3D extension is P2 |
| 50m model resolution | ROMS grid | Minimum feature scale |

### 9.2 Data Constraints

| Constraint | Reason | Impact |
|------------|--------|--------|
| Training data from single site (Ram Head) | Project scope | Generalization limits |
| ~17,000 hourly snapshots | Available ROMS output | Training data volume |
| No real AUV observations yet | Field campaigns future | Synthetic evaluation only initially |

### 9.3 Assumptions

| Assumption | If Wrong... |
|------------|-------------|
| Ocean flow approximately incompressible at this scale | Divergence-free constraint inappropriate; relax to soft penalty |
| ROMS output captures true flow statistics | Model-trained DDPM won't match real observations; need transfer learning |
| Sparse observations sufficient for accurate inpainting | Need denser sampling or stronger priors |
| Diffusion inpainting converges reliably | May need alternative conditioning approaches |
| GPU available for development/training | CPU-only significantly slower; may need cloud resources |

---

## 10. Success Criteria

### 10.1 Acceptance Criteria

- [ ] DDPM successfully trained on ROMS velocity field data
- [ ] Inpainting produces visually plausible flow fields from 10% observed pixels
- [ ] Reconstruction RMSE < baseline (linear interpolation)
- [ ] Output divergence within physically acceptable bounds
- [ ] Inference completes in < 10 seconds on reference hardware
- [ ] Model exports successfully to deployment format
- [ ] Code passes unit tests with >80% coverage

### 10.2 Key Performance Indicators

**Primary Comparison**: DDPM reconstruction quality vs **Gaussian Process (GP)** interpolation (current state-of-the-art baseline)

| KPI | Baseline (GP) | Target (DDPM) | Measurement Method |
|-----|---------------|---------------|-------------------|
| **Normalized MSE** | GP fill MSE | DDPM < GP | `calculate_mse()` with normalization over masked region |
| **Percent Error** | GP percent error | DDPM < GP | `calculate_percent_error()` - \|pred - true\| / true |
| **Angular Error** | GP angular error | DDPM < GP | `save_angular_error_heatmap()` - angle between pred/true vectors (degrees) |
| **Normalized Magnitude Difference** | GP mag diff | DDPM < GP | `save_magnitude_difference_heatmap()` - \|mag₁ - mag₂\| / avg_mag |
| **Scaled Error Magnitude** | GP scaled error | DDPM < GP | Error vector magnitude / true vector magnitude |

**Evaluation Dimensions**:

| Dimension | Method | Purpose |
|-----------|--------|--------|
| MSE vs Mask Coverage % | Scatter plot | How does error scale with % of field being predicted? |
| MSE vs Distance to Seen Pixel | Scatter plot | How does error grow with distance from observations? |
| Per-pixel error heatmaps | Spatial visualization | Where are errors concentrated? |

**Test Protocol** (from `ModelInpainter`):
- Compare DDPM and GP on same masked images
- Vary mask patterns (different `MaskGenerator` types)
- Vary number of observation lines (`num_lines`)
- Vary resample steps during diffusion
- Crop to ocean region (44×94) excluding land
- Log results to CSV: `model_name, image_num, mask_type, num_lines, resample_steps, mse_ddpm, mse_gp, mask_percentage, avg_distance`

### 10.3 Evaluation Dataset

- **Training**: 80% of ROMS hourly snapshots (~13,600 samples)
- **Validation**: 10% for hyperparameter tuning (~1,700 samples)  
- **Test**: 10% held out for final evaluation (~1,700 samples)
- **Temporal split**: Ensure test set includes unseen tidal conditions

---

## 11. Risk Assessment

| Risk | Likelihood | Impact | Mitigation | Iteration Type |
|------|------------|--------|------------|----------------|
| Physics constraints degrade reconstruction accuracy | Medium | High | Balance MSE vs divergence loss weights; ablation study | Technical Spike |
| Model doesn't generalize to real observations | Medium | High | Plan for fine-tuning with field data; synthetic→real domain adaptation | Technical Spike |
| Inference too slow for embedded deployment | Low | High | Model compression, quantization, architecture search | Technical Spike |
| Training data insufficient | Low | Medium | Data augmentation, synthetic pre-training | Discovery Spike |
| Diffusion sampling unstable | Low | Medium | Tune noise schedule, number of steps | Technical Spike |
| 3D extension significantly harder | Medium | Medium | Prioritize 2D; 3D as stretch goal | Foundation |

### Recommended Iteration Approach

**Technical Spikes Needed**:
1. **Physics-constrained noise**: Validate divergence-free noise implementation
2. **Inpainting baseline**: Establish performance with standard DDPM before physics constraints
3. **Inference optimization**: Profile and optimize for target latency

**Foundation Iteration**:
- Core DDPM training pipeline with configurable loss functions
- Evaluation framework with physics metrics

**Feature Iterations**:
- 2D inpainting with physics constraints
- Uncertainty quantification
- 3D extension
- Deployment optimization

---

## 12. Data Specification

### 12.1 Training Data Source

**File**: `stjohn_hourly_5m_velocity_ramhead_v2.mat`  
**Location**: `data/rams_head/`  
**Format**: MATLAB .mat file

**Variables** (expected):
- `u`: Eastward velocity component (m/s)
- `v`: Northward velocity component (m/s)
- `lat`, `lon`: Coordinate grids
- `time`: Timestamp for each snapshot
- `depth`: Depth levels (if 3D)

### 12.2 Data Statistics (from data.yaml)

| Statistic | Value |
|-----------|-------|
| u_training_mean | -0.0693 m/s |
| u_training_std | 0.1358 m/s |
| v_training_mean | -0.0324 m/s |
| v_training_std | 0.0890 m/s |
| mag_mean | 0.1447 m/s |

### 12.3 Preprocessing

- **Standardization**: Z-score normalization using training statistics
- **Alternatives**: Max-magnitude normalization, unit scaling
- **Masking**: Random sparse masks for inpainting training

---

## 13. Approvals

| Role | Name | Date | Status |
|------|------|------|--------|
| Lead Researcher | Jeff Caley | 2026-01-05 | Draft |
| Technical Review | | | Pending |
| Collaborator Review | | | Pending |

---

## Appendix A: Related Work

### Diffusion Models for Scientific Data
- Ho et al. (2020) - Denoising Diffusion Probabilistic Models [38]
- Score-based generative models for physical systems
- Physics-informed neural networks (PINNs)

### Ocean Current Prediction
- Self-Organizing Maps for flow field clustering [53, 54, 55]
- Neural network ocean emulators
- Data assimilation methods

### Inpainting Approaches
- RePaint: Inpainting using pre-trained diffusion models [61, 62]
- Partial convolutions for irregular masks
- Conditional generation with known constraints

---

## Appendix B: Configuration Reference

See `data.yaml` for current training configuration including:
- Noise function options: `gaussian`, `div_free`, `hh_decomp_div_free`
- Loss function options: `mse`, `physical` with configurable weights
- Diffusion parameters: `noise_steps`, `min_beta`, `max_beta`
- Inpainting parameters: `resample_nums`, `use_comb_net`

---

*Document follows RAPID Method Phase 1 (Rationale) template*  
*Reference: extern/cp-rapid-ai/method/templates/domain-specification.md*
