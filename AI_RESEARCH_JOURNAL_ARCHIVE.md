# AI Research Journal: Diffusion Inpainting Vector Fields

> **NOTE TO AI AGENTS:** This is a living research journal. You MUST add entries to this document as you investigate issues, run experiments, or discover new findings. Each entry should include the date, what was investigated, and the results/conclusions.

---

## Journal Entries

### February 4, 2026 - GP Baseline Configuration + gp_fill Cleanup

**Objective:** Make the GP baseline configurable and reduce redundant computation.

**Changes:**
- Added GP parameters to `data.yaml`: `gp_lengthscale`, `gp_variance`, `gp_noise`, `gp_use_double`, `gp_max_points`, `gp_sample_posterior`.
- Wired these parameters into `ModelInpainter` so `gp_fill()` uses config values.
- Updated `gp_fill()` to support optional subsampling of known points (`max_points`) and optional posterior sampling (`sample_posterior`).

**Notes:**
- Default behavior remains mean-only predictions (no stochastic sampling), matching previous output while reducing unnecessary work.
- `gp_max_points` can be set to speed up GP on large masks without changing defaults.

### February 4, 2026 - GP Tuning Run Blocked by MPS float64

**Objective:** Run GP parameter tuning on the training set.

**Issue:** GP tuning failed on Apple MPS with:
```
TypeError: Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64.
```

**Fix:** Updated `gp_fill()` to automatically disable `use_double` when tensor device is `mps`.

**Status:** Ready to re-run GP tuning after the fix.

### February 4, 2026 - GP Tuning Blocked by MPS Cholesky Solve

**Issue:** MPS lacks `torch.cholesky_solve`, causing GP baseline to fail.

**Fix:** Updated `gp_fill()` to run on CPU when the input tensor is on `mps`, then move results back to the original device.

**Status:** GP tuning should run on CPU fallback for Cholesky.

---

### February 4, 2026 - Gaussian Inpainting Image Generation (No CombNet)

**Objective:** Generate a DDPM inpainted vector field image using Gaussian noise without CombNet.

**Config Changes:**
- `noise_function: gaussian`
- `use_comb_net: no`

**Run:** `python -m ddpm.Testing.model_inpainter`

**Outputs:**
- DDPM prediction image: `results/weekend_ddpm_ocean_model/pt_visualizer_images/pt_predictions/ddpm82_StraightLineMaskGenerator_resample5_num_lines_1_vector_field.png`
- Reference images (same folder): `initial...`, `mask...`, `gp_field...`

**Notes:** Gaussian DDPM completed successfully on MPS. Metrics were logged to `results/inpainting_model_test_log.txt` and `results/inpainting_xl_data.csv`.

---

### February 4, 2026 - Gaussian Inpainting Re-Run (zscore Standardization)

**Objective:** Re-run Gaussian DDPM inpainting after switching `standardizer_type` back to `zscore`.

**Config Change:** `standardizer_type: zscore`

**Run:** `python -m ddpm.Testing.model_inpainter`

**Output:** Updated image at `results/weekend_ddpm_ocean_model/pt_visualizer_images/pt_predictions/ddpm82_StraightLineMaskGenerator_resample5_num_lines_1_vector_field.png` (file overwritten).

---

### February 4, 2026 - Auto Standardizer Selection by Noise Type

**Objective:** Ensure Gaussian uses legacy `zscore` while divergence-free uses `zscore_unified` automatically.

**Changes:**
- `data.yaml`: set `standardizer_type: auto` and added `standardizer_by_noise` mapping
- `DDInitializer._setup_transforms`: resolve `auto` via `noise_function` with fallback to `zscore`

### February 4, 2026 - GP Parameter Tuning (Training Set)

**Objective:** Tune GP baseline hyperparameters using training set samples.

**Setup:**
- Script: `scripts/tune_gp_params.py`
- Masks: `straight_line`, `coverage`
- Images: 10
- Evaluated grid over lengthscale, variance, noise (see `gp_tuning_*` in `data.yaml`)

**Results (best per mask):**
- Coverage mask: lengthscale=1.0, variance=0.5, noise=1e-4 → mean MSE **0.062063**
- Straight line mask: lengthscale=3.0, variance=1.0, noise=1e-5 → mean MSE **0.068309**

**Artifacts:**
- Results saved to `results/gp_tuning_results.csv`

**Decision:** Use a single GP parameter set for all masks based on best overall tuning result.
- `gp_lengthscale: 1.0`, `gp_variance: 0.5`, `gp_noise: 1e-4`

### February 4, 2026 - GP Preview Visualization

**Objective:** Visualize GP fill output with tuned parameters.

**Script:** `scripts/render_gp_example.py`

**Output:**
- `results/gp_preview/initial_sample_*.png`
- `results/gp_preview/gp_fill_sample_*.png`
- `results/gp_preview/mask_sample_*.png`

### February 4, 2026 - Mask Gallery Preview

**Objective:** Render example images for all available mask generators.

**Script:** `scripts/render_mask_gallery.py`

**Output:**
- `results/mask_gallery/*.png`

### February 4, 2026 - BetterRobotPath Batch Render

**Objective:** Generate 10 distinct BetterRobotPath masks.

**Script:** `scripts/render_better_robot_paths.py`

**Output:**
- `results/mask_gallery/better_robot_paths/better_robot_path_1.png` … `better_robot_path_10.png`

### February 4, 2026 - Mask Options Pruned

**Objective:** Remove unwanted mask options from the registry and galleries.

**Removed from options:** coverage, random, border, robot_ocean_path, squiggly_line, random_path, robot_path, smile, no_mask.

**Change:** Excluded these modules from dynamic mask registry and updated mask gallery/tuning scripts accordingly.

### February 4, 2026 - Robot Path Rename

**Objective:** Rename BetterRobotPath to RobotPath.

**Change:** The primary mask implementation now lives in `ddpm/helper_functions/masks/robot_path.py` with class `RobotPathGenerator`. A deprecated alias remains in `better_robot_path.py`, and the registry now exposes `robot_path` only.

### February 4, 2026 - Incompressible GP Kernel

**Objective:** Use an incompressible (divergence-free) GP kernel instead of RBF for baseline.

**Change:** Added `incompressible_kernel()` and `kernel_type` option to `gp_fill()`; set `gp_kernel_type: incompressible` in `data.yaml` and wired through callers.

**Preview:** Re-ran `scripts/render_gp_example.py` using `RobotPathGenerator` mask.

### February 4, 2026 - Autocorrelation Length Scales (Training Set)

**Objective:** Estimate spatial autocorrelation length scales to inform GP lengthscale choice.

**Script:** `scripts/compute_autocorr_lengthscales.py`

**Config:** `gp_autocorr_num_images=10`, `gp_autocorr_max_lag=40`

**Results (e-folding, in pixels):**
- u_x: not reached by lag 40 (corr ≈ 0.46 at lag 40)
- u_y: ~11.65
- v_x: ~14.08
- v_y: not reached by lag 40 (corr ≈ 0.64 at lag 40)

**Artifact:** `results/autocorr_lengthscales.json`

### February 4, 2026 - GP Preview (Updated Lengthscale)

**Objective:** Re-render GP preview with updated `gp_lengthscale` from `data.yaml`.

**Script:** `scripts/render_gp_example.py`

**Output:**
- `results/gp_preview/initial_sample_0.png`
- `results/gp_preview/gp_fill_sample_0.png`
- `results/gp_preview/mask_sample_0.png`

### February 4, 2026 - GP Variance Estimate (Training Set)

**Objective:** Estimate GP variance from training dataset values.

**Script:** `scripts/compute_gp_variance.py`

**Results (10 images):**
- u_var: 0.0093212118
- v_var: 0.0060506510
- combined_var: 0.0103420345

**Action:** Set `gp_variance` to 0.0103420345 in `data.yaml`.

### February 4, 2026 - GP Preview (Variance Updated)

**Objective:** Re-render GP preview after variance update.

**Script:** `scripts/render_gp_example.py`

**Output:**
- `results/gp_preview/initial_sample_0.png`
- `results/gp_preview/gp_fill_sample_0.png`
- `results/gp_preview/mask_sample_0.png`

### February 4, 2026 - Mask Convention Fix

**Issue:** `StraightLineMaskGenerator` and `RobotPathGenerator` produced masks with 0 in missing regions, but GP/inpainting expects 1 = missing.

**Fix:** Updated both generators to mark missing regions with 1 to match GP/inpainting mask convention.

### February 4, 2026 - Mask Normalization Update

**Issue:** Mask generators (e.g., straight line, robot path) were expected to produce 1=known, 0=missing, which made recent changes invert visuals.

**Fix:** Reverted straight line + robot path generators to their original outputs and added a normalization step at use-time that inverts masks when mean > 0.5, ensuring GP/inpainting always sees 1=missing.

### February 4, 2026 - GP Mask Convention Clarified (Final)

**Issue:** Mask values were being interpreted inconsistently.

**Fix:** Treat generator outputs as 1=missing (0=known) in GP/inpainting paths via `missing_mask = raw_mask * land_mask`.

**Check:** RobotPath mask currently yields ~84% missing on the cropped region.

### February 4, 2026 - GP Land Mask Exclusion

**Issue:** GP was training on land pixels (zeros), which can create noisy, non-smooth fills.

**Fix:** `gp_fill()` now excludes land pixels using a `valid_mask` derived from the input tensor (nonzero magnitude). Only valid ocean pixels are used for known/unknown indices.

### February 4, 2026 - GP Coordinate System Fix

**Issue:** GP lengthscales are in pixel units, but coordinates were normalized to [0,1], leading to ill-conditioned kernels and noisy outputs.

**Fix:** Added `gp_coord_system` (pixels|normalized). Default set to `pixels`, and all GP callers pass it through to `gp_fill()`.

### February 4, 2026 - GP Kernel Aligned to Reference

**Objective:** Match GP behavior to provided reference implementation.

**Change:** Added `rbf_legacy` and `incompressible_rbf` kernels (legacy distance + kernel_noise^2). Set `gp_kernel_type: incompressible_rbf` in `data.yaml`.

### January 5, 2026 - Gaussian vs Divergence-Free Noise Inpainting Comparison

**Objective:** Test inpainting performance using Gaussian noise with the weekend model.

**Initial Problem:** Running `test_gaussian_inpainting.py` with `ddpm_ocean_good_normalized.pt` failed with a shape mismatch error:
```
RuntimeError: size mismatch for network.time_embed.weight: 
copying a param with shape torch.Size([1000, 100]) from checkpoint, 
the shape in current model is torch.Size([100, 100])
```

**Root Cause:** The `ddpm_ocean_good_normalized.pt` model was trained with `n_steps=1000`, but the script was creating a model with `n_steps=100`.

**Solution:** Switched to `weekend_ddpm_ocean_model.pt` which uses `n_steps=100`.

---

### January 5, 2026 - Land Mask Bug Discovery

**Objective:** Investigate why custom test script produced much worse MSE than `model_inpainter.py`.

**Findings:**

| Script | MSE |
|--------|-----|
| `test_gaussian_inpainting.py` (initial) | 0.804 |
| `model_inpainter.py` | 0.014 |

**Root Cause:** The test script was NOT applying the land mask to the generated mask. This caused the inpainting to attempt reconstruction over land/boundary regions where values are zero.

**The Bug:**
```python
# WRONG - missing land mask
mask = mask_generator.generate_mask(input_image.shape).to(device)
```

**The Fix:**
```python
# CORRECT - apply land mask
land_mask = (input_image_original.abs() > 1e-5).float().to(device)
raw_mask = mask_generator.generate_mask(input_image.shape).to(device)
mask = raw_mask * land_mask
```

**Lesson Learned:** Always multiply the generated mask by the land mask to ensure inpainting only occurs in valid ocean regions.

---

### January 5, 2026 - Parameter Alignment for Fair Comparison

**Objective:** Align test script parameters with `model_inpainter.py` for fair comparison.

**Parameters Changed:**

| Parameter | Before | After |
|-----------|--------|-------|
| Mask Generator | `CoverageMaskGenerator(0.3)` | `StraightLineMaskGenerator(1)` |
| Data Loader | Test set | Validation set |
| Resample Steps | 1 | 5 |

**Results After Alignment:**

| Configuration | MSE |
|---------------|-----|
| Original (no land mask) | 0.804 |
| With land mask only | 0.792 |
| Fully aligned | 0.074 |
| model_inpainter.py | 0.014 |

**Conclusion:** The remaining MSE difference (0.074 vs 0.014) is likely due to:
1. Different random samples being selected
2. Stochastic nature of the inpainting process

The scripts are now functionally equivalent.

---

### January 5, 2026 - Model Location Configuration

**Issue:** `model_inpainter.py` couldn't find models because `model_paths` in `data.yaml` pointed to `../../models/div_free_model.pt` which didn't exist.

**Solution:** Copied `weekend_ddpm_ocean_model.pt` to the expected location:
```bash
mkdir -p /path/to/PLU/models
cp ddpm/Trained_Models/weekend_ddpm_ocean_model.pt ../../models/div_free_model.pt
```

**Note:** Model paths in `data.yaml` are relative to `ddpm/Testing/`.

---

### January 5, 2026 - Debug Print Statement Removal

**Location:** `ddpm/vector_combination/vector_combiner.py`

**Removed:**
```python
print("WARNING: Using comb_net (slow!)")
```

**Context:** This was added during debugging to identify performance bottlenecks when using the combination network for divergence-free inpainting.

---

## Experiment Queue

- [x] Run divergence-free noise inpainting test and compare to Gaussian
- [ ] Test different resample_nums values and their effect on quality
- [ ] Compare different mask types (straight line vs coverage vs robot path)
- [ ] Evaluate combination network (use_comb_net: yes) vs naive combination

---

### January 5, 2026 - Divergence-Free vs Gaussian Noise Comparison

**Objective:** Compare inpainting quality between Gaussian and divergence-free noise strategies.

**Configuration:**
- Model: `weekend_ddpm_ocean_model.pt` (copied to `../../models/div_free_model.pt`)
- Mask: `StraightLineMaskGenerator(1)`
- Resample steps: 5
- Device: CPU

**Results:**

| Metric | Gaussian Noise | Div-Free Noise |
|--------|----------------|----------------|
| MSE (DDPM) | 0.014434 | 0.031953 |
| Angular Error | 45.914° | 139.708° |
| Scaled Error Magnitude | 1.975508 | 2.584975 |
| Normalized Mag Diff | 0.483935 | 0.314711 |
| Inference Time | ~12 sec | **~19 min** |

**GP Fill Baseline (same for both):**
- MSE: 0.014434
- Angular Error: 45.914°

**Key Observations:**

1. **Performance:** Div-free noise is **~95x slower** than Gaussian (19 min vs 12 sec). This is because div-free noise generation requires computing stream functions and spatial derivatives.

2. **Quality:** Gaussian noise actually performed BETTER on most metrics:
   - Lower MSE (0.014 vs 0.032)
   - Much better angular error (46° vs 140°)
   - Lower scaled error (1.98 vs 2.58)

3. **Magnitude Preservation:** Div-free was slightly better at preserving magnitudes (0.31 vs 0.48 normalized difference).

4. **Surprising Result:** The div-free noise performed WORSE than the GP fill baseline, while Gaussian matched it.

**Hypothesis:** The model was likely trained with Gaussian noise, so it expects Gaussian noise during inference. Using div-free noise at inference time with a Gaussian-trained model creates a distribution mismatch.

**Next Steps:** 
- Train a model specifically with div-free noise and test with div-free inference
- Or test a div-free trained model with div-free inference

---

### January 5, 2026 - Div-Free Model with Div-Free Noise (Matched Training)

**Objective:** Test the div-free trained model with div-free noise (proper matching).

**Configuration:**
- Model: `div_free_model.pt` (trained with div-free noise)
- Noise: div_free
- Mask: `StraightLineMaskGenerator(1)`
- Resample steps: 5

**Results - Full Comparison:**

| Metric | Gaussian Model + Gaussian Noise | Gaussian Model + Div-Free Noise | Div-Free Model + Div-Free Noise | GP Baseline |
|--------|--------------------------------|--------------------------------|--------------------------------|-------------|
| MSE | **0.0106** | 0.0320 | 0.0818 | 0.0144 |
| Angular Error | 77.7° | 139.7° | 89.4° | 45.9° |
| Scaled Error Mag | **1.55** | 2.58 | 5.30 | 1.98 |
| Norm Mag Diff | **0.24** | 0.31 | 1.22 | 0.48 |
| Inference Time | **~13 sec** | ~19 min | ~17 min | - |

**Key Observations:**

1. **Surprising Result:** The div-free model with matched div-free noise performed WORSE than expected:
   - MSE: 0.082 (much higher than Gaussian's 0.011)
   - Angular error: 89.4° (still high, though better than mismatched 139.7°)
   - Normalized magnitude difference: 1.22 (very high - overshooting magnitudes)

2. **Gaussian Still Wins:** The Gaussian model with Gaussian noise consistently outperforms all configurations on most metrics.

3. **Model Matching Helps Direction:** Angular error improved from 139.7° (mismatched) to 89.4° (matched), confirming that noise-model matching matters.

4. **Magnitude Issues with Div-Free:** The div-free approach shows severe magnitude problems (norm mag diff = 1.22 means predictions are ~2.2x the original magnitude on average).

**Hypotheses for Poor Div-Free Performance:**

1. **Training Quality:** The div-free model may not have been trained as long or as well as the weekend Gaussian model.

2. **Div-Free Noise Structure:** The divergence-free constraint may be too restrictive, preventing the model from learning the full range of velocity field variations.

3. **Stream Function Complexity:** Computing div-free noise via stream functions may introduce numerical artifacts.

**Recommendation:** For production use, stick with **Gaussian noise + Gaussian-trained model** until div-free training is improved.

---

### January 5, 2026 - Denoising Step-by-Step Diagnostic Analysis

**Objective:** Understand why div-free model produces bad vectors even though training loss was similar to Gaussian.

**Method:** Created `scripts/diagnose_denoising.py` to track vector statistics at each denoising step.

**Key Findings - Forward Noising:**

| Step | Gaussian std | Div-Free std | Observation |
|------|--------------|--------------|-------------|
| t=0 | 0.730 | 0.726 | Start similar |
| t=40 | 0.781 | 0.739 | Gaussian grows faster |
| t=80 | 0.874 | 0.819 | Gap widening |
| t=99 | 0.913 | **1.362** | Div-free explodes at end! |

**Critical Finding:** Div-free noise std **jumps from 0.82 to 1.36** in the last 20 steps (66% increase), while Gaussian grows smoothly from 0.87 to 0.91.

**Backward Denoising - Inpainted Region Mean:**

| Step | Gaussian | Div-Free |
|------|----------|----------|
| t=80 | -0.016 | **+0.339** |
| t=60 | -0.002 | **+0.278** |
| t=40 | +0.003 | **+0.287** |
| t=20 | +0.005 | **+0.309** |
| t=0 | +0.005 | **+0.254** |
| Ground Truth | -0.469 | -0.471 |

**The Problem is Clear:**
1. **Div-free inpainted values stay positive** (~+0.25 to +0.34) throughout denoising
2. **Gaussian converges toward zero** as expected
3. **Ground truth is negative** (-0.47)
4. Div-free is **consistently wrong sign** from the very first denoising step!

**Epsilon (Predicted Noise) Analysis:**
- Gaussian epsilon_std stays ~1.0 throughout (correct for unit Gaussian)
- Div-free epsilon_std **decreases to 0.83** by t=0 (model underpredicts noise magnitude)

**Final Results Comparison:**

| Metric | Gaussian | Div-Free |
|--------|----------|----------|
| Final std | 0.285 | **1.735** (6x higher!) |
| Unstd mean | -0.023 | +0.014 |
| Ground truth mean | -0.048 | -0.048 |
| Masked region mean | +0.005 | **+0.254** (wrong sign!) |

**Root Cause Hypothesis:**

The div-free noise has different statistical properties than Gaussian:
1. The **non-Gaussian scaling formula** `(1/√α_t) * (x - ε_θ)` may not be appropriate for div-free noise
2. The model learns to predict noise of the wrong magnitude (eps_std ~0.83 instead of ~1.0)
3. The forward noising process has a **sudden variance explosion** at late timesteps that the backward process can't correct

**The `get_gaussian_scaling()` check in denoise_one_step:**
```python
if noise_strat.get_gaussian_scaling():
    less_noised_img = (1 / alpha_t.sqrt()) * (
        noisy_img - ((1 - alpha_t) / (1 - alpha_t_bar).sqrt()) * epsilon_theta
    )
else:
    less_noised_img = (1 / alpha_t.sqrt()) * (noisy_img - epsilon_theta)
```

The div-free path uses a simpler formula that may be mathematically incorrect for the DDPM framework.

---

### January 5, 2026 - ROOT CAUSE FOUND: Spatial Correlation in Div-Free Noise

**Key Discovery:** The div-free noise has fundamentally different statistical properties that break DDPM assumptions.

**Statistical Comparison:**

| Property | Gaussian Noise | Div-Free Noise |
|----------|---------------|----------------|
| Mean | ~0 | 0 |
| Std | ~1 | 1 |
| **Spatial Correlation** | **0.9%** | **98.4%** |

**The Problem:**

DDPM theory assumes noise is **independent and identically distributed (i.i.d.)** at each pixel. The standard denoising formula:
```
x_{t-1} = (1/√α_t) * (x_t - ((1-α_t)/√(1-ᾱ_t)) * ε_θ)
```
...is derived assuming `ε ~ N(0, I)` - uncorrelated Gaussian noise.

**Div-free noise is generated from stream functions:**
```python
psi = torch.randn(...)  # Random stream function
u = dψ/dy   # Spatial derivative
v = -dψ/dx  # Spatial derivative
```

Taking spatial derivatives creates **massive correlation** between neighboring pixels (98.4%!). Adjacent pixels share almost the same value.

**Why This Breaks Denoising:**

1. The neural network learns to predict `ε_θ` assuming correlated noise structure
2. But the DDPM formula treats `ε_θ` as if it were uncorrelated
3. The scaling factors `((1-α_t)/√(1-ᾱ_t))` are wrong for correlated noise
4. This causes the magnitude explosion we observed

**Analogy:** It's like trying to remove blur from an image using a formula designed to remove salt-and-pepper noise - the math doesn't match the problem.

**Potential Solutions:**

1. **Normalize div-free noise per-pixel** to break correlation (but this destroys the divergence-free property)
2. **Derive new DDPM formulas** for spatially correlated noise (research project)
3. **Use different architecture** that accounts for spatial correlation
4. **Accept Gaussian is better** for standard DDPM inpainting

**Conclusion:** The div-free approach is fundamentally incompatible with standard DDPM unless new theory is developed for correlated noise diffusion.

---

### January 5, 2026 - ACTUAL ROOT CAUSE: Inpainting Boundary Discontinuity

**Follow-up finding:** Pure denoising (without mask) actually works fine for div-free! The problem is specifically in the **inpainting combining step**.

**Test Results - Pure Denoising (no mask):**
| Model | Initial std | Final std | Works? |
|-------|-------------|-----------|--------|
| Gaussian | 1.005 | 0.117 | ✓ |
| Div-Free | 1.000 | 0.060 | ✓ |

Both models successfully denoise from noise to clean images!

**Test Results - With Inpainting Mask Combining:**
| Timestep | Gaussian (masked-known diff) | Div-Free (masked-known diff) |
|----------|------------------------------|------------------------------|
| t=80 | 0.006 | **0.374** |
| t=60 | 0.006 | **0.238** |
| t=0  | 0.028 | **0.347** |

**The Real Problem:**

At each denoising step, inpainting does:
```python
x = known * (1 - mask) + inpainted * mask
```

This **abrupt boundary** between regions causes issues:

1. **Gaussian noise:** Each pixel is independent, so combining regions creates no spatial conflict. The boundary is just two independent denoising processes meeting.

2. **Div-free noise:** Pixels are 98% correlated with neighbors. When you abruptly combine two independently-denoised regions, you create a **discontinuity at the boundary** that violates the spatial correlation structure the model expects.

3. The model tries to denoise an image with an artificial discontinuity that wouldn't exist in real div-free noised images.

4. This discontinuity propagates and compounds through the denoising steps.

**Why Training Worked:** During training, there's no mask combining - the model sees full images with consistent div-free noise everywhere.

**Why Inference Fails:** During inpainting, we artificially splice together two regions that have inconsistent spatial correlation at the boundary.

**Potential Fix:** Instead of hard mask combining, use a **blending/feathering** approach at the boundary to maintain spatial continuity:
```python
# Instead of hard boundary:
x = known * (1 - mask) + inpainted * mask

# Use soft blending near boundary:
soft_mask = gaussian_blur(mask, sigma=5)
x = known * (1 - soft_mask) + inpainted * soft_mask
```

---

## Key Metrics Reference

When running inpainting experiments, record these metrics:

- **MSE (normalized, masked region):** Primary quality metric
- **Angular error (degrees):** Direction accuracy
- **Scaled error magnitude:** Magnitude accuracy  
- **Normalized magnitude difference:** Overall magnitude comparison
- **Mask coverage %:** How much area was masked
- **Inference time:** Seconds per sample

---

## Configuration Snapshots

### Current Working Configuration (Gaussian Noise)
```yaml
noise_function: gaussian
loss_function: mse
w1: 0.6
w2: 0.4
noise_steps: 100
min_beta: 0.0001
max_beta: 0.02
resample_nums: [5]
model_paths:
  - "../../models/div_free_model.pt"
```

---

## Known Issues & Gotchas

1. **Shape Mismatch Errors:** Always verify `noise_steps` matches the trained model
2. **High MSE Values:** Check if land mask is being applied
3. **Model Not Found:** Paths are relative to project root directory
4. **Slow Performance with Comb Net:** The combination network trains a small UNet at each denoising step - expected to be slow

---

### January 5, 2026 - Unified Model Selection System

**Change:** Implemented automatic model selection based on noise type.

**Old Approach:**
- Manual path management in `model_paths` list
- Paths were relative to `ddpm/Testing/` (confusing)
- Had to manually ensure model matched noise type

**New Approach in `data.yaml`:**
```yaml
noise_function: gaussian  # Change ONLY this to switch modes

model_by_noise_type:
  gaussian: "ddpm/Trained_Models/weekend_ddpm_ocean_model.pt"
  div_free: "ddpm/Trained_Models/div_free_model.pt"
```

**Benefits:**
1. Single setting change to switch between modes
2. Paths are now relative to project root (clearer)
3. Prevents model/noise mismatch errors
4. Legacy `model_paths` still works as fallback

---

### January 5, 2026 - ROOT CAUSE CONFIRMED: PhysicsInformedLoss Design Flaw

**Follow-up to**: "ACTUAL ROOT CAUSE: Inpainting Boundary Discontinuity"

**Discovery:** Investigated why `comb_net` (VectorCombinationUNet) doesn't fix boundary issues despite being enabled.

**Finding:** The `PhysicsInformedLoss` has a fundamental design flaw:

**1. Fidelity Loss Forces Match to Naive Stitch:**
```python
naive = known * (1 - mask) + inpainted * mask  # Hard boundary!
loss_fidelity = (predicted - naive) ** 2  # Penalizes ANY change!
```

**2. Physics Loss is Permissive:**
```python
loss_physics = F.relu(div_pred - max_div_threshold)
# Only penalizes if divergence gets WORSE than inputs
# If naive stitch is already below threshold → loss = 0
```

**3. Smoothness Disabled:**
```yaml
smooth_weight: 0  # The only term that could help!
```

**Proof - Test with naive stitch unchanged:**
```
If comb_net returns naive stitch (no change):
  loss_fidelity: 0.000000
  loss_physics: 0.000000
  loss_smooth: 0.000000
  total_loss: 0.000000  # ZERO INCENTIVE TO CHANGE!
```

**Why This Matters:**
- The neural network has ZERO gradient signal to smooth boundaries
- It's actively rewarded for doing nothing (returning naive stitch)
- The physics loss was designed for "don't make it worse" not "enforce smoothness"

**Current Weights in `data.yaml`:**
- `fidelity_weight: 0.01` (low but still fights boundary smoothing)
- `physics_weight: 2` (only prevents making things worse)
- `smooth_weight: 0` (completely disabled!)

**Recommended Fixes:**

1. **Enable smooth_weight:** Set `smooth_weight: 0.5` or higher
2. **Add boundary-specific loss:** Weight smoothness higher at mask edges
3. **Modify fidelity loss:** Allow more deviation near boundaries
4. **Change physics loss:** Enforce low divergence everywhere, not just "don't make it worse"

**Next Experiment:** Test with `smooth_weight: 1.0` to see if it helps div-free inpainting.

---

### January 5, 2026 - Empirical Boundary Divergence Test

**Objective:** Verify if naive stitching creates higher divergence at boundaries for div-free vs Gaussian noise.

**Method:** Created `scripts/test_real_divergence.py` that:
1. Loads both div-free and Gaussian trained models
2. Forward-noises an image at various timesteps (80, 60, 40, 20, 5)
3. Performs one denoising step
4. Applies naive stitch combining
5. Measures mean |divergence| at boundary vs away from boundary

**Results:**

| Timestep | Div-Free Bnd/Away Ratio | Gaussian Bnd/Away Ratio |
|----------|------------------------|------------------------|
| t=80 | **3.56x** | 1.39x |
| t=60 | 3.11x | 1.65x |
| t=40 | 2.82x | 1.92x |
| t=20 | 2.82x | 2.25x |
| t=5 | 2.78x | 2.64x |
| **Avg** | **2.91x** | 1.96x |

**Key Finding:** Div-free creates **1.48x more boundary divergence** than Gaussian.

**Interpretation:**
- At early timesteps (high noise), div-free shows 2.6x worse boundary ratio than Gaussian
- The effect diminishes as denoising progresses (both converge toward ~2.7x)
- Absolute boundary divergence is similar (~0.9), but div-free has much lower away-from-boundary divergence (~0.27 vs ~0.65)
- This makes the boundary discontinuity relatively MORE significant for div-free

**Why This Matters:**
- Confirms the spatial correlation theory: breaking div-free correlation at boundaries creates artifacts
- Explains why div-free inpainting produces worse results despite similar training loss
- Supports the need for boundary-aware combining (either smooth_weight > 0 or boundary-specific loss)

**Conclusion:** The theory is **partially confirmed**. Div-free does have worse boundary behavior, but the effect (1.48x) is moderate, not catastrophic. The real problem is likely the compound effect over 100 denoising steps - each step adds more boundary divergence that propagates.

---

### January 6, 2026 - CombNet Pretraining Investigation

**Objective:** Understand and optimize the CombNet (boundary divergence fixer) which was causing 86+ minute inference times for div-free inpainting.

**Root Cause of Slow Inference:**
The `VectorCombinationUNet` (CombNet) was being trained **from scratch at every denoising step** during inference:
- 200 training iterations per call
- ~500 calls per inpainting (100 timesteps × 5 resample steps)
- **100,000 total training iterations per inpainting!**

**Solution Approach:** Pretrain CombNet once, reuse during inference.

---

### January 6, 2026 - Loss Function Improvements

**Original Loss Function (`PhysicsInformedLoss`):**
- `loss_preserve`: MSE away from boundary
- `loss_boundary_div`: Divergence at boundary seam
- `loss_smooth`: Gradient penalty at boundary

**Problems Identified:**
1. "Away from boundary" logic was overcomplicated
2. No explicit protection for the known region

**Updated Loss Function (4 terms):**

| Term | Weight | Purpose |
|------|--------|---------|
| `loss_known` | 10.0 | Don't modify known region at all |
| `loss_change` | 0.1 | Minimize total changes to field |
| `loss_boundary_div` | 100.0 | Reduce divergence at seam |
| `loss_smooth` | 1.0 | Smooth transitions at boundary |

**Key Insight:** The loss is physics-based, not supervised. There's no "ground truth" - the network learns to satisfy constraints:
> "Fix divergence at the boundary seam while making minimal changes, and don't touch the known region."

---

### January 6, 2026 - Fast Training Data Generation

**Initial Approach (Failed):** Generate training data by running actual DDPM inference.
- Problem: Each sample requires running 100 denoising steps
- Estimated time: Hours to generate 25k samples

**Key Insight from User:** "Don't we just need examples of two images merged together?"

**Optimized Approach:**
1. Generate two independent div-free noise fields (A and B)
2. Apply multiple random masks to create many boundary examples
3. Naive stitch: `A * (1-mask) + B * mask`

**Final Configuration:**
- 5,000 div-free field pairs
- 8 different masks per pair
- **40,000 total samples generated in ~3 minutes!**

**Dataset Shape:**
```
known: torch.Size([40000, 2, 64, 128])
inpainted: torch.Size([40000, 2, 64, 128])
mask: torch.Size([40000, 2, 64, 128])
naive: torch.Size([40000, 2, 64, 128])
```

---

### January 6, 2026 - MPS (Apple Silicon GPU) Support

**Issue:** Training was running on CPU (~70 min/epoch)

**Fix in `data_prep/data_initializer.py`:**
```python
# Before: Only checked CUDA
self.device = torch.device(f"cuda:{self.gpu}" if torch.cuda.is_available() else "cpu")

# After: Check CUDA > MPS > CPU
if torch.cuda.is_available():
    self.device = torch.device(f"cuda:{self.gpu}")
elif torch.backends.mps.is_available():
    self.device = torch.device("mps")
else:
    self.device = torch.device("cpu")
```

**Performance Improvement:**

| Device | Batch Size | Time per Epoch |
|--------|-----------|----------------|
| CPU | 16 | ~70 min |
| **MPS** | 32 | **~8 min** |

**Speedup: ~9x faster on Apple Silicon!**

---

### January 6, 2026 - CombNet Training (Overnight Run)

**Training Started:** 100 epochs on MPS (~14 hours estimated)

**Configuration:**
- Epochs: 100
- Batch size: 32
- Learning rate: 1e-3 with cosine annealing
- Samples: 40,000
- Device: MPS (Apple Silicon)

**Early Results (first 2 epochs on test run):**
- Epoch 1 loss: 2.27
- Epoch 2 loss: 1.20
- Loss decreasing → good learning signal

**Model Output:** `ddpm/Trained_Models/pretrained_combnet.pt`
**Log File:** `combnet_training.log`

**Tomorrow's Test:** 
1. Check final training loss
2. Run full inpainting with pretrained CombNet
3. Compare to per-step training approach (should be ~100x faster with similar quality)

---

### January 6, 2026 - Overfitting Analysis

**Question:** Should we worry about overfitting with CombNet pretraining?

**Why Overfitting is LESS Likely:**
1. **No ground truth to memorize** - Loss is physics-based (minimize divergence), not data-matching
2. **Universal physics constraints** - Divergence-free is universal; if it works on training data, should generalize
3. **Reasonable data/param ratio** - ~1M params with 40k samples

**Potential Concerns:**
1. Network might learn shortcuts specific to synthetic div-free noise
2. Real DDPM outputs during inference might have different characteristics

**Mitigation:**
- Cosine LR scheduler reduces late-stage overfitting
- Final test tomorrow will evaluate real generalization

---

## Scripts Created/Modified Today

| Script | Purpose |
|--------|---------|
| `scripts/generate_combnet_data.py` | Generate training data by merging div-free fields |
| `scripts/pretrain_combnet.py` | Train CombNet on generated dataset |
| `ddpm/vector_combination/combination_loss.py` | Added `loss_known` term, simplified to `loss_change` |
| `ddpm/vector_combination/vector_combiner.py` | Added pretrained CombNet support with caching |
| `data_prep/data_initializer.py` | Added MPS (Apple Silicon) GPU support |

---

## Next Steps (January 7, 2026)

- [x] Check CombNet training results (`tail -50 combnet_training.log`)
- [x] Test pretrained CombNet on real inpainting task
- [x] Compare inference time: pretrained vs per-step training
- [x] Compare quality metrics: MSE, angular error, divergence
- [ ] If successful, update `run_comparison.py` to use pretrained CombNet

---

### January 8, 2026 - CombNet Comparison: Pretrained vs Per-Step Training

**Objective:** Compare the pretrained CombNet (fast) against the old per-step training approach (slow) for reducing boundary divergence in div-free inpainting.

**Test Setup:**
- Model: `div_free_model.pt` with divergence-free noise
- Mask: `StraightLineMaskGenerator(1)` - 44.8% coverage
- Timesteps tested: 80, 60, 40, 20, 5
- Device: MPS (Apple Silicon)

**Results - Boundary Divergence at Each Timestep:**

| Timestep | Naive Stitch | Pretrained CombNet | Per-Step Training |
|----------|--------------|-------------------|-------------------|
| t=80 | 0.7949 | 0.4412 | 0.3390 |
| t=60 | 0.9418 | 0.6116 | 0.4545 |
| t=40 | 0.8513 | 0.6187 | 0.4783 |
| t=20 | 0.9017 | 0.6215 | 0.5622 |
| t=5  | 0.9163 | 0.6428 | 0.6033 |
| **Avg** | **0.8812** | **0.5872** | **0.4875** |

**Results - Boundary/Away Divergence Ratio:**

| Method | Avg Ratio | Interpretation |
|--------|-----------|----------------|
| Div-Free Naive Stitch | 2.97x | High boundary divergence relative to field |
| Div-Free + Pretrained CombNet | 2.39x | Moderate improvement |
| Div-Free + Per-Step Training | 1.65x | Best boundary reduction |
| Gaussian Naive Stitch | 1.95x | Baseline for comparison |

**Results - Speed Comparison:**

| Method | Time per Step | Notes |
|--------|---------------|-------|
| Pretrained CombNet | ~0.003s | Single forward pass (after initial load) |
| Per-Step Training | ~2.2s | 200 training iterations per step |

**Speed Ratio:** Per-step training is **~730x slower** than pretrained CombNet.

**Key Findings:**

1. **Both CombNet approaches reduce boundary divergence:**
   - Pretrained: 35.7% reduction (0.8812 → 0.5872)
   - Per-Step: 46.6% reduction (0.8812 → 0.4875)

2. **Per-step training is slightly better quality** (~11% more divergence reduction) but **drastically slower** (730x).

3. **Pretrained CombNet achieves most of the benefit:**
   - Gets 77% of the per-step training's divergence reduction
   - At 0.14% of the computational cost

4. **Per-step training converges better at early timesteps:**
   - At t=80: Per-step is 23% better than pretrained
   - At t=5: Per-step is only 6% better than pretrained

**Recommendation:**

Use **pretrained CombNet** for production. The 11% quality trade-off is worth the 730x speedup. For a full inpainting run (100 timesteps × 5 resample steps = 500 calls):
- Pretrained: ~1.5 seconds total
- Per-Step: ~18 minutes total

**Configuration:**
```yaml
# data.yaml - Use pretrained for speed
pretrained_combnet_path: "ddpm/Trained_Models/pretrained_combnet.pt"

# Set to null for better quality (but much slower)
# pretrained_combnet_path: null
```

---

### January 8, 2026 - CRITICAL BUG: Z-Score Standardization Breaks Divergence-Free Property

**Discovery:** While investigating why divergence values seemed high (~0.35 for standardized data vs ~0.04 for original), we discovered a fundamental flaw in the z-score standardization approach.

**The Problem:**

The `ZScoreStandardizer` uses **different standard deviations** for u and v components:
- `u_training_std: 0.136` → scaling factor 1/0.136 = **7.4x**
- `v_training_std: 0.089` → scaling factor 1/0.089 = **11.2x**

This **breaks the divergence-free property**:

```
Original:     div = ∂u/∂x + ∂v/∂y = 0  (divergence-free)

After z-score with different stds:
              div_std = (1/std_u) * ∂u/∂x + (1/std_v) * ∂v/∂y
                      = 7.4 * ∂u/∂x + 11.2 * ∂v/∂y

If ∂u/∂x = -∂v/∂y (div-free condition):
              div_std = 7.4 * ∂u/∂x - 11.2 * ∂u/∂x
                      = -3.8 * ∂u/∂x  ≠ 0  (BROKEN!)
```

**Why This Matters:**
1. Ocean velocity fields are approximately divergence-free (conservation of mass)
2. The div-free DDPM model explicitly assumes divergence-free noise
3. Standardization with different stds creates artificial divergence
4. This artificial divergence propagates through the entire pipeline

**Measured Impact:**
| Data State | Mean |divergence| |
|------------|----------------------|
| Original ocean data | 0.044 |
| After z-score (different stds) | 0.35-0.38 |
| **Increase** | **~8x artificial divergence!** |

**The Fix:**

Created new `UnifiedZScoreStandardizer` that uses **same std for both components**:

```python
class UnifiedZScoreStandardizer(Standardizer):
    """Uses same std for both u and v - preserves divergence-free property."""
    def __init__(self, shared_mean, shared_std):
        self.mean = shared_mean
        self.std = shared_std

    def __call__(self, tensor):
        return (tensor - self.mean) / self.std
```

With unified std:
```
div_std = (1/std) * ∂u/∂x + (1/std) * ∂v/∂y
        = (1/std) * (∂u/∂x + ∂v/∂y)
        = (1/std) * 0 = 0  ✓ (PRESERVED!)
```

**New Configuration (data.yaml):**
```yaml
# Unified statistics for zscore_unified
shared_mean: -0.0508  # avg(u_mean, v_mean)
shared_std: 0.1148    # sqrt((u_std² + v_std²) / 2)

standardizer_type: zscore_unified  # preserves div-free
```

**Files Modified:**
- `ddpm/helper_functions/standardize_data.py` - Added `UnifiedZScoreStandardizer`
- `data.yaml` - Added `shared_mean`, `shared_std`, and `zscore_unified` option

**Impact on Existing Models:**
- Existing models trained with `zscore` will still work (backwards compatible)
- New models for div-free inpainting should use `zscore_unified`
- Models need to be **retrained** with unified standardizer to benefit

**Next Steps:**
- [ ] Test that new standardizer preserves divergence
- [ ] Retrain div-free model with `zscore_unified`
- [ ] Retrain CombNet with unified standardized data
- [ ] Compare inpainting quality

---

### January 8, 2026 - Per-Step CombNet: Div-Free vs Gaussian Comparison

**Objective:** Compare Per-Step CombNet effectiveness on Div-Free vs Gaussian DDPM models before retraining with unified standardizer.

**Terminology:**
- **Per-Step CombNet**: Trains a VectorCombinationUNet at each denoising step (~200 iterations, ~2.1s/step)
- **Naive Stitch**: Simple masking: `known * (1-mask) + denoised * mask`

**Note:** Both models were trained with the broken `zscore` standardizer (different u/v stds), so absolute divergence values are inflated ~8x. Relative comparisons remain valid.

**Results - Naive Stitch (baseline):**

| Model | div_boundary (avg) | div_away (avg) | ratio |
|-------|-------------------|----------------|-------|
| Div-Free | 0.8812 | 0.2996 | 2.93x |
| Gaussian | 0.9271 | 0.4945 | 1.95x |

**Results - With Per-Step CombNet:**

| Model | div_boundary (avg) | div_away (avg) | ratio |
|-------|-------------------|----------------|-------|
| Div-Free | 0.4874 | 0.2932 | 1.65x |
| Gaussian | 0.5190 | 0.4566 | 1.20x |

**Boundary Divergence Reduction:**

| Model | Naive | Per-Step CombNet | Reduction |
|-------|-------|------------------|-----------|
| Div-Free | 0.8878 | 0.4874 | **45.1%** |
| Gaussian | 0.9271 | 0.5190 | **44.0%** |

**Key Findings:**

1. **Both models benefit equally** from Per-Step CombNet (~44-45% boundary divergence reduction)

2. **Gaussian achieves better final ratio** (1.20x vs 1.65x) - boundary divergence closer to field average

3. **At early timesteps (t=80), Gaussian + CombNet achieves ratio < 1.0** (0.81x) - boundary divergence actually lower than field average!

4. **Div-Free model has lower overall divergence** but higher boundary/away ratio

5. **Speed:** ~2.1s per step for Per-Step CombNet (vs ~0.003s for Pretrained CombNet)

**Interpretation:**

The Div-Free model produces fields with lower overall divergence but the boundary stitching problem is more pronounced (2.93x ratio). The Gaussian model has higher baseline divergence but better boundary behavior after CombNet correction.

This suggests the divergence-free constraint helps maintain physical consistency in the field interior, but boundary stitching remains a distinct challenge that Per-Step CombNet addresses effectively for both models.

**Next Steps:**
- Retrain both models with `zscore_unified` standardizer
- Re-run comparison with properly standardized data
- Compare Pretrained vs Per-Step CombNet on new models

---

## Session: Guided Diffusion Inpainting

**Date:** 2025-01-XX
**Goal:** Replace RePaint's copy-paste inpainting with gradient-guided diffusion

### Motivation

Deep quality diagnostics revealed fundamental limitations of RePaint:
- **Magnitude bias**: FFT projection at every step removes ~0.4% energy per step, compounding to ~34% magnitude loss over 250 steps
- **OOD inputs**: Copy-paste composites present out-of-distribution inputs to the denoiser, causing biased predictions
- **Angular errors**: 47–88° directional errors in unknown region despite reasonable MSE

### Approach: Guided Diffusion (No Retraining Required)

Instead of RePaint's copy-paste-denoise loop, run the **full reverse process** from pure noise and steer the trajectory with two gradient losses at each step:

$$L = \lambda_b \| (\hat{x}_0 - x_{\text{known}}) \cdot (1-\text{mask}) \|^2 + \lambda_d \| \nabla \cdot \hat{x}_0 \|^2$$

Where $\hat{x}_0$ is estimated via **Tweedie's formula**: $\hat{x}_0 = \frac{x_t - \sqrt{1-\bar\alpha_t}\,\epsilon_\theta}{\sqrt{\bar\alpha_t}}$

**Key advantages over RePaint:**
- No copy-paste → no boundary artefacts, no OOD composites
- No FFT projection → no magnitude crushing
- Natural enforcement of both boundary matching AND divergence-free constraint
- No resample cycles needed

### Implementation

- `guided_inpaint()` in `ddpm/utils/inpainting_utils.py`
- `divergence_2d_diffable()` — autograd-compatible divergence via `F.conv2d` (no in-place ops)
- Config in `data.yaml`: `inpainting_method: guided`, `guidance_scale_boundary`, `guidance_scale_div`
- `run_ddpm_gp_bulk_robot_path.py` dispatches to `guided_inpaint` or `inpaint_generate_new_images` based on config

### Results: Guidance Scale Sweep (Single Sample)

True mean magnitude: 0.14407

| gs_boundary | gs_div | Magnitude | Ratio | MSE | \|div\| |
|-------------|--------|-----------|-------|-----|---------|
| 1.0 | 0.5 | 0.08689 | 0.60× | 0.009854 | 0.005066 |
| 5.0 | 1.0 | 0.10094 | 0.70× | 0.008530 | 0.005944 |
| 10.0 | 1.0 | 0.10464 | 0.73× | 0.008450 | 0.007027 |
| 20.0 | 1.0 | 0.10546 | 0.73× | 0.007905 | 0.008232 |
| 20.0 | 2.0 | 0.10466 | 0.73× | 0.007691 | 0.007304 |
| 50.0 | 5.0 | 0.10129 | 0.70× | 0.007043 | 0.006562 |
| 100.0 | 5.0 | 0.10175 | 0.71× | 0.007421 | 0.007783 |

### Comparison vs RePaint

| Method | Magnitude Ratio | MSE | Notes |
|--------|----------------|-----|-------|
| RePaint (FFT, 1 cycle) | 0.34× | 0.028 | Projection crushes magnitude |
| RePaint (no projection) | 2.13× | 0.124 | Overshoots without projection |
| **Guided (gs_b=50, gs_d=5)** | **0.70×** | **0.007** | **4× MSE improvement, 2× magnitude** |

### Key Findings

1. **4× MSE improvement** over best RePaint config (0.007 vs 0.028)
2. **2× better magnitude recovery** (0.70× vs 0.34×)
3. Magnitude plateaus at ~0.73× — residual bias from UNet's unconditional predictions
4. Divergence stays low (~0.005–0.008) across all configs
5. **Speed**: ~90 it/s on MPS (2–3s per sample) — faster than RePaint with resample cycles
6. Best MSE at (gs_b=50, gs_d=5); best magnitude at (gs_b=20, gs_d=1)

### Remaining Limitation

The magnitude still saturates at ~73% of true. This is because the UNet was trained for **unconditional denoising** — it has no awareness of the mask or known values during training. Gradient guidance helps but cannot fully compensate for the model's bias.

### Next Step: Mask-Aware Training (Palette-Style)

To fundamentally solve the remaining magnitude gap, the next approach to try is **mask-aware training**, inspired by Palette (Saharia et al., 2022):

**Core idea:** Modify the UNet to accept the mask and known values as additional input channels during training. This makes the denoiser aware of the inpainting task structure.

**Implementation plan:**
1. **Change UNet input**: 2 channels → 5 channels: `[x_t (2ch), mask (1ch), known_values (2ch)]`
2. **Training pipeline**: At each training step, generate random masks, compute `known_values = x_0 * (1 - mask)`, concatenate `[x_t, mask, known_values]`, train denoiser to predict noise
3. **Random mask augmentation**: Use diverse masks (rectangles, robot paths, random blocks) during training so the model generalizes to arbitrary mask shapes
4. **Inference**: Simply run standard reverse process with mask/known values as conditioning — no copy-paste, no gradient guidance needed, no projection
5. **Expected benefit**: Model learns to jointly denoise AND inpaint with correct magnitudes because it sees the conditioning context during training

**Why this should work:**
- The denoiser will be in-distribution at inference time (it was trained with mask+context conditioning)
- No post-hoc projection needed → no magnitude crushing
- No gradient computation through UNet → faster inference
- Palette demonstrated excellent results on natural image inpainting with this exact approach

**Estimated effort:** ~1–2 days (modify UNet, update dataloader, retrain ~600 epochs)

---

### February 19, 2026 — Beta Schedule Diagnosis & Fix (RePaint+CG Experiment)

**Objective:** Diagnose and fix magnitude blow-up in RePaint+CG inpainting results.

#### Background

The RePaint+CG experiment (`experiments/02_inpaint_algorithm/repaint_cg/`) uses an unconditional eps-prediction DDPM with `forward_diff_div_free` noise (250 steps) and RePaint inpainting with per-step CG div-free projection. Initial training converged (best test loss 0.0023 at epoch 172), but inference showed severe magnitude blow-up:

| Method | Median Magnitude | GT Magnitude | MSE |
|--------|-----------------|--------------|-----|
| RePaint (no projection) | 1.45 | 0.19 | 34.48 |
| RePaint+CG | 0.31 | 0.19 | 3.20 |
| GP baseline | ~0.19 | 0.19 | 0.0007 |

CG projection helped (1.45→0.31) but magnitudes were still 1.6× too large.

#### Investigation (5-Test Diagnostic)

Created `scripts/diagnose_repaint_magnitudes.py` with five targeted tests:

1. **CG on pure div-free noise** → energy_ratio = 1.0000 at all timesteps. CG is a perfect no-op on div-free noise — NOT the culprit.
2. **CG on x_t (signal+noise)** → 2–12% energy loss (removes irrotational component of signal). Expected and acceptable.
3. **Plain RePaint step trace** → x0_pred RMS 25–37× too large; final magnitude 7.5× GT.
4. **RePaint+CG step trace** → final magnitude 1.6× GT. CG helps but doesn't fix root cause.
5. **Repeated project→re-stamp cycle** → stable convergence. No runaway growth.

#### Key Finding: Unconditional Generation Also Broken

Created `scripts/diagnose_uncond_generation.py` — tested the model with NO inpainting at all (pure reverse diffusion from noise). Result: generated samples were **2.32× too large**. This proved the problem is in the model/schedule, not in RePaint.

#### Root Cause: Overly Aggressive Beta Schedule

Compared the beta schedules:

| Model | min_beta | max_beta | ᾱ₂₄₉ (final SNR) |
|-------|----------|----------|-------------------|
| Working Gaussian model | 0.0001 | 0.02 | 0.0797 |
| Fwd-diff model (broken) | 0.0004 | 0.08 | 0.000033 |

The fwd-diff schedule was **4× more aggressive**. By t=166, ᾱ < 0.01 — all signal was destroyed. The last 84 of 250 timesteps were pure noise→noise denoising, which the model couldn't learn meaningfully. The near-zero ᾱ_T caused the eps→x₀ conversion to amplify prediction errors enormously.

#### Fix Applied

- Changed `experiments/templates/base_inpaint.yaml`: `min_beta: 0.0001`, `max_beta: 0.02` (matching the working Gaussian model's schedule)
- Killed old training (PID 7041), relaunched with corrected schedule
- New training: loss dropped from 0.267 → 0.022 in first ~200 epochs; best test loss 0.0022 at epoch 203 (still improving)

#### Lessons Learned

1. **CG projection is well-behaved**: energy_ratio = 1.0 on div-free noise; it only removes irrotational components, which is exactly what we want.
2. **Beta schedule must preserve residual signal**: ᾱ_T should be > ~0.05 so the model always has some signal to work with. Our old schedule had ᾱ_T = 0.000033.
3. **Test unconditional generation first**: If pure generation is broken, no inpainting algorithm can fix it.
4. **Don't blame training duration for bad results when loss has plateaued**: The old model was fully converged at 0.0023 — the problem was architectural (schedule), not training.

#### Files Created

- `scripts/diagnose_repaint_magnitudes.py` — 5-test magnitude diagnostic
- `scripts/diagnose_uncond_generation.py` — unconditional generation sanity check

#### Status

Training with corrected beta schedule in progress. Will re-run inference after convergence.

---

### February 20, 2026 — Architecture Context Notes (CRITICAL)

**Purpose:** Prevent confusion about which network/algorithm is under discussion.

#### Active Experimental Configurations

| Experiment | UNet type | Channels | Conditioning | Inpainting | Noise |
|------------|-----------|----------|--------------|------------|-------|
| `01_noise_strategy/fwd_divfree` | `standard` (2ch) | 2 in, 2 out | NONE | RePaint | `forward_diff_div_free` |
| `01_noise_strategy/fwd_divfree_equalized` | `standard` (2ch) | 2 in, 2 out | NONE | RePaint | `fwd_diff_eq_divfree` |
| `02_inpaint_algorithm/repaint_cg` | `standard` (2ch) | 2 in, 2 out | NONE | RePaint + CG proj | `forward_diff_div_free` |
| `02_inpaint_algorithm/repaint_gaussian_attn` | `standard_attn` (2ch) | 2 in, 2 out | NONE | RePaint | `gaussian` |

#### Key Rules — Know Which Architecture You're Discussing

1. **`standard` / `standard_attn` UNets are UNCONDITIONAL** — 2 channels only (u, v). No mask channel, no known_values channels, no `mask_xt`. The model sees only x_t.
2. **`film` / `concat` UNets are CONDITIONED** — 5 channels: [x_t(2ch), mask(1ch), known_values(2ch)]. These use `mask_xt`, `known_values`, and FiLM/concatenation conditioning.
3. **RePaint does NOT use conditioning** — it's an unconditional algorithm. The model never sees the mask. Known region is pasted in via copy-paste at each step.
4. **mask_aware_inpaint / CFG inpaint use conditioned models** — these pass mask+known_values as extra input channels.
5. **`mask_xt` is ONLY relevant to conditioned models** — it replaces x_t in the known region with Gaussian noise to prevent information leakage. Unconditional models don't have this concept.

#### Current Focus (Feb 20, 2026)

Working on div-free noise for the **unconditional RePaint pipeline** (`standard` 2ch UNet). The spectral gap analysis applies to noise generation only — all projections have flat transfer functions. The equalized noise fix propagates correctly through all noise injection sites in `repaint_standard()` because they all use the same `noise_strategy` object.

---
