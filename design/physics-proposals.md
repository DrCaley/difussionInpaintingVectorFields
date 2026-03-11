# 5 Novel Physics-Informed Diffusion Proposals for Ocean Velocity Inpainting

**Date:** 2025-01-27
**Context:** Research proposals for integrating deeper ocean physics into our DDPM-based inpainting framework for vector (velocity) fields.

---

## Background: What We Already Have

Our codebase implements several physics-informed components:

| Component | Physics | Assurance Level |
|-----------|---------|-----------------|
| Divergence-free noise (streamfunction → forward-diff curl) | Incompressibility | **Hard** (exact) |
| Equalized div-free noise (spectral coloring of ψ) | Flat velocity power spectrum | **Hard** |
| Topology-aware loss (vorticity + divergence + speed MSE) | Differential structure | **Soft** (weighted penalty) |
| Physical loss (divergence penalty on predicted noise) | Incompressibility | **Soft** |
| CG/FFT/Jacobi div-free projection (post-hoc) | Incompressibility | **Hard** (per application) |
| Helmholtz-Hodge decomposition | Solenoidal/irrotational split | Diagnostic only |
| Eddy detection (Okubo-Weiss, Graftieaux Γ₁) | Vortex identification | Diagnostic only |

### What's Missing

- **No geostrophic/rotational-frame physics** (Coriolis, Rossby balance)
- **No energy spectrum constraints** on model outputs (only noise is spectrally shaped)
- **No potential vorticity** conservation or invertibility
- **No enstrophy** (vorticity variance) constraints
- **Divergence-free is enforced either softly (loss) or post-hoc (projection)** — never structurally in the model architecture
- **Eddy detection is diagnostic only** — not used during training or inference

---

## Proposal 1: Streamfunction-Space Diffusion

### Core Idea

Instead of diffusing the 2-channel velocity field $(u, v)$ and projecting to
divergence-free afterward, **diffuse a 1-channel scalar streamfunction $\psi$**
and recover velocity via the curl operator:

$$u = \frac{\partial \psi}{\partial y}, \quad v = -\frac{\partial \psi}{\partial x}$$

### Physics Motivation

The 2D incompressible Navier-Stokes equations admit a streamfunction
formulation that eliminates the pressure and continuity equation entirely.
For incompressible 2D flow, $\nabla \cdot \mathbf{v} = 0$ is satisfied identically
by $\mathbf{v} = \nabla \times \psi$. Ocean surface currents at scales larger than
~10 km are approximately 2D incompressible, making this representation natural.

The streamfunction-vorticity formulation $\partial_t \omega + J(\psi, \omega) = \nu \nabla^2 \omega$
is the standard computational framework for 2D geophysical fluid dynamics
(Charney, 1948; Arakawa, 1966). Our model would be learning in this same space.

### How It Works

1. **Data preparation**: Convert each training sample from $(u, v)$ → $\psi$ by
   solving $\nabla^2 \psi = \omega = \partial v / \partial x - \partial u / \partial y$
   (Poisson inversion, well-defined up to a constant). Or fit $\psi$ on the
   $(H+1) \times (W+1)$ grid via our existing CG normal-equation solver.

2. **Forward process**: Add Gaussian noise to $\psi$ (standard scalar diffusion).
   The noise in $\psi$-space induces divergence-free noise in velocity space
   automatically — this is exactly what `ForwardDiffDivFreeNoise` already does.

3. **UNet**: Modified to 1-channel input, 1-channel output. Predicts $\hat{\psi}_0$
   (or noise $\hat{\epsilon}_\psi$).

4. **Velocity recovery**: After denoising, apply forward-diff curl to get $(u, v)$.
   **Every single sample is exactly divergence-free. Always. No projection needed.**

5. **Inpainting**: The mask and known values are expressed in $\psi$-space.
   The known velocities constrain $\psi$ at their locations.

6. **Loss**: Can be computed either in $\psi$-space (MSE on $\psi$) or in
   velocity space ($\ell(\text{curl}(\hat{\psi}_0), \text{curl}(\psi_0))$).
   The velocity-space loss ensures the model optimizes what we actually care about.

### Why It's Novel

- No prior work on diffusion models operating in streamfunction space for vector
  field inpainting.
- Eliminates the entire class of divergence problems (discontinuities at
  copy-paste boundaries, need for per-step projections, CG solver overhead).
- Reduces model from 2-channel to 1-channel — roughly halves the generation complexity.
- Training in the correct physical representation space (streamfunction–vorticity)
  matches the standard formulation of 2D geophysical fluid dynamics.

### Implementation Complexity: **Medium**

- Need a preprocessing step to convert velocity → streamfunction.
- UNet architecture change (1ch in/out instead of 2ch).
- Mask conditioning needs rethinking (mask was on velocity grid; now on ψ grid).
- Existing `forward_diff_project_div_free` CG solver can convert velocity → ψ.

### Risks

- Streamfunction is defined up to a constant → need consistent normalization.
- The mask might not map cleanly from velocity grid to ψ grid
  (ψ is on the staggered $(H+1) \times (W+1)$ grid).
- Boundary conditions of ψ near land need careful handling.

---

## Proposal 2: Energy Spectrum Regularization Loss

### Core Idea

Add a spectral loss term that penalizes deviations between the radially-averaged
kinetic energy spectrum of the model's predicted velocity field and the expected
ocean energy spectrum:

$$\mathcal{L}_{\text{spectral}} = \sum_{k} w(k) \left( \log E_{\hat{x}_0}(k) - \log E_{\text{target}}(k) \right)^2$$

where $E(k) = \frac{1}{2} \langle |\hat{u}(k)|^2 + |\hat{v}(k)|^2 \rangle$ is the
radially-averaged kinetic energy spectrum.

### Physics Motivation

Ocean velocity fields obey well-established spectral scaling laws from 2D
turbulence theory:

- **Kraichnan (1967) inverse energy cascade**: $E(k) \sim k^{-5/3}$ for
  wavenumbers below the forcing scale (energy flows to large scales).
- **Kraichnan forward enstrophy cascade**: $E(k) \sim k^{-3}$ for wavenumbers
  above the forcing scale (enstrophy flows to small scales).
- **Charney (1971) QG turbulence**: $E(k) \sim k^{-3}$ for both energy and
  enstrophy cascades on the $\beta$-plane.
- Observations show **ocean surface velocity spectra** roughly follow
  $E(k) \sim k^{-\alpha}$ with $\alpha \in [2, 3]$ depending on scale and region.

Currently, only `ForwardDiffEqualizedDivFreeNoise` has any spectral awareness
(it whitens the noise). The **model output** has no spectral constraint at all.
This means inpainted fields can have the wrong balance of large-scale and
small-scale features — too smooth (lacking fine structure) or too noisy
(lacking large-scale coherence).

### How It Works

1. **During training**: After reconstructing $\hat{x}_0$ from the model's
   prediction (available in `TopologyAwareLossStrategy` already), compute its
   2D FFT and radially average to get $E_{\hat{x}_0}(k)$.

2. **Compute target spectrum** $E_{\text{target}}(k)$ from the clean training
   sample $x_0$.

3. **Spectral loss**: Penalize the log-domain difference (log because spectra
   span orders of magnitude). Weight by $w(k)$ — e.g., uniform, or emphasize
   the inertial range where the power law should hold.

4. **Combine with existing losses**:
   $\mathcal{L} = \text{MSE} + \lambda_v L_{\text{vort}} + \lambda_d L_{\text{div}} + \lambda_E L_{\text{spectral}}$

5. **Timestep gating**: Like topology-aware loss, only active for $t < t_{\max}$
   where $\hat{x}_0$ reconstruction is meaningful.

### Why It's Novel

- No prior work on spectral energy distribution loss for diffusion-based
  inpainting of fluid fields.
- Addresses a fundamental limitation: pixel-wise MSE doesn't preserve spectral
  structure. Two fields can have the same MSE but vastly different spectra.
- Directly connects to established geophysical turbulence theory.
- Goes beyond noise coloring (which only controls noise spectrum) to control
  the **output distribution's** spectral properties.

### Implementation Complexity: **Low**

- Implemented entirely as a new loss term, no architecture changes.
- Can be added as a method in `TopologyAwareLossStrategy` or as a new strategy.
- FFT + radial averaging is ~5 lines of PyTorch.
- Differentiable through the FFT (PyTorch natively supports `torch.fft.fft2` gradients).

### Risks

- A single snapshot's spectrum is noisy — radial averaging helps but sample
  variance is high.
- The spectral exponent varies by ocean region and scale — may need to
  estimate it from the training data rather than imposing a fixed value.
- Log-domain loss can be unstable for very low-energy wavenumber bins (add ε).

---

## Proposal 3: Potential Vorticity Guidance During Inference

### Core Idea

Use the quasi-geostrophic potential vorticity (QGPV) invertibility principle as
a physics-based guidance signal during the reverse diffusion process. From the
known (observed) region, estimate the PV field, interpolate it into the unknown
region, then guide the diffusion trajectory toward fields consistent with this
PV estimate.

For 2D incompressible barotropic flow (our setting), PV simplifies to:

$$q = \omega + f = \frac{\partial v}{\partial x} - \frac{\partial u}{\partial y} + f$$

where $f = f_0 + \beta y$ is the Coriolis parameter (beta-plane approximation).

### Physics Motivation

Potential vorticity is the single most important dynamical quantity in
geophysical fluid dynamics. The PV invertibility principle (Hoskins,
McIntyre & Robertson, 1985) states that **given a PV distribution and a
balance condition (geostrophic or gradient wind balance), the full flow field
can be recovered by inverting an elliptic operator**:

$$\nabla^2 \psi = q - f \implies \psi \implies (u, v)$$

This is extraordinarily powerful for inpainting: if we can estimate PV in the
unknown region (by smooth interpolation from the known region, or by assuming
material conservation along streamlines), we can construct a physically
consistent first guess that the diffusion model refines.

In our setting (Rams Head, Virgin Islands), the domain is small enough that
$f$ is approximately constant ($f_0 \approx 2\Omega \sin(18.3°) \approx 4.6 \times 10^{-5}$ s⁻¹),
but the relative vorticity $\omega$ varies significantly.

### How It Works

**Option A: PV-Guided Inference (gradient guidance)**

1. At each reverse step $t$, after predicting $\hat{x}_0$, compute
   $\hat{q} = \omega(\hat{x}_0) + f$ in the full domain.

2. Compute a reference PV field $q_{\text{ref}}$ by:
   - Using exact PV from known (observed) pixels.
   - Smoothly interpolating/extrapolating into the unknown region
     (e.g., via Laplacian diffusion or radial basis functions on
     the PV field).

3. Define a guidance loss:
   $\mathcal{L}_{\text{PV}} = \| q(\hat{x}_0) - q_{\text{ref}} \|^2_{\text{unknown region}}$

4. Compute $\nabla_{x_t} \mathcal{L}_{\text{PV}}$ and adjust $x_{t-1}$:
   $x_{t-1} \leftarrow x_{t-1} - \eta \nabla_{x_t} \mathcal{L}_{\text{PV}}$
   (exactly like DPS / gradient-guided inpainting, which we already have in
   the codebase as `guided_inpaint()`).

**Option B: PV Inversion as initialization**

1. From the known region, estimate PV and interpolate to full domain.
2. Solve $\nabla^2 \psi = q - f_0$ for $\psi$ via FFT or CG.
3. Recover $(u, v) = \text{curl}(\psi)$.
4. Use this as the x₀ estimate in the first stage of our 3-stage rollout,
   or as the initial condition for a shorter diffusion (fewer reverse steps needed).

### Why It's Novel

- No prior work on PV-guided diffusion for ocean inpainting.
- Leverages the most powerful principle in geophysical fluid dynamics
  (PV invertibility) to propagate physical constraints from observed to
  unobserved regions.
- Connects DDPM inference to the classical meteorological/oceanographic
  analysis framework (PV thinking).
- The PV interpolation step provides a physics-based "prior" that the
  diffusion model refines — combining data-driven and physics-based approaches.

### Implementation Complexity: **Medium**

- PV computation: trivial (vorticity + constant, already computed in loss).
- PV interpolation into unknown region: need thin-plate spline or Laplacian
  diffusion solve (moderate, but similar to existing CG solvers).
- Gradient guidance: already implemented in `guided_inpaint()` — just need
  a new guidance function.
- PV inversion (Poisson solve): trivial via FFT or existing Jacobi/CG solvers.

### Risks

- On our small domain (~10 km), $f$ is nearly constant, so PV ≈ relative
  vorticity — the PV inversion is less powerful than at larger scales.
- PV interpolation into a large unknown region (70-95% missing) is uncertain.
- The guidance gradients might fight with the model's learned distribution.

---

## Proposal 4: Enstrophy-Preserving Spectral Denoising Step

### Core Idea

In 2D turbulence, **enstrophy** $Z = \frac{1}{2} \int \omega^2 \, dA$ is a second
conserved quantity (alongside energy). During the reverse diffusion process,
add a **spectral correction step** after each denoising prediction that adjusts
the vorticity power spectrum to match the expected enstrophy distribution,
preserving the correct balance between large-scale flow and fine-scale vorticity
filaments.

### Physics Motivation

2D turbulence has a remarkable dual cascade (Kraichnan, 1967):

- **Inverse energy cascade**: Energy injected at intermediate scales cascades
  upward to larger scales → $E(k) \sim k^{-5/3}$ for $k < k_f$
- **Forward enstrophy cascade**: Enstrophy cascades downward to smaller scales
  → $E(k) \sim k^{-3}$ (or equivalently $Z(k) \sim k^{-1}$) for $k > k_f$

The total enstrophy $Z = \int k^2 E(k) dk$ weights high-$k$ (small-scale)
features heavily. If the diffusion model over-smooths (common failure mode),
**enstrophy drops** because fine-scale vorticity filaments are lost. If the
model generates spurious noise, **enstrophy increases artificially**.

Currently, our `TopologyAwareLossStrategy` penalizes vorticity MSE point-wise
but doesn't constrain the **total enstrophy** or its spectral distribution.

### How It Works

1. **Estimate target enstrophy statistics** from the training set:
   - Mean and std of total enstrophy per sample
   - The enstrophy spectrum $Z(k) = k^2 E(k)$ averaged over training set

2. **During inference**, after each reverse step's $\hat{x}_0$ prediction:
   a. Compute vorticity $\omega = \partial v / \partial x - \partial u / \partial y$
   b. Compute enstrophy spectrum $Z(k) = k^2 |\hat{\omega}(k)|^2$
   c. Compare per-wavenumber to target $Z_{\text{target}}(k)$
   d. Rescale each Fourier mode of $\omega$ to match:
      $\hat{\omega}'(k) = \hat{\omega}(k) \sqrt{Z_{\text{target}}(k) / Z_{\hat{x}_0}(k)}$
   e. Invert: $\nabla^2 \psi' = \omega'$, then $\mathbf{v}' = \text{curl}(\psi')$

3. This is a **projection step** (analogous to our existing div-free projection)
   but in spectral enstrophy space. It preserves the model's spatial patterns
   while ensuring the correct amount of energy at each scale.

4. **Schedule**: Apply more aggressively at low timesteps (near $t=0$ where
   details matter) and less at high timesteps (where the broad structure is
   set). This matches the coarse-to-fine nature of diffusion.

### Why It's Novel

- No prior work on enstrophy-preserving diffusion for fluid fields.
- Directly connects to the fundamental dual-cascade theory of 2D turbulence.
- Addresses the critical failure mode where DDPM over-smooths small-scale
  vorticity structure — a problem observed in our current results.
- Acts as a physics-based spectral normalization, analogous to how batch norm
  normalizes feature statistics.

### Implementation Complexity: **Medium**

- Enstrophy spectrum computation: FFT + radial averaging (~10 lines).
- Per-mode rescaling: element-wise multiply in Fourier space.
- Poisson inversion for ψ from ω: already implemented (FFT solver).
- Integration into reverse loop: wrap as a callback after each step.

### Risks

- Per-mode rescaling can introduce artifacts if individual modes have very
  low energy (division instability — use smoothing/clipping).
- The enstrophy spectrum varies between samples; using the mean is a first
  approximation. Could also estimate from the known region.
- Interaction with div-free projection: need to ensure both steps are compatible
  (they are if we go ω → ψ → curl, since that's automatically div-free).

---

## Proposal 5: Helmholtz-Decomposed Dual-Head Architecture

### Core Idea

Replace the single-output UNet with a **dual-head architecture** where one head
predicts the streamfunction $\psi$ (divergence-free/solenoidal component) and the
other predicts the velocity potential $\phi$ (irrotational component):

$$\mathbf{v} = \underbrace{\nabla \times \psi}_{\text{div-free (head 1)}} + \underbrace{\nabla \phi}_{\text{irrotational (head 2)}}$$

For ocean currents, the irrotational component should be near-zero. The $\phi$
head learns the residual, which can be zeroed out at inference time for a
hard divergence-free guarantee.

### Physics Motivation

The Helmholtz-Hodge decomposition is a fundamental theorem of vector calculus:
any sufficiently smooth vector field can be uniquely decomposed into a
divergence-free (solenoidal) part and a curl-free (irrotational) part. For
ocean surface currents:

- The **solenoidal component** $\nabla \times \psi$ captures the geophysically
  meaningful flow: eddies, jets, gyres, and geostrophically balanced currents.
- The **irrotational component** $\nabla \phi$ captures convergence/divergence
  (surface water piling up or spreading). For 2D surface currents at
  mesoscale (>10 km), this is very small.

The existing `HH_decomp.py` performs this decomposition post-hoc as a diagnostic.
This proposal bakes it into the model architecture, so the network **learns**
the decomposition during training.

### How It Works

1. **Architecture**: Share the encoder and bottleneck. Split at the decoder
   into two parallel heads:
   - **Head ψ** (streamfunction): Outputs 1 channel on $(H+1) \times (W+1)$ grid.
     Velocity via forward-diff curl.
   - **Head φ** (velocity potential): Outputs 1 channel on $H \times W$ grid.
     Velocity via central-diff gradient.

2. **Velocity reconstruction**:
   ```
   v_solenoidal = forward_diff_curl(ψ)     # exactly div-free
   v_irrotational = central_diff_grad(φ)    # exactly curl-free
   v_total = v_solenoidal + v_irrotational
   ```

3. **Training loss**:
   ```
   L = MSE(v_total, v_target)
       + λ_φ ‖φ‖²                          # L2 penalty on irrotational component
       + λ_topo * TopologyLoss(v_total)      # existing topology-aware terms
   ```
   The $\lambda_\phi \|\phi\|^2$ term biases the model toward divergence-free
   solutions (since ocean flows are nearly incompressible). The model learns
   to put almost everything into $\psi$ and use $\phi$ only for residual
   corrections.

4. **Inference**: Set $\phi = 0$ for hard divergence-free guarantee, or keep
   the small $\phi$ for slight realism (some real ocean flows have weak
   convergence/divergence).

### Why It's Novel

- No prior diffusion model has used Helmholtz-decomposed architecture.
- The model learns physics-informed internal representations rather than
  applying post-hoc corrections.
- The shared encoder sees the full velocity field; the split decoder forces
  the network to separate physical (solenoidal) from unphysical (irrotational)
  components — a physics-based inductive bias.
- The irrotational head provides a built-in diagnostic: if $\|\phi\|$ is large
  after training, the model hasn't learned incompressibility well.
- Allows a **hard-to-soft spectrum**: at inference, φ can be zeroed (hard),
  scaled down (soft), or left as-is (learned) — maximum flexibility.

### Implementation Complexity: **Medium-High**

- Architecture modification: split decoder with two heads (moderate; our UNet
  already has modular encoder/decoder).
- Forward-diff curl operator: already implemented in `noise_utils.py`.
- Gradient operator for φ: simple finite difference (~5 lines).
- Different output sizes (ψ on staggered grid): needs padding logic.
- Training: the penalty on φ needs tuning.

### Risks

- The staggered grid for ψ introduces complexity in the decoder head.
- Two heads means more parameters in the decoder (but shared encoder saves).
- The $\phi$ penalty schedule might be tricky — too strong and the model
  can't learn, too weak and it doesn't prefer the ψ head.

---

## Dataset Divergence Analysis (2025-01-27)

Before prioritizing proposals, we quantified how much divergence (irrotational
energy) the Rams Head dataset actually contains via Helmholtz-Hodge decomposition
on all 17,040 samples.

### Results

| Metric | Value |
|--------|-------|
| **Irrotational KE fraction (mean)** | **32.25%** |
| Irrotational KE fraction (median) | 31.00% |
| Irrotational KE fraction (P90) | 47.04% |
| Irrotational KE fraction (max) | 79.19% |
| Divergence / Vorticity RMS ratio | **0.77** (divergence is 77% of vorticity magnitude) |
| Streamfunction-only reconstruction MSE | 30.3% of total KE |

### Implications

**This is a lot of divergence.** Nearly a third of the kinetic energy is in the
irrotational (divergent) component, and in some samples, it exceeds the solenoidal
component entirely. This has first-order implications for every proposal:

**Proposal 1 (Streamfunction-space diffusion): SERIOUSLY COMPROMISED.**
A streamfunction can only represent the solenoidal component. With 32% of KE in the
irrotational component, the best possible reconstruction has a 30% KE error floor —
the model literally cannot represent a third of the signal. This makes Proposal 1
unsuitable as a standalone approach. It could still work as one head of a dual-head
model (see Proposal 5).

**Proposal 2 (Energy spectrum loss): FULLY COMPATIBLE.**
The energy spectrum $E(k)$ is computed on the raw velocity field, not on a
decomposed component. It captures both solenoidal and irrotational energy at each
wavenumber. No modifications needed.

**Proposal 3 (PV guidance): PARTIALLY COMPATIBLE.**
PV ($q = \omega + f$) only constrains the solenoidal component (vorticity is the curl
of velocity, independent of divergence). PV guidance will correctly steer the
rotational flow structure but provides zero information about the divergent component.
With 32% of energy unconstrained by PV, guidance is useful but incomplete.

**Proposal 4 (Enstrophy preservation): NEEDS MODIFICATION.**
The current design goes $\omega \to \psi \to \text{curl}(\psi)$, which discards the
irrotational component entirely — same problem as Proposal 1. Fixed version must:
(a) decompose the field, (b) rescale only the solenoidal vorticity spectrum,
(c) add back the original irrotational component unchanged.

**Proposal 5 (Helmholtz dual-head): BEST FIT FOR THIS DATASET.**
With significant real divergence, the dual-head design becomes essential rather than
optional. The $\psi$ head learns the 68% solenoidal component, the $\phi$ head learns
the 32% irrotational component. At inference: keep both heads active (unlike the
original proposal to zero $\phi$). The $\lambda_\phi$ penalty should be reduced or
removed since the divergent component is real physics (likely ageostrophic dynamics,
tidal convergence, or vertical velocity effects), not noise.

### Revised Priority Given Divergent Data

1. **Proposal 5 (Dual-head Helmholtz)** — Now the clear priority. It's the only
   proposal that handles both components. The significant divergence makes this
   architecturally necessary.
2. **Proposal 2 (Spectral loss)** — Still lowest cost, works on full velocity.
   Good first experiment regardless.
3. **Proposal 3 (PV guidance)** — Still useful for the solenoidal 68%, but
   explicitly incomplete. Document this limitation.
4. **Proposal 4 (Enstrophy preservation)** — Viable with the decompose-correct-recombine
   modification. Acts on the solenoidal component only.
5. **Proposal 1 (Streamfunction diffusion)** — Demoted. Cannot be used standalone.
   Only viable as the $\psi$ head within Proposal 5.

---

## Comparison of Proposals

| # | Proposal | Physics Principle | Assurance Level | Complexity | Impact |
|---|----------|-------------------|-----------------|------------|--------|
| 1 | Streamfunction-space diffusion | Incompressibility (structural) | **Hard** (by construction) | Medium | **Very High** — eliminates entire class of div problems |
| 2 | Energy spectrum loss | Turbulence theory (Kraichnan k⁻³) | **Soft** (loss penalty) | Low | **Medium** — corrects spectral bias in outputs |
| 3 | PV guidance during inference | PV invertibility (Hoskins 1985) | **Soft** (gradient guidance) | Medium | **High** — physics-based propagation from known→unknown |
| 4 | Enstrophy-preserving denoising | Dual cascade conservation | **Hard** (spectral projection) | Medium | **Medium-High** — controls fine-scale vorticity fidelity |
| 5 | Helmholtz dual-head architecture | Helmholtz-Hodge theorem | **Tunable** (hard if φ=0) | Medium-High | **High** — learned physics decomposition |

### Recommended Priority

1. **Proposal 2 (Spectral loss)** — Lowest implementation cost, immediate benefit, can
   be added to current experiments today as a new loss term.
2. **Proposal 1 (Streamfunction diffusion)** — Most fundamental change with highest
   potential impact. Requires new experiment but is architecturally clean.
3. **Proposal 5 (Dual-head Helmholtz)** — High novelty, physics-informed architecture.
   Good paper contribution.
4. **Proposal 3 (PV guidance)** — Leverages existing gradient-guidance infrastructure.
   Strong physics motivation.
5. **Proposal 4 (Enstrophy preservation)** — Most specialized; best as an add-on to
   Proposal 1 or 5.

### Combination Strategy

These proposals are **not mutually exclusive**. The strongest system would combine:
- Streamfunction-space diffusion (Proposal 1) for structural div-free guarantee
- Spectral loss (Proposal 2) for correct energy distribution
- Enstrophy preservation (Proposal 4) for fine-scale vorticity fidelity
- PV guidance at inference (Proposal 3) for physics-informed inpainting

This would give a model that operates in the physically natural representation
(streamfunction), trained with spectral constraints from turbulence theory,
and guided at inference by the most fundamental conservation law in geophysics
(potential vorticity). This combination would be a strong contribution.

---

## Second-Round Ideas (2026-03-08)

After the Helmholtz dual-head (Proposal 5) was implemented and is training,
we re-examined how physics can be injected at each stage of the diffusion
pipeline. Key insight: **div-free noise is theoretically flawed** for our data
because 32% of the energy is irrotational. Proper DDPM requires the forward
process to converge to a known distribution; div-free noise converges to a
restricted subspace that can't represent a third of the signal.

Below are new ideas organized by injection point.

### Where Can You Inject Physics Without Breaking the Math?

There are four injection points in a DDPM pipeline:

| Injection Point | Math Constraint | Risk of Breaking DDPM |
|-----------------|----------------|-----------------------|
| **A. Forward process (noise)** | Must converge to tractable distribution | **High** — changes the ELBO |
| **B. Architecture (denoiser)** | Must map $x_t \to \hat{x}_0$ with correct dims | **None** — invisible to DDPM |
| **C. Reverse steps (projections)** | Modifies $x_{t-1}$ between steps | **Low** if small corrections |
| **D. Training loss** | Modifies gradient signal | **None** — standard practice |

---

### Proposal 6: Helmholtz-Decomposed Noise Schedule

**Injection point: A (forward process)**

Instead of one noise schedule $\beta_t$, use **two independent schedules** for
the two orthogonal Helmholtz subspaces:

- $\beta_t^{sol}$ for the solenoidal component (slower schedule — ocean flow
  is mostly rotational, so preserve large-scale structure longer)
- $\beta_t^{irr}$ for the irrotational component (faster schedule — smaller,
  noisier signal can be destroyed more quickly)

$$x_t = \sqrt{\bar\alpha_t^{sol}} x_0^{sol} + \sqrt{1-\bar\alpha_t^{sol}} \epsilon^{sol}
      + \sqrt{\bar\alpha_t^{irr}} x_0^{irr} + \sqrt{1-\bar\alpha_t^{irr}} \epsilon^{irr}$$

where $\epsilon^{sol}$ is div-free noise and $\epsilon^{irr}$ is curl-free noise.

**Why it's mathematically sound:** The Helmholtz decomposition is an orthogonal
projection in $L^2$. The solenoidal and irrotational subspaces are orthogonal
complements. Running independent DDPM processes in orthogonal subspaces is
equivalent to running a single DDPM in the product space — the ELBO factorizes.
Each component follows standard diffusion in its own subspace.

**Why it's interesting:** It lets the model denoise the large-scale solenoidal
structure first (few steps at low noise) while spending more steps on the
smaller irrotational details. It's a physics-informed curriculum built into
the noise schedule. The solenoidal component carries 68% of KE but has more
spatial structure; the irrotational component carries 32% but is smoother
(gradient fields are smoother than curl fields).

**Implementation complexity: High.** Requires decomposing every sample into
$x_0^{sol}$ and $x_0^{irr}$ at data-loading time (or caching), two separate
noise sampling paths, and a modified reverse process that denoise each
component with its own schedule. The UNet must be aware of which component
is at which noise level — possibly via separate timestep embeddings.

**Risks:** The interaction between solenoidal and irrotational denoisers at
different noise levels is unexplored. If the two schedules are far apart, the
model sees one component nearly clean while the other is still noisy — the
architecture must handle this gracefully.

---

### Proposal 7: Circulation Conservation Loss

**Injection point: D (training loss)**

For any closed curve $\mathcal{C}$ in 2D inviscid flow, Kelvin's circulation
theorem states that $\Gamma = \oint_\mathcal{C} \mathbf{v} \cdot d\mathbf{l}$ is
conserved. Use this as a non-local loss term.

**How it works:**

1. Pick closed rectangular paths that straddle the known/unknown boundary
   (several sizes and positions, sampled randomly per batch).
2. Along each path, some edges lie in the known region (ground truth velocity)
   and some in the unknown region (predicted velocity).
3. The total circulation around the closed path is the sum of line integrals
   along all edges. Penalize the difference between predicted total circulation
   and the partially-known target:

$$\mathcal{L}_{circ} = \sum_{\mathcal{C}} \left(\oint_\mathcal{C} \hat{v} \cdot dl
   - \Gamma_{\text{known legs}}\right)^2$$

**Why it's powerful:** This propagates information *along* boundaries from
known to unknown regions, providing **non-local constraints** that pixel-wise
MSE losses miss entirely. If three sides of a rectangle are in the known
region, the fourth side's line integral is fully determined. This is a
physics-based way to "reach into" the unknown region with hard constraints.

**Why it's novel:** No prior work uses Kelvin's circulation theorem as a
training loss for either diffusion models or neural inpainting.

**Implementation complexity: Low-Medium.** Line integrals along grid-aligned
paths are just sums of $u \cdot \Delta x$ or $v \cdot \Delta y$ — trivially
differentiable. Need a path-sampling strategy (random rectangles spanning the
known/unknown boundary). Could start with fixed paths and move to stochastic.

**Risks:** On our small domain (~10 km), viscous effects are not negligible,
so Kelvin's theorem is approximate. The loss should be weighted softly, not
enforced as a hard constraint. Path sampling adds a hyperparameter (how many
paths, what sizes).

---

### Proposal 8: Spectral-Band Denoisers

**Injection point: B (architecture)**

Instead of one UNet predicting the full field, use **separate lightweight
denoisers for different wavenumber bands**:

- A large-scale denoiser for $k < k_1$ (energy-containing scales, large eddies)
- A mesoscale denoiser for $k_1 < k < k_2$ (eddy interactions)
- A fine-scale denoiser for $k > k_2$ (filaments, fronts)

Each denoiser only sees its frequency band of $x_t$ and predicts that band of
$x_0$. Frequency splitting and recombination done via FFT bandpass filters.

**Physics motivation:** Ocean turbulence has a strong scale separation.
Large eddies (>50 km) evolve quasi-independently of small filaments (<5 km).
The energy cascade is predominantly *inverse* (large scales gain energy from
small scales), while the enstrophy cascade is *forward* (small scales gain
vorticity variance). These are fundamentally different dynamics that benefit
from specialized denoisers.

**Why it's novel:** Analogous to wavelet-based diffusion models but motivated
by turbulence cascade physics rather than image statistics.

**Implementation complexity: High.** Need FFT-based band splitting, separate
small UNets per band, recombination, and careful handling of band boundaries
(Gibbs phenomenon). Also need to decide: shared timestep embedding or separate?

**Risks:** Band boundaries introduce artifacts. Cross-scale interactions
(nonlinear in real physics) are lost when bands are processed independently.
May need a fusion step to recombine coherently. Computational overhead of
multiple forward passes.

---

### Proposal 9: Geostrophic Balance Projection

**Injection point: C (reverse-step correction)**

At each reverse step, after predicting $\hat{x}_0$, check if the velocity
field satisfies approximate geostrophic balance and softly project toward it.

**How it works:**

1. Estimate the geostrophic velocity from the pressure gradient implied by
   the predicted field. For incompressible 2D flow, the geostrophic component
   satisfies $f \hat{k} \times \mathbf{v}_g = -\nabla p / \rho$.
2. Decompose: $\mathbf{v} = \mathbf{v}_g + \mathbf{v}_{ag}$ (geostrophic + ageostrophic).
3. Shrink the ageostrophic component: $\mathbf{v}' = \mathbf{v}_g + \gamma \mathbf{v}_{ag}$
   with $\gamma \in [0.5, 1.0]$.
4. Feed $\mathbf{v}'$ to the next reverse step.

**Applicability to Rams Head:** At latitude 18.3°N, $f \approx 4.6 \times 10^{-5}$ s⁻¹.
Rossby number $Ro = U/(fL) \approx 0.1/(4.6 \times 10^{-5} \times 10^4) \approx 0.2$.
Geostrophic balance is *approximate* — significant but not dominant. A **soft**
projection ($\gamma = 0.7$--$0.9$) is appropriate; hard geostrophic enforcement
($\gamma = 0$) would be wrong.

**Implementation complexity: Medium.** Need to estimate pressure from velocity
(Poisson solve on the momentum equation) or use the simpler approximation that
the geostrophic component equals the nondivergent part (which is just the ψ head
output from our Helmholtz UNet — a natural synergy).

**Risks:** The approximation $\mathbf{v}_g \approx \mathbf{v}_{sol}$ only holds
when the solenoidal component is geostrophically balanced, which isn't exact at
our domain size. Over-projecting could kill valid ageostrophic dynamics.

---

### Proposal 10: Lagrangian Coherence Guidance

**Injection point: C (reverse-step correction)**

During inference, compute finite-time Lyapunov exponents (FTLE) from $\hat{x}_0$
to identify Lagrangian coherent structures (LCS): transport barriers, eddy
boundaries, and attracting/repelling manifolds. Penalize configurations where
these structures are discontinuous across the known/unknown boundary.

**Physics motivation:** LCS are the "skeleton" of ocean transport. Eddy
boundaries, jet cores, and frontal zones appear as ridges in the FTLE field.
If the inpainted region introduces a discontinuous FTLE ridge at the boundary,
the result is physically implausible — parcels can't teleport across transport
barriers.

$$\mathcal{L}_{LCS} = \|\text{FTLE}(\hat{x}_0)|_{\text{boundary strip}}\|_{\text{TV}}$$

where TV is total variation across the known/unknown boundary.

**Implementation complexity: High.** FTLE requires advecting a grid of particles
forward in time under the predicted velocity field (ODE integration), computing
the deformation gradient tensor, and extracting eigenvalues. This is expensive
and requires differentiable particle advection.

**Risks:** Very expensive per reverse step. FTLE from a single snapshot is less
meaningful than from a time series. The ODE integration makes the gradient
computation complex (adjoint method needed). Best as a post-hoc diagnostic or
final-step correction rather than per-step guidance.

---

### Updated Priority (2026-03-08)

Including both original (1–5) and new (6–10) proposals:

| Priority | Proposal | Status | Next Action |
|----------|----------|--------|-------------|
| 1 | **#5 Helmholtz dual-head** | **Training (ep 88/400)** | Evaluate when done |
| 2 | **#7 Circulation conservation loss** | New | Implement as loss term — low cost, high novelty |
| 3 | **#2 Spectral energy loss** | Proposed | Implement as loss term — low cost |
| 4 | **#6 Helmholtz-split noise schedule** | New | Design experiment after #5 results |
| 5 | **#9 Geostrophic balance projection** | New | Natural synergy with #5 (ψ head ≈ geostrophic) |
| 6 | **#3 PV guidance** | Proposed | Useful for solenoidal 68% only |
| 7 | **#4 Enstrophy preservation** | Proposed | Needs irrotational fix |
| 8 | **#8 Spectral-band denoisers** | New | Too complex for now |
| 9 | **#10 Lagrangian coherence** | New | Best as post-hoc diagnostic |
| 10 | **#1 Streamfunction diffusion** | Proposed | Subsumed by #5 dual-head |

### Critical Assessment of Div-Free Noise

**Our existing div-free noise strategies are theoretically flawed for this dataset.**
The DDPM forward process requires convergence to a known, tractable distribution.
Div-free noise converges to a restricted Gaussian on the solenoidal subspace.
For data with 32% irrotational energy, 32% of the signal is not properly noised
in the forward process, which means the reverse process can't properly reconstruct
it. This is the likely root cause of why div-free noise experiments haven't shown
clear wins over standard Gaussian noise. The **only** mathematically rigorous way
to use physics-constrained noise is the Helmholtz-split schedule (Proposal 6),
where each orthogonal subspace gets its own proper DDPM forward/reverse process.
