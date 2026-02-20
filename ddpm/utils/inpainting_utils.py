import torch
import torch.nn.functional as f
from tqdm import tqdm

from data_prep.data_initializer import DDInitializer
from ddpm.vector_combination.vector_combiner import combine_fields

dd = DDInitializer()


def divergence_2d(vel, dx=1.0, dy=1.0):
    """Finite-difference divergence of a (B,2,H,W) velocity field."""
    u = vel[:, 0]
    v = vel[:, 1]

    div = torch.zeros_like(u)
    div[:, 1:-1, 1:-1] = (
        (u[:, 1:-1, 2:] - u[:, 1:-1, :-2]) / (2.0 * dx)
        + (v[:, 2:, 1:-1] - v[:, :-2, 1:-1]) / (2.0 * dy)
    )
    return div


def divergence_2d_diffable(vel, dx=1.0, dy=1.0):
    """Differentiable divergence via conv2d (no in-place ops, supports autograd)."""
    # Sobel-style central-difference kernels
    kernel_u = torch.zeros(1, 1, 3, 3, device=vel.device, dtype=vel.dtype)
    kernel_u[0, 0, 1, 0] = -1.0 / (2.0 * dx)
    kernel_u[0, 0, 1, 2] = 1.0 / (2.0 * dx)

    kernel_v = torch.zeros(1, 1, 3, 3, device=vel.device, dtype=vel.dtype)
    kernel_v[0, 0, 0, 1] = -1.0 / (2.0 * dy)
    kernel_v[0, 0, 2, 1] = 1.0 / (2.0 * dy)

    du_dx = f.conv2d(vel[:, 0:1], kernel_u, padding=1)
    dv_dy = f.conv2d(vel[:, 1:2], kernel_v, padding=1)
    return (du_dx + dv_dy).squeeze(1)  # (B, H, W)


def grad_2d(p, dx=1.0, dy=1.0):
    dpdx = torch.zeros_like(p)
    dpdy = torch.zeros_like(p)

    dpdx[:, 1:-1, 1:-1] = (p[:, 1:-1, 2:] - p[:, 1:-1, :-2]) / (2.0 * dx)
    dpdy[:, 1:-1, 1:-1] = (p[:, 2:, 1:-1] - p[:, :-2, 1:-1]) / (2.0 * dy)
    return dpdx, dpdy


def poisson_solve_jacobi(div, jacobi_iters=50):
    p = torch.zeros_like(div)

    for _ in range(jacobi_iters):
        p_new = torch.zeros_like(p)
        p_new[:, 1:-1, 1:-1] = 0.25 * (
            p[:, 2:, 1:-1]
            + p[:, :-2, 1:-1]
            + p[:, 1:-1, 2:]
            + p[:, 1:-1, :-2]
            - div[:, 1:-1, 1:-1]
        )

        # Neumann boundary: copy adjacent interior values
        p_new[:, :, 0] = p_new[:, :, 1]
        p_new[:, :, -1] = p_new[:, :, -2]
        p_new[:, 0, :] = p_new[:, 1, :]
        p_new[:, -1, :] = p_new[:, -2, :]

        p = p_new

    return p


def project_div_free_2d(vel, dx=1.0, dy=1.0, jacobi_iters=50):
    div = divergence_2d(vel, dx=dx, dy=dy)
    p = poisson_solve_jacobi(div, jacobi_iters=jacobi_iters)
    dpdx, dpdy = grad_2d(p, dx=dx, dy=dy)
    projected = vel.clone()
    projected[:, 0] = projected[:, 0] - dpdx
    projected[:, 1] = projected[:, 1] - dpdy
    return projected


def forward_diff_project_div_free(vel: torch.Tensor, cg_iters: int = 200) -> torch.Tensor:
    """Divergence-free projection via streamfunction fitting (CG solver).

    Finds the streamfunction ψ on an (H+1)×(W+1) grid whose forward-
    difference curl best fits the input velocity field, then reconstructs
    (u, v) from that ψ.  This is the EXACT same construction used by
    ForwardDiffDivFreeNoise:

        u[i,j] = ψ[i+1,j] - ψ[i,j]
        v[i,j] = -(ψ[i,j+1] - ψ[i,j])

    Because the output is built from a single scalar streamfunction via
    forward differences, it is GUARANTEED pixelwise divergence-free
    under the forward-difference operator — by construction, not
    approximately.

    Method: least-squares fit of ψ via Conjugate Gradient on the normal
    equations  A^T A ψ = A^T vel,  where A is the forward-difference
    curl operator.  CG converges in ~O(√κ) iterations (≈60 for a
    64×128 grid) vs ~O(N²) for Jacobi.

    Args:
        vel: (B, 2, H, W) velocity field.
        cg_iters: Maximum CG iterations (typically converges in <100).
    Returns:
        (B, 2, H, W) projected velocity field (exactly div-free).
    """
    B, C, H, W = vel.shape
    assert C == 2
    M, N = H + 1, W + 1  # streamfunction grid size

    # --- Forward operator: ψ → (u, v) via forward-diff curl ----------
    def curl_fwd(psi):
        u = psi[:, 1:, :-1] - psi[:, :-1, :-1]
        v = -(psi[:, :-1, 1:] - psi[:, :-1, :-1])
        return torch.stack([u, v], dim=1)

    # --- Adjoint operator: (u, v) → ψ  (transpose of curl_fwd) -------
    def curl_adj(r):
        ru, rv = r[:, 0], r[:, 1]
        p = torch.zeros(B, M, N, device=vel.device, dtype=vel.dtype)
        p[:, 1:, :-1] += ru          # ψ[i+1,j] gets +u[i,j]
        p[:, :-1, :-1] += -ru + rv   # ψ[i,j]   gets -u[i,j] + v[i,j]
        p[:, :-1, 1:] -= rv          # ψ[i,j+1] gets -v[i,j]
        return p

    # --- Normal-equations operator: A^T A ----------------------------
    def apply_ata(psi):
        return curl_adj(curl_fwd(psi))

    # RHS = A^T vel
    b = curl_adj(vel)

    # CG solve of  A^T A ψ = A^T vel
    # (A^T A is positive semi-definite; null space = constants.
    #  b is orthogonal to constants, so CG converges to min-norm soln.)
    x = torch.zeros(B, M, N, device=vel.device, dtype=vel.dtype)
    r = b.clone()
    p = r.clone()
    rs_old = (r * r).sum(dim=(-2, -1), keepdim=True)

    for _ in range(cg_iters):
        ap = apply_ata(p)
        pap = (p * ap).sum(dim=(-2, -1), keepdim=True)
        alpha = rs_old / (pap + 1e-30)
        x = x + alpha * p
        r = r - alpha * ap
        rs_new = (r * r).sum(dim=(-2, -1), keepdim=True)
        if rs_new.max().item() < 1e-20:
            break
        beta = rs_new / (rs_old + 1e-30)
        p = r + beta * p
        rs_old = rs_new

    # Reconstruct velocity from ψ — SAME as ForwardDiffDivFreeNoise
    return curl_fwd(x)


def spectral_project_div_free(vel: torch.Tensor) -> torch.Tensor:
    """Exact divergence-free projection via FFT (Helmholtz decomposition).

    Removes the irrotational (curl-free / divergent) component at every
    Fourier mode:  v_hat_df = v_hat - k (k . v_hat) / |k|^2

    Assumes periodic boundary conditions, which is an acceptable
    approximation for removing the boundary divergence introduced by
    RePaint's copy-paste step.

    Args:
        vel: (B, 2, H, W) velocity field tensor.
    Returns:
        (B, 2, H, W) divergence-free velocity field.
    """
    _, _, H, W = vel.shape
    device = vel.device

    u_hat = torch.fft.rfft2(vel[:, 0])   # (B, H, W//2+1)
    v_hat = torch.fft.rfft2(vel[:, 1])

    ky = (torch.fft.fftfreq(H, device=device) * 2 * torch.pi).unsqueeze(1)   # (H, 1)
    kx = (torch.fft.rfftfreq(W, device=device) * 2 * torch.pi).unsqueeze(0)  # (1, W//2+1)

    k_sq = kx ** 2 + ky ** 2
    k_sq[0, 0] = 1.0  # avoid division by zero at DC

    # Remove the irrotational (divergent) component
    k_dot_v = kx * u_hat + ky * v_hat
    u_hat = u_hat - kx * k_dot_v / k_sq
    v_hat = v_hat - ky * k_dot_v / k_sq

    u_df = torch.fft.irfft2(u_hat, s=(H, W))
    v_df = torch.fft.irfft2(v_hat, s=(H, W))

    return torch.stack([u_df, v_df], dim=1)


# ──────────────────────────────────────────────────────────────────────
# Guided-diffusion inpainting
# ──────────────────────────────────────────────────────────────────────

def guided_inpaint(
    ddpm,
    input_image,
    mask,
    n_samples=1,
    device=None,
    channels=2,
    height=64,
    width=128,
    noise_strategy=None,
    guidance_scale_boundary=1.0,
    guidance_scale_div=0.5,
    return_debug=False,
):
    """Inpaint using classifier-free guidance on boundary + divergence losses.

    Instead of RePaint's copy-paste, we run the *full* reverse process and
    steer the trajectory with gradients of two losses:

        L = λ_b ‖(x̂₀ - x_known)·(1-mask)‖²  +  λ_d ‖∇·x̂₀‖²

    At each reverse step we:
      1. Predict noise ε_θ  (with gradients through the UNet)
      2. Estimate x̂₀ via Tweedie's formula
      3. Compute L on x̂₀
      4. Backprop to get ∇_{x_t} L
      5. Subtract guidance_scale * grad from the next-step mean

    This avoids copy-paste entirely so there are no boundary artefacts
    and no magnitude-crushing from projection.

    Args:
        ddpm: GaussianDDPM model (eval mode).
        input_image: (1, 2, H, W) standardised input.
        mask: (1, 2, H, W) mask, 1=unknown / 0=known.
        guidance_scale_boundary: strength of known-value matching.
        guidance_scale_div: strength of divergence-free penalty.
    """
    if noise_strategy is None:
        noise_strategy = dd.get_noise_strategy()

    device = device or dd.get_device()
    noise_strat = noise_strategy

    input_img = input_image.clone().to(device)
    mask_dev = mask.to(device)
    known_mask = 1.0 - mask_dev            # 1 where known
    single_known = known_mask[:, 0:1]      # (1,1,H,W) for scalar ops

    # Pre-noise the known image to every level (for reference, not copy-paste)
    # We only need x_known (standardised) for the boundary loss.
    x_known = input_img                    # already standardised

    # Initial noise
    x = noise_strat(
        torch.zeros(n_samples, channels, height, width, device=device),
        torch.tensor([ddpm.n_steps - 1], device=device),
    )

    # Ensure model is in eval mode but weights require no grad (we grad w.r.t. x_t)
    ddpm.eval()
    for p in ddpm.parameters():
        p.requires_grad_(False)

    with tqdm(total=ddpm.n_steps, desc="Guided denoising") as pbar:
        for t in range(ddpm.n_steps - 1, -1, -1):
            alpha_t = ddpm.alphas[t].to(device)
            alpha_bar_t = ddpm.alpha_bars[t].to(device)
            beta_t = ddpm.betas[t].to(device)

            # ── forward pass with gradients w.r.t. x_t ──────────────
            x_in = x.detach().requires_grad_(True)
            time_tensor = torch.full(
                (n_samples, 1), t, device=device, dtype=torch.long
            )
            eps_theta = ddpm.backward(x_in, time_tensor)

            # Tweedie's estimate of x_0
            x0_hat = (
                x_in - (1 - alpha_bar_t).sqrt() * eps_theta
            ) / alpha_bar_t.sqrt().clamp(min=1e-8)

            # ── guidance losses ──────────────────────────────────────
            # Boundary matching: predicted x0 should match known vals
            diff_known = (x0_hat - x_known) * known_mask
            loss_boundary = (diff_known ** 2).sum() / known_mask.sum().clamp(min=1)

            # Divergence-free: ∇·x̂₀ should be zero
            div_x0 = divergence_2d_diffable(x0_hat)
            loss_div = (div_x0 ** 2).mean()

            loss = (
                guidance_scale_boundary * loss_boundary
                + guidance_scale_div * loss_div
            )

            # ── gradient step ────────────────────────────────────────
            grad = torch.autograd.grad(loss, x_in)[0]

            # ── standard DDPM reverse step (no copy-paste!) ──────────
            with torch.no_grad():
                # Mean of p(x_{t-1} | x_t)
                mu = (1.0 / alpha_t.sqrt()) * (
                    x_in - ((1 - alpha_t) / (1 - alpha_bar_t).sqrt()) * eps_theta
                )
                # Apply guidance
                mu = mu - grad

                if t > 0:
                    sigma_t = beta_t.sqrt()
                    z = noise_strat(
                        torch.zeros_like(x),
                        torch.tensor([t], device=device),
                    )
                    x = mu + sigma_t * z
                else:
                    x = mu

            pbar.update(1)

    # Force the known region to exact original values
    result = x_known * known_mask + x * mask_dev

    if return_debug:
        return result, {"x0_hat_final": x0_hat.detach()}
    return result


def x0_predict_inpaint(
    ddpm,
    input_image,
    mask,
    t_inference=100,
    n_samples=1,
    device=None,
    channels=2,
    height=64,
    width=128,
    noise_strategy=None,
    n_avg=1,
):
    """Single-step x₀-prediction inpainting.

    Instead of 250-step iterative denoising, this:
      1. Noises the known region to level t_inference
      2. Replaces known region of x_t with independent noise (mask_xt)
      3. Runs the model ONCE → predicts x₀ directly
      4. Pastes back exact known values

    The model must have been trained with prediction_target='x0'.

    Args:
        ddpm: GaussianDDPM with x₀-predicting network.
        input_image: (1, 2, H, W) standardised input.
        mask: (1, 2, H, W) mask, 1=unknown / 0=known.
        t_inference: noise level to use (0 = no noise, 249 = max noise).
        n_samples: number of samples (typically 1).
        noise_strategy: noise generation strategy.
        n_avg: number of predictions to average (ensemble).

    Returns:
        (1, 2, H, W) inpainted result in standardised space.
    """
    if noise_strategy is None:
        noise_strategy = dd.get_noise_strategy()
    if device is None:
        device = dd.get_device()

    input_img = input_image.clone().to(device)
    mask_dev = mask.to(device)
    known_mask = 1.0 - mask_dev                      # 1 where known

    # Single-channel mask (1=missing)
    mask_single = mask_dev[:, 0:1]                    # (1, 1, H, W)

    # Known values in standardised space, zeroed in missing region
    known_values = input_img * known_mask             # (1, 2, H, W)

    ddpm.eval()
    predictions = []

    with torch.no_grad():
        for _ in range(n_avg):
            time_tensor = torch.full(
                (n_samples, 1), t_inference, device=device, dtype=torch.long
            )

            # Create x_t at noise level t_inference
            # Forward: x_t = sqrt(α̅_t) * x₀ + sqrt(1-α̅_t) * ε
            alpha_bar_t = ddpm.alpha_bars[t_inference].to(device)
            noise = noise_strategy(
                torch.zeros(n_samples, channels, height, width, device=device),
                time_tensor[:, 0],
            )

            # IMPORTANT: only noise the known region from ground truth.
            # In the missing region we use pure noise — we don't have
            # ground truth there at inference time.
            pure_noise = noise_strategy(
                torch.zeros(n_samples, channels, height, width, device=device),
                time_tensor[:, 0],
            )
            x_t_known = alpha_bar_t.sqrt() * input_img + (1 - alpha_bar_t).sqrt() * noise
            x_t = x_t_known * known_mask[:, :2] + pure_noise * mask_dev[:, :2]

            # mask_xt: replace known region with independent noise
            # (must match training — model never sees clean known values in x_t)
            indep_noise = torch.randn_like(x_t)
            x_for_model = x_t * mask_dev[:, :2] + indep_noise * known_mask[:, :2]

            # Build 5-channel input
            x_cond = torch.cat([x_for_model, mask_single, known_values], dim=1)

            # Single forward pass → x₀ prediction
            x0_pred = ddpm.network(x_cond, time_tensor)
            predictions.append(x0_pred)

    # Average predictions (if n_avg > 1)
    x0_pred = torch.stack(predictions, dim=0).mean(dim=0)

    # Force known region to exact original values
    result = input_img * known_mask + x0_pred * mask_dev

    return result


def x0_full_reverse_inpaint(
    ddpm,
    input_image,
    mask,
    n_samples=1,
    device=None,
    channels=2,
    height=64,
    width=128,
    noise_strategy=None,
    mask_xt=True,
    repaint_steps=0,
    project_steps=0,
):
    """Full-reverse x₀-prediction inpainting (correct algorithm).

    For a model trained with prediction_target='x0' and mask_xt=True.

    This runs the FULL 250-step reverse diffusion process using the
    DDPM posterior mean parameterized by the predicted x̂₀:

        q(x_{t-1} | x_t, x̂₀) = N(x_{t-1}; μ̃_t, β̃_t I)

    where:
        μ̃_t = (√ᾱ_{t-1} · β_t)/(1-ᾱ_t) · x̂₀
             + (√α_t · (1-ᾱ_{t-1}))/(1-ᾱ_t) · x_t

    This is mathematically equivalent to converting x̂₀ → ε_equiv and
    using the standard ε-prediction formula, but more numerically direct.

    Why full reverse works here:
        During training, x_t(missing) = √ᾱ_t · x₀ + √(1-ᾱ_t) · ε.
        At t=T-1, ᾱ_{T-1} ≈ 0, so both training and inference see ~pure noise.
        The posterior chain progressively builds signal, matching the training
        distribution at each step — no train/inference mismatch.

    Args:
        ddpm: GaussianDDPM with x₀-predicting FiLM UNet (5-ch input).
        input_image: (1, 2, H, W) standardised input.
        mask: (1, 2, H, W) mask, 1=unknown / 0=known.
        mask_xt: if True, replace known region of x_t with independent noise
                 (must match training config).
        repaint_steps: if > 0, do RePaint-style resampling at each step.
        noise_strategy: noise generation strategy.
        project_steps: if > 0, apply spectral divergence-free projection
                 after the copy-paste step at each reverse step, cycling
                 project_steps times (project → restore-known → repeat).

    Returns:
        (1, 2, H, W) inpainted result in standardised space.
    """
    if noise_strategy is None:
        noise_strategy = dd.get_noise_strategy()
    if device is None:
        device = dd.get_device()

    input_img = input_image.clone().to(device)
    mask_dev = mask.to(device)
    known_mask = 1.0 - mask_dev                      # 1 where known

    # Single-channel mask (1=missing)
    mask_single = mask_dev[:, 0:1]                    # (1, 1, H, W)

    # Known values in standardised space, zeroed in missing region
    known_values = input_img * known_mask             # (1, 2, H, W)

    # Start from pure noise
    x = noise_strategy(
        torch.zeros(n_samples, channels, height, width, device=device),
        torch.tensor([ddpm.n_steps - 1], device=device),
    )

    ddpm.eval()
    with torch.no_grad():
        for t in tqdm(range(ddpm.n_steps - 1, -1, -1), desc="x₀-pred reverse"):
            n_resample = (repaint_steps if t > 0 else 1)

            for r in range(n_resample):
                alpha_t = ddpm.alphas[t].to(device)
                alpha_bar_t = ddpm.alpha_bars[t].to(device)
                beta_t = ddpm.betas[t].to(device)

                time_tensor = torch.full(
                    (n_samples, 1), t, device=device, dtype=torch.long
                )

                # ---- Build 5-channel input [x_for_model, mask, known_values] ----
                if mask_xt:
                    # Replace known region with independent noise (match training)
                    indep_noise = torch.randn_like(x)
                    x_for_model = x * mask_dev[:, :2] + indep_noise * known_mask[:, :2]
                else:
                    x_for_model = x

                x_cond = torch.cat([x_for_model, mask_single, known_values], dim=1)

                # ---- Model predicts x̂₀ (NOT ε) ----
                x0_pred = ddpm.network(x_cond, time_tensor)

                # ---- DDPM posterior: q(x_{t-1} | x_t, x̂₀) ----
                if t > 0:
                    alpha_bar_prev = ddpm.alpha_bars[t - 1].to(device)

                    # Posterior mean
                    # μ̃ = (√ᾱ_{t-1} · β_t)/(1-ᾱ_t) · x̂₀
                    #    + (√α_t · (1-ᾱ_{t-1}))/(1-ᾱ_t) · x_t
                    coeff_x0 = (alpha_bar_prev.sqrt() * beta_t) / (1 - alpha_bar_t)
                    coeff_xt = (alpha_t.sqrt() * (1 - alpha_bar_prev)) / (1 - alpha_bar_t)
                    mu = coeff_x0 * x0_pred + coeff_xt * x

                    # Posterior variance: β̃_t = (1-ᾱ_{t-1})/(1-ᾱ_t) · β_t
                    beta_tilde = ((1 - alpha_bar_prev) / (1 - alpha_bar_t)) * beta_t
                    sigma_t = beta_tilde.sqrt()

                    z = noise_strategy(
                        torch.zeros_like(x),
                        torch.tensor([t], device=device),
                    )
                    x = mu + sigma_t * z
                else:
                    # At t=0: x = x̂₀ (no noise added)
                    x = x0_pred

                # ---- Optional RePaint: paste noised known region ----
                if repaint_steps > 0 and t > 0:
                    alpha_bar_prev = ddpm.alpha_bars[t - 1].to(device)
                    noise_known = noise_strategy(
                        torch.zeros_like(input_img),
                        torch.tensor([t - 1], device=device),
                    )
                    x_known = (alpha_bar_prev.sqrt() * input_img
                               + (1 - alpha_bar_prev).sqrt() * noise_known)
                    x = x_known * known_mask + x * mask_dev

                    # Re-noise for next resample pass (if not the last one)
                    if r < n_resample - 1:
                        noise_back = noise_strategy(
                            torch.zeros_like(x), torch.tensor([t], device=device)
                        )
                        x = (alpha_bar_t.sqrt() / alpha_bar_prev.sqrt() * x
                             + (1 - alpha_bar_t / alpha_bar_prev).sqrt() * noise_back)

    # Force known region to exact original values
    result = input_img * known_mask + x * mask_dev

    # Optional: forward-difference div-free projection on the final result
    if project_steps > 0:
        for _ in range(project_steps):
            pre_energy = (result ** 2).sum()
            result = forward_diff_project_div_free(result)
            post_energy = (result ** 2).sum()
            if post_energy > 1e-12:
                result = result * (pre_energy / post_energy).sqrt()
            result = input_img * known_mask + result * mask_dev

    return result


def mask_aware_inpaint_cfg(
    ddpm,
    input_image,
    mask,
    n_samples=1,
    device=None,
    channels=2,
    height=64,
    width=128,
    noise_strategy=None,
    guidance_scale=3.0,
):
    """Inpaint using Classifier-Free Guidance with mask-aware DDPM.

    At each denoising step, does TWO forward passes:
      - eps_cond:   UNet([x_t, mask, known_values], t)  (conditioned)
      - eps_uncond: UNet([x_t, 0,    0            ], t)  (unconditional)

    Then blends:  eps = eps_uncond + w * (eps_cond - eps_uncond)
    where w = guidance_scale.  w=1 => normal conditional, w>1 amplifies conditioning.

    Args:
        ddpm: GaussianDDPM with MyUNet_Inpaint (5-ch, trained with p_uncond > 0).
        input_image: (1, 2, H, W) standardised input.
        mask: (1, 2, H, W) mask, 1=unknown / 0=known.
        guidance_scale: CFG weight (try 2–7, higher = stronger conditioning).
        noise_strategy: noise generation strategy.

    Returns:
        (1, 2, H, W) inpainted result in standardised space.
    """
    if noise_strategy is None:
        noise_strategy = dd.get_noise_strategy()
    if device is None:
        device = dd.get_device()

    input_img = input_image.clone().to(device)
    mask_dev = mask.to(device)
    known_mask = 1.0 - mask_dev                      # 1 where known
    mask_single = mask_dev[:, 0:1]                    # (1, 1, H, W)
    known_values = input_img * known_mask             # (1, 2, H, W)

    # Unconditional inputs (zeros for mask & known)
    zeros_mask = torch.zeros_like(mask_single)        # (1, 1, H, W)
    zeros_known = torch.zeros_like(known_values)      # (1, 2, H, W)

    # Start from pure noise
    x = noise_strategy(
        torch.zeros(n_samples, channels, height, width, device=device),
        torch.tensor([ddpm.n_steps - 1], device=device),
    )

    w = guidance_scale

    ddpm.eval()
    with torch.no_grad():
        for t in tqdm(range(ddpm.n_steps - 1, -1, -1), desc="CFG denoising"):
            alpha_t = ddpm.alphas[t].to(device)
            alpha_bar_t = ddpm.alpha_bars[t].to(device)
            beta_t = ddpm.betas[t].to(device)

            time_tensor = torch.full(
                (n_samples, 1), t, device=device, dtype=torch.long
            )

            # Conditioned forward pass
            x_cond = torch.cat([x, mask_single, known_values], dim=1)
            eps_cond = ddpm.network(x_cond, time_tensor)

            # Unconditional forward pass
            x_uncond = torch.cat([x, zeros_mask, zeros_known], dim=1)
            eps_uncond = ddpm.network(x_uncond, time_tensor)

            # CFG blending
            eps_theta = eps_uncond + w * (eps_cond - eps_uncond)

            # Standard DDPM reverse step
            mu = (1.0 / alpha_t.sqrt()) * (
                x - ((1 - alpha_t) / (1 - alpha_bar_t).sqrt()) * eps_theta
            )

            if t > 0:
                sigma_t = beta_t.sqrt()
                z = noise_strategy(
                    torch.zeros_like(x),
                    torch.tensor([t], device=device),
                )
                x = mu + sigma_t * z
            else:
                x = mu

    # Force the known region to exact original values
    result = input_img * known_mask + x * mask_dev
    return result


def mask_aware_inpaint(
    ddpm,
    input_image,
    mask,
    n_samples=1,
    device=None,
    channels=2,
    height=64,
    width=128,
    noise_strategy=None,
    mask_xt=False,
):
    """Inpaint using a Palette-style mask-aware DDPM.

    The model's UNet accepts 5-channel input: [x_t, mask, known_values].
    We run the standard reverse process, passing conditioning at every step.
    No copy-paste, no projection, no gradient guidance needed — the model
    was trained to denoise conditioned on the mask and known values.

    Args:
        ddpm: GaussianDDPM with a MyUNet_Inpaint network (5-channel input).
        input_image: (1, 2, H, W) standardised input.
        mask: (1, 2, H, W) mask, 1=unknown / 0=known.
        n_samples: number of samples to generate (typically 1).
        noise_strategy: noise generation strategy.
        mask_xt: if True, replace x_t in known region with independent noise
                 (must match training -- used with mask_xt training).

    Returns:
        (1, 2, H, W) inpainted result in standardised space.
    """
    if noise_strategy is None:
        noise_strategy = dd.get_noise_strategy()
    if device is None:
        device = dd.get_device()

    input_img = input_image.clone().to(device)
    mask_dev = mask.to(device)
    known_mask = 1.0 - mask_dev                      # 1 where known

    # Single-channel mask for conditioning (1=missing)
    mask_single = mask_dev[:, 0:1]                    # (1, 1, H, W)

    # Known values: standardised input zeroed in missing region
    known_values = input_img * known_mask             # (1, 2, H, W)

    # Start from pure noise
    x = noise_strategy(
        torch.zeros(n_samples, channels, height, width, device=device),
        torch.tensor([ddpm.n_steps - 1], device=device),
    )

    ddpm.eval()
    with torch.no_grad():
        for t in tqdm(range(ddpm.n_steps - 1, -1, -1), desc="Mask-aware denoising"):
            alpha_t = ddpm.alphas[t].to(device)
            alpha_bar_t = ddpm.alpha_bars[t].to(device)
            beta_t = ddpm.betas[t].to(device)

            time_tensor = torch.full(
                (n_samples, 1), t, device=device, dtype=torch.long
            )

            # Build 5-channel input: [x_t, mask, known_values]
            # If mask_xt: replace known region of x with independent noise
            # so model sees same distribution as during training
            if mask_xt:
                indep_noise = torch.randn_like(x)
                x_for_model = x * mask_dev[:, :2] + indep_noise * known_mask[:, :2]
            else:
                x_for_model = x
            x_cond = torch.cat([x_for_model, mask_single, known_values], dim=1)

            # UNet predicts noise
            eps_theta = ddpm.network(x_cond, time_tensor)

            # Standard DDPM reverse step
            mu = (1.0 / alpha_t.sqrt()) * (
                x - ((1 - alpha_t) / (1 - alpha_bar_t).sqrt()) * eps_theta
            )

            if t > 0:
                sigma_t = beta_t.sqrt()
                z = noise_strategy(
                    torch.zeros_like(x),
                    torch.tensor([t], device=device),
                )
                x = mu + sigma_t * z
            else:
                x = mu

    # Force the known region to exact original values
    result = input_img * known_mask + x * mask_dev
    return result


def repaint_standard(
    ddpm,
    input_image,
    mask,
    n_samples=1,
    device=None,
    channels=2,
    height=64,
    width=128,
    noise_strategy=None,
    prediction_target="x0",
    resample_steps=5,
    project_div_free=False,
    project_final_steps=0,
):
    """Standard RePaint inpainting (Lugmayr et al., 2022).

    Works with an UNCONDITIONAL DDPM (standard UNet, 2-channel input).
    No conditioning channels, no clamping — textbook RePaint algorithm
    with optional divergence-free projection.

    Supports both x₀-prediction and ε-prediction models.

    At each reverse step t:
      1. Denoise x_t → x_{t-1}  (via DDPM posterior)
      2. Forward-noise known region to t-1:
           x_{t-1}^known = √ᾱ_{t-1} · x₀ + √(1-ᾱ_{t-1}) · ε
      3. Paste:  x_{t-1} = x_{t-1}^known · known_mask + x_{t-1}^unknown · miss_mask
      4. (Optional) CG div-free projection to fix divergence at the paste boundary.
      5. Optionally resample (re-noise to x_t and repeat step 1-3).

    All noise samples use `noise_strategy`, so if you pass
    ForwardDiffDivFreeNoise the entire process is divergence-free.

    Args:
        ddpm: GaussianDDPM with a standard UNet.
        input_image: (1, 2, H, W) standardised input (known values).
        mask: (1, 2, H, W) mask, 1 = missing (to inpaint), 0 = known.
        n_samples: batch size (typically 1).
        noise_strategy: NoiseStrategy instance; defaults to dd.get_noise_strategy().
        prediction_target: "x0" or "eps" — what the UNet was trained to predict.
        resample_steps: number of RePaint resample iterations per timestep.
        project_div_free: if True, apply forward-difference CG div-free projection
            after each copy-paste step to fix the divergence introduced at the
            known/unknown boundary.  Energy is preserved via rescaling.
        project_final_steps: if > 0, apply CG projection this many times to the
            final result (project → restore known region → repeat).

    Returns:
        (1, 2, H, W) inpainted result in standardised space.
    """
    if noise_strategy is None:
        noise_strategy = dd.get_noise_strategy()
    if device is None:
        device = dd.get_device()

    input_img = input_image.clone().to(device)
    mask_dev = mask.to(device)
    known_mask = 1.0 - mask_dev  # 1 where known

    # Start from pure noise
    x = noise_strategy(
        torch.zeros(n_samples, channels, height, width, device=device),
        torch.tensor([ddpm.n_steps - 1], device=device),
    )

    ddpm.eval()
    with torch.no_grad():
        for t in tqdm(range(ddpm.n_steps - 1, -1, -1), desc="RePaint"):
            n_resample = resample_steps if t > 0 else 1

            for r in range(n_resample):
                alpha_t = ddpm.alphas[t].to(device)
                alpha_bar_t = ddpm.alpha_bars[t].to(device)
                beta_t = ddpm.betas[t].to(device)

                time_tensor = torch.full(
                    (n_samples, 1), t, device=device, dtype=torch.long
                )

                # --- Reverse step: x_t → x_{t-1} ---
                net_out = ddpm.backward(x, time_tensor)

                if prediction_target == "x0":
                    x0_pred = net_out
                else:  # eps
                    # Convert ε → x̂₀
                    x0_pred = (
                        x - (1 - alpha_bar_t).sqrt() * net_out
                    ) / alpha_bar_t.sqrt().clamp(min=1e-8)

                if t > 0:
                    alpha_bar_prev = ddpm.alpha_bars[t - 1].to(device)

                    # Posterior mean: μ̃_t
                    coeff_x0 = (alpha_bar_prev.sqrt() * beta_t) / (1 - alpha_bar_t)
                    coeff_xt = (alpha_t.sqrt() * (1 - alpha_bar_prev)) / (1 - alpha_bar_t)
                    mu = coeff_x0 * x0_pred + coeff_xt * x

                    # Posterior variance: β̃_t
                    beta_tilde = ((1 - alpha_bar_prev) / (1 - alpha_bar_t)) * beta_t
                    sigma_t = beta_tilde.sqrt()

                    z = noise_strategy(
                        torch.zeros_like(x),
                        torch.tensor([t], device=device),
                    )
                    x_denoised = mu + sigma_t * z
                else:
                    x_denoised = x0_pred

                # --- RePaint: paste forward-noised known region ---
                if t > 0:
                    alpha_bar_prev = ddpm.alpha_bars[t - 1].to(device)
                    noise_known = noise_strategy(
                        torch.zeros_like(input_img),
                        torch.tensor([t - 1], device=device),
                    )
                    x_known = (alpha_bar_prev.sqrt() * input_img
                               + (1 - alpha_bar_prev).sqrt() * noise_known)
                    x = x_known * known_mask + x_denoised * mask_dev
                else:
                    # t=0: paste clean known region
                    x = input_img * known_mask + x_denoised * mask_dev

                # --- CG div-free projection after paste (fixes boundary divergence) ---
                if project_div_free and t > 0:
                    pre_energy = (x ** 2).sum()
                    x = forward_diff_project_div_free(x)
                    post_energy = (x ** 2).sum()
                    if post_energy > 1e-12:
                        x = x * (pre_energy / post_energy).sqrt()
                    # Re-stamp known region after projection
                    if t > 0:
                        x = x_known * known_mask + x * mask_dev
                    else:
                        x = input_img * known_mask + x * mask_dev

                # --- Resample: re-noise x_{t-1} → x_t for next iteration ---
                if r < n_resample - 1 and t > 0:
                    alpha_bar_prev = ddpm.alpha_bars[t - 1].to(device)
                    noise_back = noise_strategy(
                        torch.zeros_like(x),
                        torch.tensor([t], device=device),
                    )
                    # q(x_t | x_{t-1}) forward step
                    x = (alpha_t.sqrt() * x
                         + (1 - alpha_t).sqrt() * noise_back)

    # --- Final CG projection (optional, project → restore known → repeat) ---
    if project_final_steps > 0:
        for _ in range(project_final_steps):
            pre_energy = (x ** 2).sum()
            x = forward_diff_project_div_free(x)
            post_energy = (x ** 2).sum()
            if post_energy > 1e-12:
                x = x * (pre_energy / post_energy).sqrt()
            x = input_img * known_mask + x * mask_dev

    return x


def inpaint_generate_new_images(ddpm, input_image, mask, n_samples=16, device=None,
                                resample_steps=1, channels=2, height=64, width=128,
                                noise_strategy=dd.get_noise_strategy(), return_debug=False):
    """
    Given a DDPM model, an input image, and a mask, generates in-painted samples.
    
    The boundary fix is applied at EVERY denoising step so the neural network
    always sees clean, physically consistent inputs (no boundary discontinuities).
    """
    noised_images = [None] * (ddpm.n_steps + 1)
    device = dd.get_device()

    def denoise_one_step(noisy_img, noise_strat, t):
        time_tensor = torch.full((n_samples, 1), t, device=device, dtype=torch.long)
        epsilon_theta = ddpm.backward(noisy_img, time_tensor)

        alpha_t = ddpm.alphas[t].to(device)
        alpha_t_bar = ddpm.alpha_bars[t].to(device)

        if noise_strat.get_gaussian_scaling():
            less_noised_img = (1 / alpha_t.sqrt()) * (
                    noisy_img - ((1 - alpha_t) / (1 - alpha_t_bar).sqrt()) * epsilon_theta
            )
        else:
            less_noised_img = (1 / alpha_t.sqrt()) * (noisy_img - epsilon_theta)

        tensor_size = torch.zeros(n_samples, channels, height, width, device=device)

        if t > 0:
            z = noise_strat(tensor_size, torch.tensor([t], device=device))
            beta_t = ddpm.betas[t].to(device)
            sigma_t = beta_t.sqrt()
            less_noised_img = less_noised_img + sigma_t * z
            noise = z
        else:
            noise = None

        return less_noised_img, noise

    def noise_one_step(unnoised_img, t, noise_strat):
        time_tensor = torch.tensor([t], device=unnoised_img.device)
        epsilon = noise_strat(unnoised_img, time_tensor)
        noised_img = ddpm(unnoised_img, t, epsilon, one_step=True)
        return noised_img

    def assert_finite(tensor, label, t, i):
        if not torch.isfinite(tensor).all():
            raise ValueError(
                f"NaN/inf detected at t={t}, resample={i}, stage={label}"
            )

    def clamp_magnitude(vel, max_mag, eps=1e-6):
        mag = torch.sqrt((vel ** 2).sum(dim=1, keepdim=True))
        scale = torch.clamp(max_mag / (mag + eps), max=1.0)
        return vel * scale

    with torch.no_grad():
        noise_strat = noise_strategy

        enable_projection = bool(dd.get_attribute("enable_divergence_projection"))
        clamp_project_cycles = int(dd.get_attribute("clamp_project_cycles") or 5)
        jacobi_iters = int(dd.get_attribute("poisson_jacobi_iters") or 200)
        dx = float(dd.get_attribute("poisson_dx") or 1.0)
        dy = float(dd.get_attribute("poisson_dy") or 1.0)
        clamp_scale = float(dd.get_attribute("resample_clamp_scale") or 3.0)

        input_img = input_image.clone().to(device)
        mask = mask.to(device)

        # Mask convention: 1.0 = missing (to inpaint), 0.0 = known.
        noise = noise_strat(input_img, torch.tensor([ddpm.n_steps - 1], device=device))

        effective_resample_steps = resample_steps

        max_known_mag = torch.sqrt((input_img ** 2).sum(dim=1, keepdim=True)).max()
        max_allowed_mag = max_known_mag * clamp_scale

        # Step-by-step forward noising
        noised_images[0] = input_img
        for t in range(ddpm.n_steps):
            noised_images[t + 1] = noise_one_step(noised_images[t], t, noise_strat)

        doing_the_thing = True

        if doing_the_thing:
            x = noised_images[ddpm.n_steps] * (1 - mask) + (noise * mask)
        else:
            x = masked_poisson_projection(noised_images[ddpm.n_steps], mask)

        with tqdm(total=ddpm.n_steps, desc="Denoising") as pbar:
            last_inpainted = None
            for idx, t in enumerate(range(ddpm.n_steps - 1, -1, -1)):
                for i in range(effective_resample_steps):
                    inpainted, noise = denoise_one_step(x, noise_strat, t)
                    assert_finite(inpainted, "denoise", t, i)
                    last_inpainted = inpainted
                    known = noised_images[t - 1] if t > 0 else noised_images[0]

                    combined = combine_fields(known, inpainted, mask)

                    # Spectral project-then-restore at every step.
                    # The copy-paste above creates boundary divergence;
                    # exact FFT projection removes it, then we restore
                    # known values.
                    if enable_projection:
                        for _ in range(clamp_project_cycles):
                            # Measure energy before projection
                            pre_energy = (combined ** 2).sum()

                            combined = spectral_project_div_free(combined)

                            # Renormalize: scale projected field back to
                            # pre-projection energy so the projection only
                            # rotates toward div-free without draining magnitude.
                            post_energy = (combined ** 2).sum()
                            if post_energy > 1e-12:
                                scale = (pre_energy / post_energy).sqrt()
                                combined = combined * scale

                            # Restore known region exactly
                            combined = known * (1 - mask) + combined * mask

                    if (i + 1) < effective_resample_steps and t > 0:
                        clamped = clamp_magnitude(combined, max_allowed_mag)
                        x = noise_one_step(clamped, t - 1, noise_strat)
                        assert_finite(x, "resample_noise", t, i)
                    else:
                        x = combined  # Pass combined to next timestep
                pbar.update(1)

    # Force known region to be original input (CombNet may have modified it)
    result = input_img * (1 - mask) + combined * mask
    if return_debug:
        return result, {
            "last_inpainted": last_inpainted,
            "last_combined": combined,
        }
    return result


def calculate_mse(original_image, predicted_image, mask, normalize=False):
    """
    Calculates masked MSE between original and predicted image.
    Optionally normalizes both using original_image's masked region stats.

    Args:
        original_image: (1, 2, H, W)
        predicted_image: (1, 2, H, W)
        mask: (1, 2, H, W)
        normalize: bool, whether to normalize both images using shared scale

    Returns:
        Scalar MSE
    """
    single_mask = mask[:, 0:1, :, :]  # shape (1, 1, H, W)

    if normalize:
        original_image, predicted_image = normalize_pair(original_image, predicted_image, single_mask)

    squared_error = (original_image - predicted_image) ** 2  # (1, 2, H, W)
    per_pixel_error = squared_error.sum(dim=1, keepdim=True)  # (1, 1, H, W)

    masked_error = per_pixel_error * single_mask
    total_error = masked_error.sum()
    num_valid_pixels = single_mask.sum()

    if num_valid_pixels == 0:
        return torch.tensor(float('nan'))

    return total_error / num_valid_pixels


def calculate_percent_error(original_image, predicted_image, mask):
    """
    Calculates masked percent error between original and predicted image.
    Optionally normalizes both using original_image's masked region stats.

    Args:
        original_image: (1, 2, H, W)
        predicted_image: (1, 2, H, W)
        mask: (1, 2, H, W)

    Returns:
        Scalar MSE
    """
    single_mask = mask[:, 0:1, :, :]  # shape (1, 1, H, W)

    percent_error = ( torch.abs( (predicted_image - original_image) / original_image ) )  # (1, 2, H, W)
    per_pixel_error = percent_error.sum(dim=1, keepdim=True)  # (1, 1, H, W)

    masked_error = per_pixel_error * single_mask
    total_error = masked_error.nansum()
    num_valid_pixels = single_mask.nansum()

    if num_valid_pixels == 0:
        return torch.tensor(float('nan'))

    return total_error / num_valid_pixels


def normalize_pair(original_img, predicted_img, mask):
    """
    Normalize both images to [0, 1] using the min/max of the original image
    over the masked region, applied per channel.

    Args:
        original_img, predicted_img: (1, C, H, W)
        mask: (1, 1, H, W)

    Returns:
        Tuple of normalized (original_img, predicted_img)
    """
    B, C, H, W = original_img.shape
    norm_original = torch.zeros_like(original_img)
    norm_predicted = torch.zeros_like(predicted_img)

    for c in range(C):
        masked_pixels = original_img[0, c][mask[0, 0].bool()]
        if masked_pixels.numel() == 0:
            # Avoid div-by-zero
            norm_original[0, c] = original_img[0, c]
            norm_predicted[0, c] = predicted_img[0, c]
            continue

        min_val = masked_pixels.min()
        max_val = masked_pixels.max()
        range_val = max_val - min_val + 1e-8  # avoid divide-by-zero

        norm_original[0, c] = (original_img[0, c] - min_val) / range_val
        norm_predicted[0, c] = (predicted_img[0, c] - min_val) / range_val

    return norm_original, norm_predicted


def top_left_crop(tensor, crop_h, crop_w):
    """
    Crop the top-left corner of a tensor of shape (1, 2, H, W).

    Args:
        tensor: PyTorch tensor of shape (1, 2, H, W)
        crop_h: Desired crop height
        crop_w: Desired crop width

    Returns:
        Cropped tensor of shape (1, 2, crop_h, crop_w)
    """
    return tensor[:, :, :crop_h, :crop_w]


def avg_pixel_value(original_image, predicted_image, mask):
    avg_pixel_value = torch.sum(torch.abs(original_image * mask)) / mask.sum()
    avg_diff = torch.sum(torch.abs((predicted_image * mask) - (original_image * mask))) / mask.sum()
    return avg_diff * (100 / avg_pixel_value)


def masked_poisson_projection(vector_field, mask, num_iter=500, tol=1e-5):
    """
    Performs divergence-free projection of a 2D vector field with masked inpainting regions.

    Args:
        vector_field: (N, 2, H, W) torch tensor (vx, vy)
        mask:         (N, 2, H, W) binary tensor, 1 = region to inpaint
        num_iter:     max Jacobi iterations
        tol:          early stopping tolerance on residual (L2 norm)

    Returns:
        projected_field: (N, 2, H, W) divergence-free vector field
    """
    N, _, H, W = vector_field.shape
    device = vector_field.device

    vx, vy = vector_field[:, 0], vector_field[:, 1]

    # Compute divergence: ∂vx/∂x + ∂vy/∂y (forward diff)
    div = torch.zeros(N, H, W, device=device)
    div[:, :, :-1] += vx[:, :, 1:] - vx[:, :, :-1]
    div[:, :-1, :] += vy[:, 1:, :] - vy[:, :-1, :]

    # Initialize scalar potential φ
    phi = torch.zeros(N, H, W, device=device)

    # Combine mask across components: (N, H, W)
    M = torch.maximum(mask[:, 0], mask[:, 1])  # 1 where inpaint, 0 where known
    known = (1 - M)

    # Jacobi solver
    for i in range(num_iter):
        phi_new = phi.clone()

        # Sum of neighbors (up, down, left, right)
        neighbor_sum = torch.zeros_like(phi)

        neighbor_sum[:, 1:, :] += phi[:, :-1, :]    # up
        neighbor_sum[:, :-1, :] += phi[:, 1:, :]    # down
        neighbor_sum[:, :, 1:] += phi[:, :, :-1]    # left
        neighbor_sum[:, :, :-1] += phi[:, :, 1:]    # right

        # Jacobi update
        phi_new = (div + neighbor_sum) / 4.0

        # Only update masked/inpaint regions
        updated_phi = torch.where(M == 1, phi_new, phi)

        # Residual for early stopping
        residual = torch.norm(updated_phi - phi, dim=(1, 2)).mean()

        phi = updated_phi

        if residual < tol:
            break

    # Compute gradient of φ (forward diff)
    dphix = torch.zeros_like(vx)
    dphiy = torch.zeros_like(vy)

    dphix[:, :, :-1] = phi[:, :, 1:] - phi[:, :, :-1]
    dphiy[:, :-1, :] = phi[:, 1:, :] - phi[:, :-1, :]

    # Subtract gradient to get divergence-free field
    vx_proj = vx - dphix
    vy_proj = vy - dphiy

    # Restore known values (preserve unmasked regions per channel)
    vx_proj = torch.where(mask[:, 0] == 0, vx, vx_proj)
    vy_proj = torch.where(mask[:, 1] == 0, vy, vy_proj)

    return torch.stack([vx_proj, vy_proj], dim=1)
