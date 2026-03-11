"""Helmholtz-split noise schedule for DDPM.

Decomposes (u,v) into solenoidal + irrotational via FFT Helmholtz-Hodge
projection and applies independent noise schedules to each orthogonal
component.  The solenoidal schedule is slower (structure-preserving) and
the irrotational schedule is faster.

Mathematical justification: the solenoidal and irrotational subspaces are
orthogonal in L², so independent DDPM processes in each subspace compose
to a valid joint process.  The ELBO factorises across orthogonal subspaces.

Usage
-----
    split = HelmholtzSplitSchedule(n_steps=250, irr_speed=2.0, device=device)

    # Forward (noising)
    x_t = split.q_sample(x0, t)        # returns combined noisy field

    # Reverse (one posterior step)
    x_prev = split.p_step(x_t, x0_pred, t)
"""

import torch
import torch.nn.functional as F


# ── FFT Helmholtz decomposition (pure PyTorch, GPU-friendly) ──────────

def helmholtz_decompose(uv: torch.Tensor):
    """Decompose (B,2,H,W) velocity into solenoidal + irrotational via FFT.

    Returns (uv_sol, uv_irr), both (B,2,H,W).
    Exactly orthogonal; uv_sol + uv_irr == uv (up to FFT precision).
    """
    u = uv[:, 0]  # (B, H, W)
    v = uv[:, 1]

    u_f = torch.fft.fft2(u)
    v_f = torch.fft.fft2(v)

    H, W = u.shape[-2], u.shape[-1]
    ky = torch.fft.fftfreq(H, device=uv.device).reshape(1, H, 1)
    kx = torch.fft.fftfreq(W, device=uv.device).reshape(1, 1, W)
    k2 = kx ** 2 + ky ** 2
    k2_safe = k2.clone()
    k2_safe[:, 0, 0] = 1.0  # avoid div-by-zero at DC

    # Irrotational projection: project along k
    div_f = u_f * kx + v_f * ky
    phi_f = div_f / k2_safe

    u_irr_f = phi_f * kx
    v_irr_f = phi_f * ky

    u_irr = torch.fft.ifft2(u_irr_f).real
    v_irr = torch.fft.ifft2(v_irr_f).real

    # Solenoidal = residual (exactly div-free)
    u_sol = u - u_irr
    v_sol = v - v_irr

    uv_sol = torch.stack([u_sol, v_sol], dim=1)
    uv_irr = torch.stack([u_irr, v_irr], dim=1)
    return uv_sol, uv_irr


def generate_solenoidal_noise(shape, device):
    """Generate unit-variance divergence-free Gaussian noise.

    Strategy: generate isotropic Gaussian noise, project to solenoidal
    subspace, rescale to unit variance.
    """
    raw = torch.randn(shape, device=device)
    sol, _ = helmholtz_decompose(raw)
    # Rescale to unit variance (projection reduces variance)
    std = sol.std()
    if std > 1e-8:
        sol = sol / std
    return sol


def generate_irrotational_noise(shape, device):
    """Generate unit-variance curl-free Gaussian noise.

    Strategy: generate isotropic Gaussian noise, project to irrotational
    subspace, rescale to unit variance.
    """
    raw = torch.randn(shape, device=device)
    _, irr = helmholtz_decompose(raw)
    std = irr.std()
    if std > 1e-8:
        irr = irr / std
    return irr


# ── Split noise schedule ──────────────────────────────────────────────

class HelmholtzSplitSchedule:
    """Two independent linear-beta DDPM schedules for Helmholtz components.

    Parameters
    ----------
    n_steps : int
        Total diffusion timesteps (shared index t).
    min_beta, max_beta : float
        Solenoidal schedule endpoints (standard DDPM defaults).
    irr_speed : float
        Speed multiplier for irrotational schedule.  irr_speed=2.0 means
        the irrotational component's max_beta is 2× the solenoidal's,
        reaching pure noise sooner.
    device : torch.device
    """

    def __init__(self, n_steps=250, min_beta=1e-4, max_beta=0.02,
                 irr_speed=2.0, device=None):
        self.n_steps = n_steps
        self.device = device

        # Solenoidal schedule (slower)
        self.betas_sol = torch.linspace(min_beta, max_beta, n_steps, device=device)
        self.alphas_sol = 1.0 - self.betas_sol
        self.alpha_bars_sol = torch.cumprod(self.alphas_sol, dim=0)

        # Irrotational schedule (faster)
        max_beta_irr = min(max_beta * irr_speed, 0.999)  # cap at 0.999
        self.betas_irr = torch.linspace(min_beta, max_beta_irr, n_steps, device=device)
        self.alphas_irr = 1.0 - self.betas_irr
        self.alpha_bars_irr = torch.cumprod(self.alphas_irr, dim=0)

    def q_sample(self, x0, t, eps_sol=None, eps_irr=None):
        """Forward process: noise x0 with split schedules.

        Parameters
        ----------
        x0 : (B, 2, H, W) clean velocity field
        t : (B,) integer timesteps
        eps_sol, eps_irr : optional pre-generated noise tensors

        Returns
        -------
        x_t : (B, 2, H, W) noisy field (components at different SNRs)
        eps_sol, eps_irr : the noise tensors used (for loss computation)
        """
        B = x0.shape[0]

        # Decompose clean field
        x0_sol, x0_irr = helmholtz_decompose(x0)

        # Generate noise in each subspace
        if eps_sol is None:
            eps_sol = generate_solenoidal_noise(x0.shape, x0.device)
        if eps_irr is None:
            eps_irr = generate_irrotational_noise(x0.shape, x0.device)

        # Extract schedule values for this batch of timesteps
        abar_sol = self.alpha_bars_sol[t].reshape(B, 1, 1, 1)
        abar_irr = self.alpha_bars_irr[t].reshape(B, 1, 1, 1)

        # Noise each component independently
        x_t_sol = abar_sol.sqrt() * x0_sol + (1 - abar_sol).sqrt() * eps_sol
        x_t_irr = abar_irr.sqrt() * x0_irr + (1 - abar_irr).sqrt() * eps_irr

        x_t = x_t_sol + x_t_irr
        return x_t, eps_sol, eps_irr

    def p_mean_variance(self, x_t, x0_pred, t):
        """Compute posterior mean and variance for reverse step.

        Uses x0 prediction mode: given x_t and predicted x̂_0,
        computes q(x_{t-1} | x_t, x̂_0) separately for each component.

        Returns
        -------
        mu : (B, 2, H, W) posterior mean
        sigma : (B, 2, H, W) posterior std (component-wise)
        """
        B = x_t.shape[0]

        # Decompose both x_t and x0_pred
        xt_sol, xt_irr = helmholtz_decompose(x_t)
        x0_sol, x0_irr = helmholtz_decompose(x0_pred)

        mu_sol, sigma_sol = self._component_posterior(
            xt_sol, x0_sol, t,
            self.alphas_sol, self.alpha_bars_sol, self.betas_sol,
        )
        mu_irr, sigma_irr = self._component_posterior(
            xt_irr, x0_irr, t,
            self.alphas_irr, self.alpha_bars_irr, self.betas_irr,
        )

        return mu_sol + mu_irr, sigma_sol + sigma_irr

    def p_step(self, x_t, x0_pred, t, noise_sol=None, noise_irr=None):
        """One reverse diffusion step with split posteriors.

        Parameters
        ----------
        x_t : (B, 2, H, W)
        x0_pred : (B, 2, H, W) model's prediction of x_0
        t : (B,) current timesteps (will step to t-1)

        Returns
        -------
        x_prev : (B, 2, H, W)
        """
        B = x_t.shape[0]

        xt_sol, xt_irr = helmholtz_decompose(x_t)
        x0_sol, x0_irr = helmholtz_decompose(x0_pred)

        mu_sol, sigma_sol = self._component_posterior(
            xt_sol, x0_sol, t,
            self.alphas_sol, self.alpha_bars_sol, self.betas_sol,
        )
        mu_irr, sigma_irr = self._component_posterior(
            xt_irr, x0_irr, t,
            self.alphas_irr, self.alpha_bars_irr, self.betas_irr,
        )

        # Sample
        if noise_sol is None:
            noise_sol = generate_solenoidal_noise(x_t.shape, x_t.device)
        if noise_irr is None:
            noise_irr = generate_irrotational_noise(x_t.shape, x_t.device)

        # At t=0, no noise
        mask_t0 = (t == 0).float().reshape(B, 1, 1, 1)
        x_prev_sol = mu_sol + (1 - mask_t0) * sigma_sol * noise_sol
        x_prev_irr = mu_irr + (1 - mask_t0) * sigma_irr * noise_irr

        return x_prev_sol + x_prev_irr

    @staticmethod
    def _component_posterior(x_t, x0, t, alphas, alpha_bars, betas):
        """Standard DDPM posterior q(x_{t-1}|x_t, x_0) for one component."""
        B = x_t.shape[0]

        alpha_t = alphas[t].reshape(B, 1, 1, 1)
        abar_t = alpha_bars[t].reshape(B, 1, 1, 1)
        beta_t = betas[t].reshape(B, 1, 1, 1)

        # For t > 0, alpha_bar_{t-1}; for t = 0, use 1.0
        t_prev = (t - 1).clamp(min=0)
        abar_prev = alpha_bars[t_prev].reshape(B, 1, 1, 1)
        # At t=0, abar_prev should be 1.0 (no noise)
        abar_prev = torch.where(
            t.reshape(B, 1, 1, 1) == 0,
            torch.ones_like(abar_prev),
            abar_prev,
        )

        # Posterior mean
        coeff_x0 = (abar_prev.sqrt() * beta_t) / (1 - abar_t)
        coeff_xt = (alpha_t.sqrt() * (1 - abar_prev)) / (1 - abar_t)
        mu = coeff_x0 * x0 + coeff_xt * x_t

        # Posterior variance (beta_tilde)
        beta_tilde = ((1 - abar_prev) / (1 - abar_t)) * beta_t
        sigma = beta_tilde.sqrt()

        return mu, sigma
