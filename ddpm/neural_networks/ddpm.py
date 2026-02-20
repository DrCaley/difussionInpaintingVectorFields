"""Core DDPM forward/backward wrapper.

Encapsulates the noise schedule (betas, alphas, alpha_bars) and
provides:
    - ``forward(x0, t, epsilon)`` → noisy image x_t
    - ``backward(x, t)``          → UNet noise prediction

See ``ddpm.protocols.DenoiserNetworkProtocol`` for the UNet contract.

Coupling note
-------------
The ``forward()`` method queries ``DDInitializer().get_noise_strategy()``
to determine whether ε should be scaled by √(1−ᾱ).  This is a known
tight coupling to the singleton — future refactoring should inject the
noise strategy or the scaling flag at construction time.
"""

import torch
import torch.nn as nn

from data_prep.data_initializer import DDInitializer


class GaussianDDPM(nn.Module):
    """Gaussian Denoising Diffusion Probabilistic Model.

    Wraps a denoiser network (UNet) with a linear noise schedule and
    implements the DDPM forward/reverse process.

    Parameters
    ----------
    network : nn.Module
        The denoiser UNet (must accept ``(x, t)`` and return same spatial dims).
    n_steps : int
        Number of diffusion timesteps.
    min_beta, max_beta : float
        Endpoints of the linear beta schedule.
    device : torch.device
        Target device for schedule tensors.
    image_chw : tuple
        Expected image shape ``(C, H, W)`` — used for generation.
    """

    def __init__(self, network, n_steps=200, min_beta=10 ** -4, max_beta=0.02, device=None, image_chw=(1, 64, 128)):
        super(GaussianDDPM, self).__init__()
        self.n_steps = n_steps
        self.device = device
        self.image_chw = image_chw
        self.network = network.to(device)

        self.betas = torch.linspace(min_beta, max_beta, n_steps).to(device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.tensor(
            [torch.prod(self.alphas[:i + 1]) for i in range(len(self.alphas))]
        ).to(device)

    def forward(self, x0, t, epsilon=None, one_step=False):
        """Noise the clean image x0 to timestep t.

        x_t = √ᾱ_t · x0 + scale · ε

        where ``scale = √(1−ᾱ_t)`` if the noise strategy uses Gaussian
        scaling, otherwise ``scale = 1`` (precomputed noise magnitude).

        Parameters
        ----------
        x0 : (B, C, H, W)
        t : (B,) integer timesteps
        epsilon : (B, C, H, W), optional — generated if None
        one_step : bool — use alpha[t] instead of alpha_bar[t]

        Returns
        -------
        (B, C, H, W) noisy image x_t
        """
        n, c, h, w = x0.shape

        if one_step:
            a_bar = self.alphas[t]
        else:
            a_bar = self.alpha_bars[t]

        if epsilon is None:
            epsilon = torch.randn(n, c, h, w).to(self.device)

        # NOTE: tight coupling to DDInitializer singleton — see module docstring.
        dd = DDInitializer()
        if dd.get_noise_strategy().get_gaussian_scaling():
            noisy = a_bar.sqrt().reshape(n, 1, 1, 1) * x0 + (1 - a_bar).sqrt().reshape(n, 1, 1, 1) * epsilon
        else:
            noisy = a_bar.sqrt().reshape(n, 1, 1, 1) * x0 + epsilon

        return noisy

    def backward(self, x, t):
        """Run the denoiser network: predict noise from noisy input.

        Parameters
        ----------
        x : (B, C_in, H, W)
        t : (B,) or (B, 1) integer timesteps

        Returns
        -------
        (B, 2, H, W) predicted noise (or x0, depending on training target).
        """
        return self.network(x, t)
