"""Loss strategies for DDPM training.

Each strategy computes a scalar loss from (predicted, target, noisy_img).
The ``noisy_img`` parameter is optional for pure reconstruction losses
but required for physics-based losses that reconstruct x₀.

See ``ddpm.protocols.LossStrategyProtocol`` for the formal contract.

Registry
--------
Concrete strategies are registered in ``LOSS_REGISTRY`` and resolved
by name via ``get_loss_strategy()``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ddpm.helper_functions.compute_divergence import compute_divergence


class LossStrategy(nn.Module):
    """Base class for all loss strategies.

    Extends ``nn.Module`` so loss parameters (if any) are visible to
    the optimizer.  Subclasses must override ``forward()``.

    See Also
    --------
    ddpm.protocols.LossStrategyProtocol
    """
    def forward(self, predicted_noise: torch.Tensor, target_noise: torch.Tensor, noisy_img: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError("Loss strategy must implement forward()")


class MSELossStrategy(LossStrategy):
    """Simple mean-squared-error loss: ||ε̂ − ε||².

    Ignores ``noisy_img`` — operates directly on the predicted
    and target noise (or x₀, depending on prediction target).
    """
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss()

    def forward(self, predicted_noise, target_noise, noisy_img=None, **kwargs) -> torch.Tensor:
        return self.loss_fn(predicted_noise, target_noise)


class PhysicalLossStrategy(LossStrategy):
    """MSE + divergence penalty on the predicted noise.

    ``loss = w1 * MSE(ε̂, ε) + w2 * mean(div(ε̂)²)``

    The divergence penalty encourages the model to produce
    noise with low divergence, biasing outputs toward physically
    plausible (incompressible) velocity fields.

    Parameters
    ----------
    w1, w2 : float
        Weights for MSE and divergence terms.
    """
    def __init__(self, w1=1.0, w2=1.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.w1 = w1
        self.w2 = w2

    def forward(self, predicted_noise, target_noise, noisy_img=None, **kwargs) -> torch.Tensor:
        mse_loss = self.mse(predicted_noise, target_noise)
        div_loss = self.physical_loss(predicted_noise)
        return self.w1 * mse_loss + self.w2 * div_loss

    @staticmethod
    def physical_loss(predicted: torch.Tensor) -> torch.Tensor:
        batch_divs = []
        for field in predicted:
            u, v = field[0], field[1]
            div = compute_divergence(u, v)
            batch_divs.append(div.pow(2).mean())
        return torch.stack(batch_divs).mean()

class HotGarbage(LossStrategy):
    """MSE + divergence comparison on unstandardized fields.

    Reconstructs x₀ = x_t − ε̂, unstandardizes both predicted and
    target, then penalizes the divergence difference in physical
    units.  Couples directly to the ``DDInitializer`` singleton
    for the standardizer — this is a known coupling concern.

    ``loss = w1 * MSE(ε̂, ε) + w2 * 100 * MSE(div(f̂), div(f))``

    where f̂ and f are the unstandardized predicted and target fields.

    Parameters
    ----------
    w1, w2 : float
        Weights for MSE and physics terms.
    """
    def __init__(self, w1=1.0, w2=1.0):
        super().__init__()
        self.w1 = w1
        self.w2 = w2

        from data_prep.data_initializer import DDInitializer
        dd = DDInitializer()

        self.standardizer = dd.get_standardizer()
        self.mse = nn.MSELoss()

    def forward(self, predicted_noise, target_noise, noisy_img, **kwargs):
        # 1. MSE Loss between noise predictions
        mse_loss = self.mse(predicted_noise, target_noise)

        # 2. Unstandardize predicted and target velocity fields
        prediction = noisy_img - predicted_noise
        real = noisy_img - target_noise

        unstandardized_prediction = self.standardizer.unstandardize(prediction)
        unstandardized_real = self.standardizer.unstandardize(real)

        # 3. Physical loss: divergence between prediction and real
        div_loss = self.physical_loss(unstandardized_prediction, unstandardized_real) * 100

        # 4. Weighted combination
        return self.w1 * mse_loss + self.w2 * div_loss

    @staticmethod
    def physical_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes the mean squared error between the divergence of predicted and real fields.
        """
        batch_divs = []
        for pred_field, target_field in zip(predicted, target):
            u_pred, v_pred = pred_field[0], pred_field[1]
            u_real, v_real = target_field[0], target_field[1]

            div_pred = compute_divergence(u_pred, v_pred)
            div_real = compute_divergence(u_real, v_real)

            div_mse = (div_pred - div_real).pow(2).mean()
            batch_divs.append(div_mse)

        return torch.stack(batch_divs).mean()


class TopologyAwareLossStrategy(LossStrategy):
    """MSE + topology-aware auxiliary losses on derived scalar fields.

    Reconstructs x̂₀ from predicted noise (eps prediction) using the DDPM
    alpha_bar schedule, then computes vorticity/divergence/speed MSE as
    auxiliary losses that penalise incorrect differential structure.

    Only applies topology terms at low diffusion timesteps (t < t_max_frac × T)
    where the x̂₀ reconstruction is meaningful.  At high timesteps the
    reconstruction amplifies noise, so we gate it off.

    Stability features
    ------------------
    * **Warmup epoch**: topo terms are disabled for the first ``warmup_epochs``
      training epochs so the MSE can establish a reasonable baseline before
      derivative-based losses start backpropagating.
    * **pred_x₀ clamping**: the reconstructed x̂₀ is clamped to ``[-x0_clamp,
      x0_clamp]`` to prevent extreme values from the 1/√ᾱ division at higher t.

    Parameters
    ----------
    lambda_vort : float
        Weight on vorticity MSE auxiliary loss.
    lambda_div : float
        Weight on divergence MSE auxiliary loss.
    lambda_speed : float
        Weight on speed MSE auxiliary loss (0 = disabled).
    t_max_frac : float
        Fraction of total timesteps below which topo terms are active.
    warmup_epochs : int
        Number of initial epochs where topo terms are disabled (MSE only).
    x0_clamp : float
        Symmetric clamp on reconstructed x̂₀ to prevent gradient explosions.
    """

    def __init__(
        self,
        lambda_vort: float = 0.01,
        lambda_div: float = 0.005,
        lambda_speed: float = 0.0,
        t_max_frac: float = 0.1,
        warmup_epochs: int = 10,
        x0_clamp: float = 6.0,
    ):
        super().__init__()
        self.lambda_vort = lambda_vort
        self.lambda_div = lambda_div
        self.lambda_speed = lambda_speed
        self.t_max_frac = t_max_frac
        self.warmup_epochs = warmup_epochs
        self.x0_clamp = x0_clamp
        self.mse_fn = nn.MSELoss()

        # Central-difference kernels for spatial derivatives
        dx = torch.tensor([[[[-1.0, 0.0, 1.0]]]]) / 2.0   # (1,1,1,3)
        dy = torch.tensor([[[[-1.0], [0.0], [1.0]]]]) / 2.0  # (1,1,3,1)
        self.register_buffer("dx_kernel", dx)
        self.register_buffer("dy_kernel", dy)

        # Ocean mask: 44×94 ocean pixels in the 64×128 padded grid
        ocean = torch.zeros(1, 1, 64, 128)
        ocean[:, :, :44, :94] = 1.0
        self.register_buffer("ocean_mask", ocean)

    # ── helper spatial derivatives ──────────────────────────────
    def _ddx(self, f: torch.Tensor) -> torch.Tensor:
        return F.conv2d(f, self.dx_kernel, padding=(0, 1))

    def _ddy(self, f: torch.Tensor) -> torch.Tensor:
        return F.conv2d(f, self.dy_kernel, padding=(1, 0))

    def _vorticity(self, uv: torch.Tensor) -> torch.Tensor:
        """ω = ∂v/∂x − ∂u/∂y.  Input (B,2,H,W) → (B,1,H,W)."""
        return self._ddx(uv[:, 1:2]) - self._ddy(uv[:, 0:1])

    def _divergence(self, uv: torch.Tensor) -> torch.Tensor:
        """δ = ∂u/∂x + ∂v/∂y.  Input (B,2,H,W) → (B,1,H,W)."""
        return self._ddx(uv[:, 0:1]) + self._ddy(uv[:, 1:2])

    def _speed(self, uv: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """|v| = √(u²+v²).  Input (B,2,H,W) → (B,1,H,W)."""
        return torch.sqrt(uv[:, 0:1] ** 2 + uv[:, 1:2] ** 2 + eps)

    @staticmethod
    def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        diff_sq = (pred - target) ** 2
        return (diff_sq * mask).sum() / mask.sum().clamp(min=1)

    # ── forward ─────────────────────────────────────────────────
    def forward(self, predicted_noise, target_noise, noisy_img, **kwargs):
        """
        Args (positional — same as all LossStrategies):
            predicted_noise : model output (ε̂ or x̂₀)
            target_noise    : target (ε or x₀)
            noisy_img       : x_t

        Kwargs (passed by training loop):
            x0               : (B,2,H,W) clean images
            t                : (B,) integer timesteps
            ddpm             : GaussianDDPM instance (carries alpha_bars)
            prediction_target: 'eps' | 'x0'
            epoch            : int, current epoch (0-indexed)
        """
        # ── 1. Standard MSE on noise / x₀ (ocean only) ────────
        mask_2ch = self.ocean_mask.expand(predicted_noise.shape[0], 2, 64, 128)
        mse_loss = self._masked_mse(predicted_noise, target_noise, mask_2ch)

        # ── 1b. Warmup gate: skip topo terms early on ──────────
        epoch = kwargs.get("epoch", 0)
        if epoch < self.warmup_epochs:
            return mse_loss

        x0 = kwargs.get("x0")
        t = kwargs.get("t")
        ddpm = kwargs.get("ddpm")
        pred_target = kwargs.get("prediction_target", "eps")

        if x0 is None or t is None or ddpm is None:
            return mse_loss

        # ── 2. Reconstruct x̂₀ ──────────────────────────────────
        if pred_target == "x0":
            pred_x0 = predicted_noise          # model directly predicts x₀
        else:
            n = predicted_noise.shape[0]
            a_bar = ddpm.alpha_bars[t].reshape(n, 1, 1, 1)
            sqrt_a = a_bar.sqrt().clamp(min=1e-5)
            sqrt_1ma = (1 - a_bar).sqrt()
            pred_x0 = (noisy_img - sqrt_1ma * predicted_noise) / sqrt_a

        # Clamp to prevent extreme values from amplifying derivatives
        pred_x0 = pred_x0.clamp(-self.x0_clamp, self.x0_clamp)

        # ── 3. Gate by timestep ─────────────────────────────────
        t_threshold = int(self.t_max_frac * ddpm.n_steps)
        topo_mask = (t.squeeze() < t_threshold)

        if not topo_mask.any():
            return mse_loss

        pred_sub = pred_x0[topo_mask]
        x0_sub = x0[topo_mask]
        B_sub = pred_sub.shape[0]

        mask_1ch = self.ocean_mask.expand(B_sub, 1, 64, 128)

        # ── 4. Auxiliary topology losses ────────────────────────
        topo = 0.0

        if self.lambda_vort > 0:
            vp = self._vorticity(pred_sub)
            vt = self._vorticity(x0_sub)
            topo = topo + self.lambda_vort * self._masked_mse(vp, vt, mask_1ch)

        if self.lambda_div > 0:
            dp = self._divergence(pred_sub)
            dt = self._divergence(x0_sub)
            topo = topo + self.lambda_div * self._masked_mse(dp, dt, mask_1ch)

        if self.lambda_speed > 0:
            sp = self._speed(pred_sub)
            st = self._speed(x0_sub)
            topo = topo + self.lambda_speed * self._masked_mse(sp, st, mask_1ch)

        # Weight by fraction of batch that qualified (keep effective weight
        # independent of t_max_frac choice)
        frac = topo_mask.float().mean()
        return mse_loss + frac * topo


# ── Helmholtz supervised decomposition loss ─────────────────────────

class HelmholtzSupervisionLoss(TopologyAwareLossStrategy):
    """Topology-aware loss + per-head Helmholtz supervision + orthogonality.

    Adds two auxiliary terms to the base topology-aware loss:

    1. **Decomposition supervision**: MSE of each head's output against the
       FFT Helmholtz decomposition of the ground-truth x₀:
       ``L_decomp = MSE(v_sol, GT_sol) + MSE(v_irr, GT_irr)``

    2. **Orthogonality penalty**: penalises the cosine similarity between
       v_sol and v_irr to discourage the degenerate cancellation solution:
       ``L_orth = |<v_sol, v_irr>| / (||v_sol|| * ||v_irr||)``

    Both terms are gated by the same warmup and timestep thresholds as the
    topology terms, and are computed over the 44×94 ocean region.

    Parameters
    ----------
    lambda_decomp : float
        Weight on the per-head decomposition supervision MSE.
    lambda_orth : float
        Weight on the orthogonality (cosine similarity) penalty.
    decomp_t_max_frac : float
        Fraction of total timesteps below which decomp/orth terms are active.
        Defaults to 1.0 (always active) since the FFT decomposition of GT x₀
        is always clean regardless of noise level — unlike derived topology
        fields which degrade at high t.
    """

    def __init__(
        self,
        lambda_vort: float = 0.01,
        lambda_div: float = 0.005,
        lambda_speed: float = 0.0,
        t_max_frac: float = 0.1,
        warmup_epochs: int = 10,
        x0_clamp: float = 6.0,
        lambda_decomp: float = 0.1,
        lambda_orth: float = 0.01,
        decomp_t_max_frac: float = 1.0,
    ):
        super().__init__(
            lambda_vort=lambda_vort,
            lambda_div=lambda_div,
            lambda_speed=lambda_speed,
            t_max_frac=t_max_frac,
            warmup_epochs=warmup_epochs,
            x0_clamp=x0_clamp,
        )
        self.lambda_decomp = lambda_decomp
        self.lambda_orth = lambda_orth
        self.decomp_t_max_frac = decomp_t_max_frac

    def forward(self, predicted_noise, target_noise, noisy_img, **kwargs):
        # Base topology-aware loss (MSE + vort/div/speed)
        base_loss = super().forward(predicted_noise, target_noise, noisy_img, **kwargs)

        epoch = kwargs.get("epoch", 0)
        ddpm = kwargs.get("ddpm")

        # When heads are detached the base MSE has no grad — decomp loss
        # must always be active to provide gradient signal.
        detached = ddpm is not None and hasattr(ddpm.network, "detach_heads") and ddpm.network.detach_heads
        if epoch < self.warmup_epochs and not detached:
            self._last_base_loss = base_loss.item()
            self._last_decomp_loss = 0.0
            self._last_orth_loss = 0.0
            return base_loss

        x0 = kwargs.get("x0")
        t = kwargs.get("t")
        pred_target = kwargs.get("prediction_target", "x0")

        if ddpm is None or x0 is None or t is None:
            self._last_base_loss = base_loss.item()
            self._last_decomp_loss = 0.0
            self._last_orth_loss = 0.0
            return base_loss

        net = ddpm.network
        if not hasattr(net, "last_v_sol") or net.last_v_sol is None:
            self._last_base_loss = base_loss.item()
            self._last_decomp_loss = 0.0
            self._last_orth_loss = 0.0
            return base_loss

        v_sol = net.last_v_sol
        v_irr = net.last_v_irr

        # Gate by decomp-specific timestep threshold (default 1.0 = always on)
        decomp_threshold = int(self.decomp_t_max_frac * ddpm.n_steps)
        decomp_mask = (t.squeeze() < decomp_threshold)
        if not decomp_mask.any():
            self._last_base_loss = base_loss.item()
            self._last_decomp_loss = 0.0
            self._last_orth_loss = 0.0
            return base_loss

        v_sol_sub = v_sol[decomp_mask]
        v_irr_sub = v_irr[decomp_mask]
        B_sub = v_sol_sub.shape[0]
        frac = decomp_mask.float().mean()

        mask_2ch = self.ocean_mask.expand(B_sub, 2, 64, 128)

        # ── Decomposition supervision ──
        # Heads decompose whatever the model predicts (x0 or noise).
        # Supervise against Helmholtz decomposition of the matching target.
        decomp_source = target_noise if pred_target == "eps" else x0
        decomp_sub = decomp_source[decomp_mask]
        from ddpm.utils.helmholtz_split import helmholtz_decompose
        with torch.no_grad():
            gt_sol, gt_irr = helmholtz_decompose(decomp_sub)

        decomp_loss = (
            self._masked_mse(v_sol_sub * mask_2ch, gt_sol * mask_2ch, mask_2ch)
            + self._masked_mse(v_irr_sub * mask_2ch, gt_irr * mask_2ch, mask_2ch)
        )

        # Store raw component values for per-component logging
        self._last_base_loss = base_loss.item()
        self._last_decomp_loss = decomp_loss.item()
        self._last_orth_loss = 0.0

        return base_loss + frac * self.lambda_decomp * decomp_loss

    def helmholtz_aux(self, x0, t, ddpm, epoch=0, noise=None, prediction_target="x0"):
        """Return only the Helmholtz auxiliary loss terms (decomp + orth).

        Used by the training pipeline when mask_xt=True: the masked MSE is
        computed separately, and this method provides only the Helmholtz
        supervision terms to add on top.
        """
        detached = hasattr(ddpm.network, "detach_heads") and ddpm.network.detach_heads
        if epoch < self.warmup_epochs and not detached:
            return torch.tensor(0.0, device=x0.device)

        net = ddpm.network
        if not hasattr(net, "last_v_sol") or net.last_v_sol is None:
            return torch.tensor(0.0, device=x0.device)

        v_sol = net.last_v_sol
        v_irr = net.last_v_irr

        decomp_threshold = int(self.decomp_t_max_frac * ddpm.n_steps)
        decomp_mask = (t.squeeze() < decomp_threshold)
        if not decomp_mask.any():
            return torch.tensor(0.0, device=x0.device)

        v_sol_sub = v_sol[decomp_mask]
        v_irr_sub = v_irr[decomp_mask]
        B_sub = v_sol_sub.shape[0]
        frac = decomp_mask.float().mean()

        mask_2ch = self.ocean_mask.expand(B_sub, 2, 64, 128)

        # Heads decompose whatever the model predicts.
        # Supervise against Helmholtz decomposition of the matching target.
        decomp_source = noise if prediction_target == "eps" and noise is not None else x0
        decomp_sub = decomp_source[decomp_mask]
        from ddpm.utils.helmholtz_split import helmholtz_decompose
        with torch.no_grad():
            gt_sol, gt_irr = helmholtz_decompose(decomp_sub)

        decomp_loss = (
            self._masked_mse(v_sol_sub * mask_2ch, gt_sol * mask_2ch, mask_2ch)
            + self._masked_mse(v_irr_sub * mask_2ch, gt_irr * mask_2ch, mask_2ch)
        )

        # Store raw component values for per-component logging
        self._last_decomp_loss = decomp_loss.item()
        self._last_orth_loss = 0.0

        return frac * self.lambda_decomp * decomp_loss


LOSS_REGISTRY = {
    "mse": MSELossStrategy,
    "physical": PhysicalLossStrategy,
    "best_loss": HotGarbage,
    "topology_aware": TopologyAwareLossStrategy,
    "helmholtz_supervised": HelmholtzSupervisionLoss,
}

def get_loss_strategy(name: str, **kwargs) -> LossStrategy:
    return LOSS_REGISTRY[name](**kwargs)

