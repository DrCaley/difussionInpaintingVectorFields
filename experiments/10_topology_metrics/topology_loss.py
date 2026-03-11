"""
Topology-aware auxiliary training losses for DDPM.

All losses here are cheap, fully differentiable, and operate on batched
tensors. They compute MSE on derived scalar fields (vorticity, divergence)
rather than raw velocity components, forcing the network to preserve
differential structure.

These are Tier 2 losses — differentiable proxies for the topological
metrics computed in persistence_metrics.py. Tier 3 (persistence Wasserstein
as a direct loss) is a future extension requiring topologylayer.

Usage:
    from experiments.10_topology_metrics.topology_loss import topology_aware_loss

    # In training loop, after obtaining x0 prediction:
    result = topology_aware_loss(pred_x0, true_x0, ocean_mask)
    result['total'].backward()
"""
import torch
import torch.nn.functional as F


# ── Derivative kernels ─────────────────────────────────────────────────

# Lazy-init globals: created once, moved to device on first call
_DX_KERNEL = None
_DY_KERNEL = None


def _get_kernels(device: torch.device):
    """Lazy-init central difference kernels on the correct device."""
    global _DX_KERNEL, _DY_KERNEL
    if _DX_KERNEL is None or _DX_KERNEL.device != device:
        # Central difference: [-1, 0, 1] / 2
        # Shape: (out_channels=1, in_channels=1, H=1, W=3)
        _DX_KERNEL = torch.tensor(
            [[[[-1.0, 0.0, 1.0]]]], device=device
        ) / 2.0
        # Shape: (1, 1, 3, 1)
        _DY_KERNEL = torch.tensor(
            [[[[-1.0], [0.0], [1.0]]]], device=device
        ) / 2.0
    return _DX_KERNEL, _DY_KERNEL


def _ddx(field: torch.Tensor) -> torch.Tensor:
    """Central difference in x (axis=3, columns). Input: (B, 1, H, W)."""
    dx, _ = _get_kernels(field.device)
    return F.conv2d(field, dx, padding=(0, 1))


def _ddy(field: torch.Tensor) -> torch.Tensor:
    """Central difference in y (axis=2, rows). Input: (B, 1, H, W)."""
    _, dy = _get_kernels(field.device)
    return F.conv2d(field, dy, padding=(1, 0))


# ── Derived field computations ─────────────────────────────────────────


def compute_vorticity(uv: torch.Tensor) -> torch.Tensor:
    """
    Compute vorticity ω = ∂v/∂x − ∂u/∂y.

    Args:
        uv: (B, 2, H, W) velocity field

    Returns:
        (B, 1, H, W) vorticity field
    """
    u = uv[:, 0:1]  # (B, 1, H, W)
    v = uv[:, 1:2]
    return _ddx(v) - _ddy(u)


def compute_divergence(uv: torch.Tensor) -> torch.Tensor:
    """
    Compute divergence δ = ∂u/∂x + ∂v/∂y.

    Args:
        uv: (B, 2, H, W) velocity field

    Returns:
        (B, 1, H, W) divergence field
    """
    u = uv[:, 0:1]
    v = uv[:, 1:2]
    return _ddx(u) + _ddy(v)


def compute_speed(uv: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute speed |v| = √(u² + v²).

    Args:
        uv: (B, 2, H, W) velocity field
        eps: small constant for numerical stability of sqrt gradient

    Returns:
        (B, 1, H, W) speed field
    """
    return torch.sqrt(uv[:, 0:1] ** 2 + uv[:, 1:2] ** 2 + eps)


# ── Loss helpers ───────────────────────────────────────────────────────


def _masked_mse(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """MSE over masked (ocean) pixels only."""
    diff_sq = (pred - target) ** 2
    return (diff_sq * mask).sum() / mask.sum().clamp(min=1)


def _ensure_mask_shape(ocean_mask: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Ensure mask is broadcastable to ref shape (B, C, H, W)."""
    if ocean_mask.dim() == 2:
        # (H, W) -> (1, 1, H, W)
        ocean_mask = ocean_mask.unsqueeze(0).unsqueeze(0)
    elif ocean_mask.dim() == 3:
        # (1, H, W) or (B, H, W) -> (B, 1, H, W)
        ocean_mask = ocean_mask.unsqueeze(1)
    # Now (B, 1, H, W) — broadcastable to (B, C, H, W)
    return ocean_mask


# ── Main loss function ─────────────────────────────────────────────────


def topology_aware_loss(
    pred_x0: torch.Tensor,
    true_x0: torch.Tensor,
    ocean_mask: torch.Tensor,
    lambda_vort: float = 0.1,
    lambda_div: float = 0.05,
    lambda_speed: float = 0.0,
) -> dict:
    """
    MSE + weighted auxiliary losses on derived scalar fields.

    All terms are differentiable, batched, and have negligible compute
    overhead compared to the UNet forward pass.

    Args:
        pred_x0: (B, 2, H, W) predicted clean velocity field
        true_x0: (B, 2, H, W) ground truth clean velocity field
        ocean_mask: (H, W) or (1, 1, H, W) or (B, 1, H, W), 1 = ocean
        lambda_vort: weight on vorticity MSE
        lambda_div: weight on divergence MSE
        lambda_speed: weight on speed MSE (0 = disabled)

    Returns:
        dict with:
            'total': scalar tensor (backprop through this)
            'mse': float, pixel-level MSE
            'vort_mse': float or None
            'div_mse': float or None
            'speed_mse': float or None
    """
    mask = _ensure_mask_shape(ocean_mask, pred_x0)

    # Expand to 2-channel for velocity MSE
    mask_2ch = mask.expand_as(pred_x0)
    # 1-channel for scalar field MSE
    mask_1ch = mask.expand(pred_x0.size(0), 1, pred_x0.size(2), pred_x0.size(3))

    # Pixel MSE on velocity
    mse = _masked_mse(pred_x0, true_x0, mask_2ch)
    total = mse

    result = {
        "mse": mse.item(),
        "vort_mse": None,
        "div_mse": None,
        "speed_mse": None,
    }

    # Vorticity MSE
    if lambda_vort > 0:
        omega_pred = compute_vorticity(pred_x0)
        omega_true = compute_vorticity(true_x0)
        vort_mse = _masked_mse(omega_pred, omega_true, mask_1ch)
        total = total + lambda_vort * vort_mse
        result["vort_mse"] = vort_mse.item()

    # Divergence MSE
    if lambda_div > 0:
        div_pred = compute_divergence(pred_x0)
        div_true = compute_divergence(true_x0)
        div_mse = _masked_mse(div_pred, div_true, mask_1ch)
        total = total + lambda_div * div_mse
        result["div_mse"] = div_mse.item()

    # Speed MSE
    if lambda_speed > 0:
        speed_pred = compute_speed(pred_x0)
        speed_true = compute_speed(true_x0)
        speed_mse = _masked_mse(speed_pred, speed_true, mask_1ch)
        total = total + lambda_speed * speed_mse
        result["speed_mse"] = speed_mse.item()

    result["total"] = total
    return result
