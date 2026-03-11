"""
Quick sanity check that both persistence_metrics and topology_loss
work on synthetic data. Run this first to verify gudhi is installed
and all shapes/dtypes are correct.

Usage:
    PYTHONPATH=. python experiments/10_topology_metrics/sanity_check.py
"""
import sys
import time

import numpy as np
import torch


def _make_rankine_vortex(H, W, cx, cy, r_max=8.0, strength=1.0):
    """Create a Rankine vortex velocity field at (cx, cy)."""
    yy, xx = np.mgrid[0:H, 0:W]
    dx = xx - cx
    dy = yy - cy
    r = np.sqrt(dx**2 + dy**2) + 1e-6
    v_theta = strength * np.where(r < r_max, r / r_max, r_max / r)
    u = -v_theta * dy / r
    v = v_theta * dx / r
    return np.stack([u, v], axis=0).astype(np.float32)


def _make_ocean_mask(H, W, border=3):
    """Rectangular ocean mask with land border."""
    mask = np.ones((H, W), dtype=bool)
    mask[:border, :] = False
    mask[-border:, :] = False
    mask[:, :border] = False
    mask[:, -border:] = False
    return mask


def check_persistence_metrics():
    """Test persistence computation on a synthetic vortex field."""
    print("1. Persistence metrics (gudhi)")
    print("-" * 40)

    try:
        import gudhi  # noqa: F401
    except ImportError:
        print("  FAIL: gudhi not installed. Run: pip install gudhi")
        return False

    try:
        from experiments.topology_metrics.persistence_metrics import (
            compute_derived_scalars,
            persistence_0d_both_signs,
            topology_distance,
        )
    except ImportError:
        # Try alternate import path
        from persistence_metrics import (
            compute_derived_scalars,
            persistence_0d_both_signs,
            topology_distance,
        )

    H, W = 44, 94
    ocean_mask = _make_ocean_mask(H, W)

    # Two synthetic vortex fields: GT and a shifted/weaker prediction
    gt = _make_rankine_vortex(H, W, cx=47, cy=22, strength=1.0)
    pred = _make_rankine_vortex(H, W, cx=50, cy=24, strength=0.7)

    # Test derived scalars
    scalars = compute_derived_scalars(gt, ocean_mask, sigma=1.0)
    print(f"  Derived scalars: {list(scalars.keys())}")
    for name, field in scalars.items():
        valid = field[ocean_mask]
        valid = valid[np.isfinite(valid)]
        print(f"    {name}: min={valid.min():.4f} max={valid.max():.4f}")

    # Test persistence
    diags = persistence_0d_both_signs(scalars["vorticity"], ocean_mask)
    n_sub = len(diags["sublevel"])
    n_sup = len(diags["superlevel"])
    print(f"  Vorticity persistence: {n_sub} sublevel, {n_sup} superlevel features")

    # Test full topology distance
    t0 = time.time()
    dist = topology_distance(gt, pred, ocean_mask, sigma=1.0)
    elapsed = time.time() - t0
    print(f"  Topology distance (gt vs shifted/weaker vortex) [{elapsed:.2f}s]:")
    print(f"    W2_vorticity:   {dist['W2_vorticity']:.4f}")
    print(f"    W2_divergence:  {dist['W2_divergence']:.4f}")
    print(f"    W2_speed:       {dist['W2_speed']:.4f}")
    print(f"    W2_okubo_weiss: {dist['W2_okubo_weiss']:.4f}")
    print(f"    W_total:        {dist['W_total_weighted']:.4f}")

    # Self-distance should be ~0 (allow floating point noise from Wasserstein solver)
    dist_self = topology_distance(gt, gt, ocean_mask, sigma=1.0)
    if dist_self["W_total_weighted"] > 1e-4:
        print(
            f"  FAIL: Self-distance should be ~0, "
            f"got {dist_self['W_total_weighted']:.2e}"
        )
        return False
    print(f"  Self-distance: {dist_self['W_total_weighted']:.2e} (should be ~0) ✓")

    # Distance to shifted vortex should be > 0
    if dist["W_total_weighted"] <= 0:
        print(f"  FAIL: Distance to shifted vortex should be > 0")
        return False
    print(f"  Distance to shifted vortex: {dist['W_total_weighted']:.4f} > 0 ✓")

    # Test with uniform flow (no eddies) vs vortex — should have large distance
    uniform = np.stack(
        [np.ones((H, W)) * 0.1, np.zeros((H, W))], axis=0
    ).astype(np.float32)
    dist_uniform = topology_distance(gt, uniform, ocean_mask, sigma=1.0)
    print(
        f"  Vortex vs uniform flow: W_total={dist_uniform['W_total_weighted']:.4f}"
    )
    if dist_uniform["W_total_weighted"] > dist["W_total_weighted"]:
        print("  Uniform flow is farther from vortex than shifted vortex ✓")
    else:
        print(
            "  WARNING: Uniform flow not farther from vortex than shifted vortex. "
            "May need sigma tuning."
        )

    print("  PASSED\n")
    return True


def check_topology_loss():
    """Test differentiable topology loss on synthetic data."""
    print("2. Topology-aware training loss (PyTorch)")
    print("-" * 40)

    try:
        from experiments.topology_metrics.topology_loss import (
            topology_aware_loss,
            compute_vorticity,
            compute_divergence,
            compute_speed,
        )
    except ImportError:
        from topology_loss import (
            topology_aware_loss,
            compute_vorticity,
            compute_divergence,
            compute_speed,
        )

    B, C, H, W = 4, 2, 44, 94
    pred = torch.randn(B, C, H, W, requires_grad=True)
    target = torch.randn(B, C, H, W)
    mask = torch.ones(1, 1, H, W)

    # Check shapes
    omega = compute_vorticity(pred)
    assert omega.shape == (B, 1, H, W), f"Vorticity shape wrong: {omega.shape}"
    print(f"  Vorticity shape: {omega.shape} ✓")

    div = compute_divergence(pred)
    assert div.shape == (B, 1, H, W), f"Divergence shape wrong: {div.shape}"
    print(f"  Divergence shape: {div.shape} ✓")

    speed = compute_speed(pred)
    assert speed.shape == (B, 1, H, W), f"Speed shape wrong: {speed.shape}"
    print(f"  Speed shape: {speed.shape} ✓")

    # Check loss computation
    result = topology_aware_loss(
        pred,
        target,
        mask,
        lambda_vort=0.1,
        lambda_div=0.05,
        lambda_speed=0.01,
    )
    print(f"  Loss components:")
    print(f"    mse:       {result['mse']:.4f}")
    print(f"    vort_mse:  {result['vort_mse']:.4f}")
    print(f"    div_mse:   {result['div_mse']:.4f}")
    print(f"    speed_mse: {result['speed_mse']:.4f}")
    print(f"    total:     {result['total'].item():.4f}")

    # Check backward pass
    result["total"].backward()
    assert pred.grad is not None, "No gradients!"
    assert pred.grad.abs().sum() > 0, "Zero gradients!"
    grad_norm = pred.grad.norm().item()
    print(f"  Gradient norm: {grad_norm:.4f} (nonzero = backprop works) ✓")

    # Check with 2D mask input
    mask_2d = torch.ones(H, W)
    result2 = topology_aware_loss(
        pred.detach().requires_grad_(True),
        target,
        mask_2d,
        lambda_vort=0.1,
    )
    assert isinstance(result2["total"], torch.Tensor)
    print(f"  2D mask input works ✓")

    # Check with partial mask (mimic land)
    mask_partial = torch.ones(1, 1, H, W)
    mask_partial[:, :, :3, :] = 0  # land border
    mask_partial[:, :, -3:, :] = 0
    result3 = topology_aware_loss(
        pred.detach().requires_grad_(True),
        target,
        mask_partial,
        lambda_vort=0.1,
        lambda_div=0.05,
    )
    assert result3["total"].item() > 0
    print(f"  Partial ocean mask works ✓")

    # Verify that adding vort loss changes the total
    result_no_vort = topology_aware_loss(
        pred.detach().requires_grad_(True),
        target,
        mask,
        lambda_vort=0.0,
        lambda_div=0.0,
        lambda_speed=0.0,
    )
    assert abs(result_no_vort["total"].item() - result_no_vort["mse"]) < 1e-6
    print(f"  lambda=0 gives pure MSE ✓")

    print("  PASSED\n")
    return True


def check_data_loading():
    """
    Check if any known reconstruction .pt files exist and
    can be loaded with our loaders.
    """
    print("3. Data loading (check for existing reconstruction files)")
    print("-" * 40)

    from pathlib import Path

    candidates = [
        (
            "experiments/08_network_architecture/repaint_gaussian_attn/"
            "results/bulk_eval_best_100samples.pt",
            "bulk-eval-list",
        ),
        (
            "experiments/07_ddpm_composite/bulk_eval/results/",
            "composite-tensors",
        ),
    ]

    found_any = False
    for path, fmt in candidates:
        p = Path(path)
        if p.exists():
            print(f"  Found: {path} (format: {fmt})")
            found_any = True

            # Try loading first sample
            if fmt == "bulk-eval-list":
                try:
                    from persistence_metrics import load_bulk_eval_list

                    for i, mask, fields in load_bulk_eval_list(str(p)):
                        methods = list(fields.keys())
                        gt_shape = fields["gt"].shape if "gt" in fields else "N/A"
                        print(
                            f"    Sample 0: methods={methods}, "
                            f"gt_shape={gt_shape}, "
                            f"ocean_pixels={mask.sum()}"
                        )
                        break
                    print(f"    Loading works ✓")
                except Exception as e:
                    print(f"    Loading failed: {e}")
        else:
            print(f"  Not found: {path}")

    if not found_any:
        print(
            "  No reconstruction files found. "
            "Run an eval script first to generate .pt files."
        )
        print("  (This is OK — persistence_metrics.py will work once data exists)")

    print()
    return True  # Not a hard failure


def main():
    print("=" * 60)
    print("Experiment 10: Topology Metrics — Sanity Check")
    print("=" * 60)
    print()

    results = {}

    results["persistence"] = check_persistence_metrics()
    results["loss"] = check_topology_loss()
    results["data"] = check_data_loading()

    print("=" * 60)
    for name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {name:20s}: {status}")
    print("=" * 60)

    all_pass = all(results.values())
    if all_pass:
        print("\nAll checks passed. Ready to run experiments.")
        print("\nNext steps:")
        print(
            "  1. Run on existing data:\n"
            "     PYTHONPATH=. python experiments/10_topology_metrics/"
            "persistence_metrics.py \\\n"
            "       --results-pt experiments/08_network_architecture/"
            "repaint_gaussian_attn/results/bulk_eval_best_100samples.pt \\\n"
            "       --output experiments/10_topology_metrics/"
            "eval_persistence/results/persistence_100.pt \\\n"
            "       --format bulk-eval-list"
        )
    else:
        print("\nSome checks failed. See above for details.")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
