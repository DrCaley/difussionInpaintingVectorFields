"""
Multi-scalar persistent homology metrics for ocean velocity field evaluation.

Computes 0-dimensional persistence on four derived scalar fields
(vorticity, divergence, speed, Okubo-Weiss) and returns Wasserstein-2
distances between ground truth and predicted diagrams.

This module is evaluation-only — no training dependencies.

Usage as library:
    from experiments.10_topology_metrics.persistence_metrics import topology_distance
    result = topology_distance(gt_uv, pred_uv, ocean_mask)

Usage as CLI (on bulk_eval_best_100samples.pt):
    PYTHONPATH=. python experiments/10_topology_metrics/persistence_metrics.py \\
        --results-pt experiments/08_network_architecture/repaint_gaussian_attn/results/bulk_eval_best_100samples.pt \\
        --output experiments/10_topology_metrics/eval_persistence/results/persistence_100samples.pt \\
        --format bulk-eval-list

Dependencies:
    pip install gudhi
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

try:
    from gudhi.cubical_complex import CubicalComplex
    from gudhi.wasserstein import wasserstein_distance

    GUDHI_AVAILABLE = True
except ImportError:
    GUDHI_AVAILABLE = False

# Ocean region within the 64x128 padded grid
OCEAN_H, OCEAN_W = 44, 94

SCALAR_NAMES = ["vorticity", "divergence", "speed", "okubo_weiss"]

DEFAULT_WEIGHTS = {
    "vorticity": 1.0,
    "divergence": 0.5,
    "speed": 0.5,
    "okubo_weiss": 0.3,
}


# ── Derived scalar fields ──────────────────────────────────────────────


def compute_derived_scalars(
    uv_field: np.ndarray,
    ocean_mask: np.ndarray,
    sigma: float = 1.0,
) -> dict:
    """
    Compute derived scalar fields from a 2-component velocity field.

    Args:
        uv_field: (2, H, W) array, channels (u, v)
        ocean_mask: (H, W) boolean, True = ocean
        sigma: Gaussian smoothing before differentiation

    Returns:
        dict of scalar_name -> (H, W) numpy array (NaN on land)
    """
    u = uv_field[0].copy()
    v = uv_field[1].copy()

    # Zero out land before smoothing to avoid land contamination
    u[~ocean_mask] = 0.0
    v[~ocean_mask] = 0.0
    u = gaussian_filter(u, sigma=sigma)
    v = gaussian_filter(v, sigma=sigma)

    dudx = np.gradient(u, axis=1)
    dudy = np.gradient(u, axis=0)
    dvdx = np.gradient(v, axis=1)
    dvdy = np.gradient(v, axis=0)

    vorticity = dvdx - dudy
    divergence = dudx + dvdy
    speed = np.sqrt(u**2 + v**2)

    sn = dudx - dvdy  # normal strain
    ss = dvdx + dudy  # shear strain
    okubo_weiss = sn**2 + ss**2 - vorticity**2

    scalars = {
        "vorticity": vorticity,
        "divergence": divergence,
        "speed": speed,
        "okubo_weiss": okubo_weiss,
    }

    # NaN on land so persistence ignores those pixels
    for name in scalars:
        scalars[name][~ocean_mask] = np.nan

    return scalars


# ── Persistence computation ────────────────────────────────────────────


def _persistence_0d(scalar_field: np.ndarray, ocean_mask: np.ndarray) -> np.ndarray:
    """
    0-dimensional sublevel-set persistence on ocean pixels only.

    Land pixels are set to +inf so they never create features
    in the filtration.

    Returns:
        (n, 2) array of finite (birth, death) pairs
    """
    if not GUDHI_AVAILABLE:
        raise ImportError("gudhi is required: pip install gudhi")

    field = scalar_field.copy()
    field[~ocean_mask] = np.inf
    field = np.nan_to_num(field, nan=np.inf, posinf=np.inf, neginf=-np.inf)

    cc = CubicalComplex(
        top_dimensional_cells=field.flatten(),
        dimensions=field.shape,
    )
    cc.persistence()
    pairs = np.array(cc.persistence_intervals_in_dimension(0))

    if len(pairs) == 0:
        return np.empty((0, 2))

    finite = np.all(np.isfinite(pairs), axis=1)
    return pairs[finite]


def persistence_0d_both_signs(
    scalar_field: np.ndarray,
    ocean_mask: np.ndarray,
) -> dict:
    """
    Sublevel persistence on +field (catches minima first) and -field
    (catches maxima first). For vorticity this separates cyclonic
    and anticyclonic eddies.

    Returns:
        {'sublevel': (n,2), 'superlevel': (m,2)}
    """
    return {
        "sublevel": _persistence_0d(scalar_field, ocean_mask),
        "superlevel": _persistence_0d(-scalar_field, ocean_mask),
    }


# ── Wasserstein distance ──────────────────────────────────────────────


def _wasserstein(diag1: np.ndarray, diag2: np.ndarray, p: int = 2) -> float:
    """Wasserstein-p distance between two persistence diagrams."""
    if not GUDHI_AVAILABLE:
        raise ImportError("gudhi is required: pip install gudhi")

    if len(diag1) == 0 and len(diag2) == 0:
        return 0.0
    if len(diag1) == 0:
        diag1 = np.empty((0, 2))
    if len(diag2) == 0:
        diag2 = np.empty((0, 2))

    return wasserstein_distance(diag1, diag2, order=p)


# ── Per-sample metric ─────────────────────────────────────────────────


def topology_distance(
    gt_uv: np.ndarray,
    pred_uv: np.ndarray,
    ocean_mask: np.ndarray,
    sigma: float = 1.0,
    weights: dict = None,
    p: int = 2,
) -> dict:
    """
    Multi-scalar topological distance between two velocity fields.

    Computes persistence diagrams on four derived scalars (vorticity,
    divergence, speed, Okubo-Weiss) in both sublevel and superlevel
    directions, and returns Wasserstein-p distances for each.

    Args:
        gt_uv: (2, H, W) ground truth velocity (numpy)
        pred_uv: (2, H, W) predicted velocity (numpy)
        ocean_mask: (H, W) boolean, True = ocean
        sigma: Gaussian smoothing before differentiation
        weights: per-scalar weights for combined metric
        p: Wasserstein order

    Returns:
        dict with per-scalar distances, feature counts, and weighted total
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    scalars_gt = compute_derived_scalars(gt_uv, ocean_mask, sigma)
    scalars_pred = compute_derived_scalars(pred_uv, ocean_mask, sigma)

    results = {}
    weighted_total = 0.0

    for name in SCALAR_NAMES:
        diags_gt = persistence_0d_both_signs(scalars_gt[name], ocean_mask)
        diags_pred = persistence_0d_both_signs(scalars_pred[name], ocean_mask)

        for direction in ["sublevel", "superlevel"]:
            d = _wasserstein(diags_gt[direction], diags_pred[direction], p=p)
            key = f"W{p}_{name}_{direction}"
            results[key] = d
            results[f"n_gt_{name}_{direction}"] = len(diags_gt[direction])
            results[f"n_pred_{name}_{direction}"] = len(diags_pred[direction])

        # Combined distance for this scalar (sum of both directions)
        combined = (
            results[f"W{p}_{name}_sublevel"] + results[f"W{p}_{name}_superlevel"]
        )
        results[f"W{p}_{name}"] = combined
        weighted_total += weights.get(name, 1.0) * combined

    results["W_total_weighted"] = weighted_total
    return results


# ── Data loading helpers ───────────────────────────────────────────────


def _to_numpy(t):
    """Convert tensor or ndarray to float32 numpy."""
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().float().numpy()
    return np.asarray(t, dtype=np.float32)


def _extract_ocean(field: np.ndarray) -> np.ndarray:
    """
    If field is (C, 64, 128) or (1, C, 64, 128), crop to (C, 44, 94).
    If already (C, 44, 94) or (2, 44, 94), return as-is.
    """
    if field.ndim == 4:
        field = field.squeeze(0)  # (C, H, W)
    if field.shape[-2:] == (64, 128):
        return field[:, :OCEAN_H, :OCEAN_W]
    return field


def _build_ocean_mask(sample_field: np.ndarray) -> np.ndarray:
    """
    Derive ocean mask from a velocity field: ocean where |v| > 0.
    sample_field: (2, H, W) numpy
    """
    speed = np.sqrt(sample_field[0] ** 2 + sample_field[1] ** 2)
    return speed > 1e-10


def load_bulk_eval_list(path: str):
    """
    Load experiments/08_*/bulk_eval_best_100samples.pt format.

    Structure: {'samples': [dict, ...], ...}
    Each dict has 'ground_truth', 'gp_output', 'ddpm_output',
    'land_mask' (all (1, 2, 64, 128) Tensors).

    Yields:
        (sample_idx, ocean_mask, {'gt': (2,H,W), 'gp': ..., 'ddpm': ...})
    """
    data = torch.load(path, map_location="cpu", weights_only=False)
    samples = data["samples"]

    for i, s in enumerate(samples):
        gt = _to_numpy(_extract_ocean(s["ground_truth"]))
        ocean_mask = _build_ocean_mask(gt)

        fields = {"gt": gt}

        if "gp_output" in s:
            fields["gp"] = _to_numpy(_extract_ocean(s["gp_output"]))
        if "ddpm_output" in s:
            fields["ddpm"] = _to_numpy(_extract_ocean(s["ddpm_output"]))

        yield i, ocean_mask, fields


def load_composite_tensors(directory: str):
    """
    Load experiments/07_*/bulk_eval/results/{pct}/sample_*/tensors.pt format.

    Each tensors.pt has 'gt', 'gp_mean', 'gpdiff', 'ocean_mask' as ndarrays
    of shape (2, 44, 94) or (44, 94).

    Yields:
        (sample_idx, ocean_mask, {'gt': ..., 'gp': ..., 'gpdiff': ...})
    """
    tensor_files = sorted(Path(directory).glob("*/tensors.pt"))
    for i, tf in enumerate(tensor_files):
        d = torch.load(str(tf), map_location="cpu", weights_only=False)

        gt = _to_numpy(d["gt"])
        ocean_mask = _to_numpy(d["ocean_mask"]).astype(bool)

        fields = {"gt": gt}
        if "gp_mean" in d:
            fields["gp"] = _to_numpy(d["gp_mean"])
        if "gpdiff" in d:
            fields["gpdiff"] = _to_numpy(d["gpdiff"])
        if "composite" in d:
            fields["composite"] = _to_numpy(d["composite"])

        yield i, ocean_mask, fields


def load_repaint_runs(directory: str):
    """
    Load experiments/02_*/repaint_gaussian/results/run_data/run_*.pt format.

    Each run_NNN.pt has 'ground_truth', 'repaint_output', 'gp_output',
    'missing_mask' as Tensors (2, 64, 128).

    Yields:
        (sample_idx, ocean_mask, {'gt': ..., 'gp': ..., 'repaint': ...})
    """
    run_files = sorted(Path(directory).glob("run_*.pt"))
    for i, rf in enumerate(run_files):
        d = torch.load(str(rf), map_location="cpu", weights_only=False)

        gt = _to_numpy(_extract_ocean(d["ground_truth"]))
        ocean_mask = _build_ocean_mask(gt)

        fields = {"gt": gt}
        if "gp_output" in d:
            fields["gp"] = _to_numpy(_extract_ocean(d["gp_output"]))
        if "repaint_output" in d:
            fields["repaint"] = _to_numpy(_extract_ocean(d["repaint_output"]))

        yield i, ocean_mask, fields


def load_vcnn_eval(path: str):
    """
    Load results/voronoi_cnn_eval/voronoi_cnn_eval_results.pt format.

    Structure: {'results': [dict, ...], ...}
    Each dict has 'voronoi_pred' (2, 44, 94), 'ground_truth' (2, 44, 94).

    Yields:
        (sample_idx, ocean_mask, {'gt': (2,44,94), 'vcnn': (2,44,94)})
    """
    data = torch.load(path, map_location="cpu", weights_only=False)
    results = data["results"]

    for i, r in enumerate(results):
        gt = _to_numpy(r["ground_truth"])  # already (2, 44, 94)
        ocean_mask = _build_ocean_mask(gt)

        fields = {"gt": gt}
        fields["vcnn"] = _to_numpy(r["voronoi_pred"])

        yield i, ocean_mask, fields


# ── Batch evaluation ───────────────────────────────────────────────────


def evaluate_samples(sample_iterator, sigma: float = 1.0, p: int = 2) -> dict:
    """
    Run topology_distance on all samples from an iterator.

    Args:
        sample_iterator: yields (idx, ocean_mask, {'gt': ..., 'method1': ...})
        sigma: Gaussian smoothing parameter
        p: Wasserstein order

    Returns:
        dict of method_name -> list of per-sample metric dicts
    """
    all_metrics = {}  # method -> [metric_dict, ...]

    for i, ocean_mask, fields in sample_iterator:
        gt = fields.pop("gt")

        for method, pred in fields.items():
            if method not in all_metrics:
                all_metrics[method] = []

            metrics = topology_distance(gt, pred, ocean_mask, sigma=sigma, p=p)
            all_metrics[method].append(metrics)

        if (i + 1) % 10 == 0:
            methods_str = ", ".join(
                f"{m}: {len(v)}" for m, v in all_metrics.items()
            )
            print(f"  Processed {i + 1} samples ({methods_str})")

    return all_metrics


def metrics_to_tensors(all_metrics: dict) -> dict:
    """Convert per-sample metric dicts to flat tensor dict for torch.save."""
    output = {}
    for method, sample_metrics in all_metrics.items():
        if not sample_metrics:
            continue
        scalar_keys = [
            k
            for k in sample_metrics[0]
            if isinstance(sample_metrics[0][k], (float, int, np.floating, np.integer))
        ]
        for k in scalar_keys:
            vals = [m[k] for m in sample_metrics]
            output[f"{method}/{k}"] = torch.tensor(vals, dtype=torch.float32)
    return output


def print_summary(all_metrics: dict):
    """Print per-method summary statistics."""
    for method, metrics in all_metrics.items():
        if not metrics:
            continue
        w_total = [m["W_total_weighted"] for m in metrics]
        print(f"\n{'='*50}")
        print(f"Method: {method}  (n={len(metrics)})")
        print(f"{'='*50}")
        print(
            f"  W_total_weighted: mean={np.mean(w_total):.4f}  "
            f"median={np.median(w_total):.4f}  "
            f"std={np.std(w_total):.4f}"
        )
        for name in SCALAR_NAMES:
            key = f"W2_{name}"
            vals = [m[key] for m in metrics]
            n_gt = [
                m[f"n_gt_{name}_sublevel"] + m[f"n_gt_{name}_superlevel"]
                for m in metrics
            ]
            n_pred = [
                m[f"n_pred_{name}_sublevel"] + m[f"n_pred_{name}_superlevel"]
                for m in metrics
            ]
            print(
                f"  W2_{name:12s}: mean={np.mean(vals):.4f}  "
                f"median={np.median(vals):.4f}  "
                f"(GT features: {np.mean(n_gt):.0f}, "
                f"pred features: {np.mean(n_pred):.0f})"
            )


# ── CLI ────────────────────────────────────────────────────────────────


FORMAT_CHOICES = ["bulk-eval-list", "composite-tensors", "repaint-runs", "vcnn-eval"]


def main():
    parser = argparse.ArgumentParser(
        description="Compute multi-scalar persistent homology topology metrics"
    )
    parser.add_argument(
        "--results-pt",
        type=str,
        required=True,
        help="Path to .pt file or directory with saved reconstructions",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save output .pt with persistence metrics",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=FORMAT_CHOICES,
        default="bulk-eval-list",
        help="Format of the input data (default: bulk-eval-list)",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        help="Gaussian smoothing before differentiation (default: 1.0)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of samples (for quick testing)",
    )
    args = parser.parse_args()

    # Select loader
    if args.format == "bulk-eval-list":
        print(f"Loading bulk-eval-list from {args.results_pt}")
        iterator = load_bulk_eval_list(args.results_pt)
    elif args.format == "composite-tensors":
        print(f"Loading composite tensors from {args.results_pt}")
        iterator = load_composite_tensors(args.results_pt)
    elif args.format == "repaint-runs":
        print(f"Loading repaint runs from {args.results_pt}")
        iterator = load_repaint_runs(args.results_pt)
    elif args.format == "vcnn-eval":
        print(f"Loading VCNN eval from {args.results_pt}")
        iterator = load_vcnn_eval(args.results_pt)
    else:
        raise ValueError(f"Unknown format: {args.format}")

    # Optionally limit samples
    if args.max_samples is not None:
        from itertools import islice

        iterator = islice(iterator, args.max_samples)

    print(f"Computing persistence metrics (sigma={args.sigma})...\n")
    all_metrics = evaluate_samples(iterator, sigma=args.sigma)

    # Save results
    output = metrics_to_tensors(all_metrics)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, str(output_path))
    print(f"\nSaved to {output_path}")

    # Print summary
    print_summary(all_metrics)


if __name__ == "__main__":
    main()
