"""Standard inpainting visualizations for ocean vector fields.

Centralised plotting helpers so every script produces consistent quiver
plots.  All functions use ``plot_vector_field`` from
``plots.visualization_tools.plot_vector_field_tool`` under the hood; this
module just wraps the boiler-plate we kept duplicating (land-mask helper,
GT-based shared scale, per-panel file naming, etc.).

Quick-start
-----------
>>> from plots.visualization_tools.standard_plots import (
...     compute_quiver_scale,
...     land_mask_2d,
...     plot_inpaint_panels,
...     plot_single_field,
... )

Usage from a .pt file
---------------------
>>> import torch
>>> from plots.visualization_tools.standard_plots import plot_inpaint_panels
>>> d = torch.load("results/example.pt", map_location="cpu")
>>> plot_inpaint_panels(
...     gt=d["gt"],
...     missing_mask=d["missing_mask"],
...     methods={"GP": d["gp_out"], "Adaptive": d["adaptive_out"]},
...     mse={"GP": d["gp_mse"], "Adaptive": d["ada_mse"]},
...     out_dir="results/plots",
...     prefix="example",
... )
"""

import math
import os
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch

from plots.visualization_tools.plot_vector_field_tool import plot_vector_field

# ── Constants ────────────────────────────────────────────────────────
# Divisor for median_mag → quiver scale.  Larger = longer arrows.
# Tuned so median-magnitude arrows span ~1.4 grid cells at step=2.
DEFAULT_ARROW_SCALE_DIVISOR = 1.4
DEFAULT_STEP = 2


# ── Helpers ──────────────────────────────────────────────────────────

def land_mask_2d(tensor: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Return (H, W) bool mask where *both* velocity components are ≈ 0.

    Accepts (1, 2, H, W), (2, H, W), or (H, W) input.
    """
    t = tensor.squeeze()
    if t.dim() == 3:  # (2, H, W)
        return t.abs().sum(dim=0) < eps
    if t.dim() == 2:  # already (H, W)
        return t.abs() < eps
    raise ValueError(f"Unexpected tensor shape {tensor.shape}")


def compute_quiver_scale(
    gt: torch.Tensor,
    divisor: float = DEFAULT_ARROW_SCALE_DIVISOR,
    eps: float = 1e-8,
) -> float:
    """Compute a shared quiver *scale* parameter from a ground-truth field.

    ``scale = median_magnitude / divisor``

    This value is passed to ``plot_vector_field(..., scale=scale,
    auto_rescale_for_display=False)`` so that every panel plotted with the
    same scale has arrows whose lengths are directly comparable.

    Parameters
    ----------
    gt : (2, H, W) or (1, 2, H, W) ground-truth velocity tensor.
    divisor : Larger → longer arrows.  Default 1.4 looks good for our
        64×128 ram's-head domain at step=2.
    """
    g = gt.squeeze()  # (2, H, W)
    mag = torch.sqrt(g[0] ** 2 + g[1] ** 2)
    nonzero = mag[mag > eps]
    if nonzero.numel() == 0:
        return 1.0
    median_mag = float(torch.quantile(nonzero, 0.5))
    return median_mag / divisor


def _common_kw(
    gt: torch.Tensor,
    step: int = DEFAULT_STEP,
    scale: Optional[float] = None,
    divisor: float = DEFAULT_ARROW_SCALE_DIVISOR,
) -> dict:
    """Build the ``**common_kw`` dict used by every panel."""
    lm = land_mask_2d(gt)
    if scale is None:
        scale = compute_quiver_scale(gt, divisor=divisor)
    return dict(
        step=step,
        land_mask=lm,
        crop_top_right_zero_pad=True,
        auto_rescale_for_display=False,
        scale=scale,
    )


# ── Single-field plot ────────────────────────────────────────────────

def plot_single_field(
    field: torch.Tensor,
    title: str,
    filepath: str,
    gt_for_scale: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    step: int = DEFAULT_STEP,
    missing_mask: Optional[torch.Tensor] = None,
    missing_color: str = "red",
    missing_alpha: float = 0.35,
    divisor: float = DEFAULT_ARROW_SCALE_DIVISOR,
):
    """Plot a single (2, H, W) velocity field to *filepath*.

    If *gt_for_scale* is given, the quiver scale is derived from that GT
    (so multiple calls with the same GT produce visually comparable
    arrows).  Otherwise *scale* can be provided directly, or auto-rescale
    is used as a fallback.

    Parameters
    ----------
    field : (2, H, W) or (1, 2, H, W) velocity tensor.
    gt_for_scale : Ground truth used to derive consistent arrow scale.
    scale : Explicit quiver scale (overrides gt_for_scale).
    missing_mask : (H, W) or (1, 1, H, W).  1 = missing.
    """
    f = field.squeeze()  # (2, H, W)
    if gt_for_scale is not None or scale is not None:
        ref = gt_for_scale if gt_for_scale is not None else field
        kw = _common_kw(ref, step=step, scale=scale, divisor=divisor)
    else:
        # fallback: auto-rescale from the field itself
        kw = dict(step=step, crop_top_right_zero_pad=True,
                  land_mask=land_mask_2d(field))

    extra = {}
    if missing_mask is not None:
        mm = missing_mask.squeeze()
        if mm.dim() == 3:
            mm = mm[0]
        extra = dict(missing_mask=mm, missing_color=missing_color,
                     missing_alpha=missing_alpha)

    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    plot_vector_field(f[0], f[1], title=title, file=filepath, **kw, **extra)


# ── Multi-panel convenience ──────────────────────────────────────────

def plot_inpaint_panels(
    gt: torch.Tensor,
    missing_mask: torch.Tensor,
    methods: Dict[str, torch.Tensor],
    mse: Optional[Dict[str, float]] = None,
    out_dir: str = ".",
    prefix: str = "inpaint",
    mask_label: str = "",
    step: int = DEFAULT_STEP,
    divisor: float = DEFAULT_ARROW_SCALE_DIVISOR,
    extra_titles: Optional[Dict[str, str]] = None,
):
    """Generate a standard set of inpainting comparison plots.

    Always produces:
      1. ``{prefix}_01_ground_truth.png``
      2. ``{prefix}_02_masked.png``   (GT + red missing overlay)
      3+ ``{prefix}_03_{name}.png`` … one per entry in *methods*

    Parameters
    ----------
    gt : (1, 2, H, W) ground-truth velocity field (physical / unstd).
    missing_mask : (1, 2, H, W) or (1, 1, H, W).  1 = missing.
    methods : ``{"GP": gp_tensor, "Adaptive": ada_tensor, ...}``
        Each value is (1, 2, H, W) in physical space.
    mse : Optional per-method MSE values for plot titles.
    out_dir : Directory for output PNGs.
    prefix : Filename prefix.
    mask_label : Extra text for the mask panel title (e.g. "transect").
    extra_titles : Optional per-method extra title text.
    """
    os.makedirs(out_dir, exist_ok=True)
    g = gt.cpu().squeeze()       # (2, H, W)
    mm = missing_mask.cpu().squeeze()
    if mm.dim() == 3:
        mm_2d = mm[0]            # (H, W)
    else:
        mm_2d = mm

    scale = compute_quiver_scale(g, divisor=divisor)
    kw = _common_kw(g, step=step, scale=scale, divisor=divisor)

    # Mask percentage
    lm = land_mask_2d(g)
    ocean = (~lm).float()
    mask_pct = float(mm_2d[ocean.bool()].sum() / (ocean.sum() + 1e-8) * 100)

    # 1. Ground truth
    plot_vector_field(
        g[0], g[1], title="Ground Truth",
        file=os.path.join(out_dir, f"{prefix}_01_ground_truth.png"),
        **kw,
    )

    # 2. Ground truth + mask overlay
    mask_title = f"Masked Input ({mask_pct:.1f}% missing)"
    if mask_label:
        mask_title += f" — {mask_label}"
    plot_vector_field(
        g[0], g[1], title=mask_title,
        file=os.path.join(out_dir, f"{prefix}_02_masked.png"),
        missing_mask=mm_2d, missing_color="red", missing_alpha=0.35,
        **kw,
    )

    # 3+ Method panels
    for i, (name, tensor) in enumerate(methods.items(), start=3):
        f = tensor.cpu().squeeze()
        title_parts = [name]
        if mse and name in mse:
            title_parts.append(f"MSE={mse[name]:.6f}")
        if extra_titles and name in extra_titles:
            title_parts.append(extra_titles[name])
        title = "  ".join(title_parts)

        safe_name = name.lower().replace(" ", "_").replace("+", "_")
        plot_vector_field(
            f[0], f[1], title=title,
            file=os.path.join(out_dir, f"{prefix}_{i:02d}_{safe_name}.png"),
            **kw,
        )

    n_panels = 2 + len(methods)
    print(f"Saved {n_panels} plots to {out_dir}/  (prefix={prefix})")


# ── Replot from .pt ──────────────────────────────────────────────────

def replot_from_pt(
    pt_path: str,
    out_dir: Optional[str] = None,
    prefix: Optional[str] = None,
    step: int = DEFAULT_STEP,
    divisor: float = DEFAULT_ARROW_SCALE_DIVISOR,
):
    """Regenerate standard plots from a saved .pt file.

    Expects the .pt to contain at minimum:
      - ``gt``:           (1, 2, H, W) ground truth
      - ``missing_mask``: (1, 2, H, W) mask
      - ``gp_out``:       (1, 2, H, W) GP result
    And optionally:
      - ``adaptive_out``: (1, 2, H, W) adaptive result
      - ``gp_mse``, ``ada_mse``: scalar MSE values
      - ``mask_pct``: scalar
    """
    d = torch.load(pt_path, map_location="cpu", weights_only=False)

    if out_dir is None:
        out_dir = os.path.dirname(pt_path)
    if prefix is None:
        prefix = os.path.splitext(os.path.basename(pt_path))[0]

    methods = {"GP": d["gp_out"]}
    mse_dict = {}
    extra_titles = {}

    if "gp_mse" in d:
        mse_dict["GP"] = d["gp_mse"]

    if "adaptive_out" in d:
        methods["Adaptive"] = d["adaptive_out"]
        # Support both "ada_mse" and legacy "adaptive_mse" key names
        ada_mse_val = d.get("ada_mse", d.get("adaptive_mse", None))
        if ada_mse_val is not None:
            mse_dict["Adaptive"] = ada_mse_val
            if "gp_mse" in d and d["gp_mse"] > 0:
                ratio = ada_mse_val / d["gp_mse"]
                extra_titles["Adaptive"] = f"({ratio:.3f}x GP)"

    plot_inpaint_panels(
        gt=d["gt"],
        missing_mask=d["missing_mask"],
        methods=methods,
        mse=mse_dict,
        out_dir=out_dir,
        prefix=prefix,
        step=step,
        divisor=divisor,
        extra_titles=extra_titles,
    )


# ── Bar-chart comparison ─────────────────────────────────────────────

# Default colours matching the reference style
_BAR_COLORS = {
    "DDPM":              "#4285F4",  # Google blue
    "Adaptive":          "#4285F4",
    "Adaptive GP-Refined": "#4285F4",
    "Gaussian Process":  "#DB4437",  # Google red / coral
    "GP":                "#DB4437",
}

_FALLBACK_COLORS = ["#4285F4", "#DB4437", "#F4B400", "#0F9D58",
                    "#AB47BC", "#00ACC1"]


def plot_method_comparison_bars(
    data: Dict[str, Sequence[float]],
    metric_label: str,
    filepath: str,
    title: Optional[str] = None,
    colors: Optional[Dict[str, str]] = None,
    figsize: tuple = (4.5, 5.5),
    capsize: float = 4,
    bar_width: float = 0.55,
    ylim: Optional[tuple] = None,
    show: bool = False,
):
    """Create a bar chart comparing methods — style matches the reference image.

    Parameters
    ----------
    data : ``{"DDPM": [0.49, 0.52, ...], "Gaussian Process": [0.41, ...]}``
        Method name → list of per-sample metric values.
    metric_label : X-axis label (e.g. ``"Magnitude Error"``).
    filepath : Output PNG path.
    title : Optional chart title (omitted if None for the clean look).
    colors : Per-method colors; falls back to built-in palette.
    figsize : Figure (width, height) in inches.
    capsize : Width of error-bar caps in points.
    bar_width : Width of each bar.
    ylim : Explicit y-axis limits ``(ymin, ymax)``; auto if None.
    show : Call ``plt.show()`` after saving.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = list(data.keys())
    n_methods = len(methods)

    means, sems = [], []
    for m in methods:
        vals = [v for v in data[m] if not (isinstance(v, float) and math.isnan(v))]
        arr = np.array(vals, dtype=np.float64)
        means.append(float(arr.mean()))
        sems.append(float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0)

    # Colours
    cmap = {**_BAR_COLORS, **(colors or {})}
    bar_colors = [cmap.get(m, _FALLBACK_COLORS[i % len(_FALLBACK_COLORS)])
                  for i, m in enumerate(methods)]

    # ── plot ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(n_methods)

    bars = ax.bar(
        x, means, width=bar_width, color=bar_colors,
        edgecolor="none", zorder=3,
    )
    ax.errorbar(
        x, means, yerr=sems, fmt="none",
        ecolor="#555555", elinewidth=1.2, capsize=capsize, capthick=1.2,
        zorder=4,
    )

    # ── styling to match reference ────────────────────────────────────
    ax.set_xticks([])                       # no x-tick labels (legend only)
    ax.set_xlabel(metric_label, fontsize=13, labelpad=10)
    if title:
        ax.set_title(title, fontsize=14, pad=12)

    # y-axis
    if ylim:
        ax.set_ylim(ylim)
    else:
        ymax = max(m + s for m, s in zip(means, sems)) * 1.2
        ax.set_ylim(0, ymax)
    ax.tick_params(axis="y", labelsize=11)

    # Light grid, white background
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    ax.yaxis.grid(True, color="#E0E0E0", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Legend
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=c, label=m) for m, c in zip(methods, bar_colors)]
    ax.legend(handles=legend_handles, loc="upper center",
              bbox_to_anchor=(0.5, 1.08), ncol=min(n_methods, 3),
              frameon=False, fontsize=11)

    fig.tight_layout()
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    fig.savefig(filepath, dpi=200, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    plt.close(fig)
    print(f"Saved bar chart → {filepath}")


# ── Scatter + trendline vs coverage ──────────────────────────────────

def plot_metric_vs_coverage(
    coverage: Sequence[float],
    methods: Dict[str, Sequence[float]],
    y_label: str,
    filepath: str,
    x_label: str = "Coverage",
    title: Optional[str] = None,
    colors: Optional[Dict[str, str]] = None,
    marker: str = "^",
    marker_size: float = 28,
    marker_alpha: float = 0.6,
    trendline: bool = True,
    figsize: tuple = (12, 5),
    xlim: Optional[tuple] = None,
    ylim: Optional[tuple] = None,
    x_as_percent: bool = True,
    show: bool = False,
):
    """Scatter plot of per-sample metric vs coverage % with trendlines.

    Matches the reference style: coloured triangle markers for each sample,
    smooth power-law trendlines, legend at top, x-axis as percentages.

    Parameters
    ----------
    coverage : Per-sample coverage percentages (len N).
    methods : ``{"Adaptive GP-Refined": [...], "Gaussian Process": [...]}``
        Per-sample metric values (same length as *coverage*).
    y_label : Y-axis label (e.g. ``"Mean Squared Error"``).
    filepath : Output PNG path.
    trendline : Fit and draw a smooth curve through the data.
    x_as_percent : Format x-axis ticks as ``"1.00%"``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    cov = np.array(coverage, dtype=np.float64)

    cmap = {**_BAR_COLORS, **(colors or {})}

    fig, ax = plt.subplots(figsize=figsize)

    legend_items = []

    for i, (name, vals) in enumerate(methods.items()):
        y = np.array(vals, dtype=np.float64)
        valid = ~np.isnan(y)
        cx, cy = cov[valid], y[valid]

        color = cmap.get(name, _FALLBACK_COLORS[i % len(_FALLBACK_COLORS)])

        # Scatter
        ax.scatter(cx, cy, marker=marker, s=marker_size, color=color,
                   alpha=marker_alpha, edgecolors="none", zorder=3,
                   label=f"{name}")

        # Trendline (power law fit: y = a * x^b)
        if trendline and len(cx) > 5:
            try:
                # Filter to positive values for log fit
                pos = (cx > 0) & (cy > 0)
                if pos.sum() > 5:
                    log_x = np.log(cx[pos])
                    log_y = np.log(cy[pos])
                    coeffs = np.polyfit(log_x, log_y, 1)
                    b, log_a = coeffs[0], coeffs[1]
                    a = np.exp(log_a)
                    x_smooth = np.linspace(cx.min() * 0.8, cx.max() * 1.1, 200)
                    y_smooth = a * x_smooth ** b
                    ax.plot(x_smooth, y_smooth, color=color, linewidth=2.5,
                            alpha=0.8, zorder=2, label=f"Trendline for {name}")
            except Exception:
                pass  # skip trendline if fit fails

    # ── styling ──────────────────────────────────────────────────────
    ax.set_ylabel(y_label, fontsize=13, labelpad=10)
    ax.set_xlabel(x_label, fontsize=13, labelpad=10)
    if title:
        ax.set_title(title, fontsize=14, pad=12)

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim(bottom=0)
    ax.tick_params(axis="both", labelsize=11)

    if x_as_percent:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}%"))

    # White background, light grid
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    ax.yaxis.grid(True, color="#E0E0E0", linewidth=0.6, zorder=0)
    ax.xaxis.grid(True, color="#E0E0E0", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Legend at top
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12),
              ncol=4, frameon=False, fontsize=10, markerscale=1.5)

    fig.tight_layout()
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    fig.savefig(filepath, dpi=200, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    plt.close(fig)
    print(f"Saved scatter plot → {filepath}")
