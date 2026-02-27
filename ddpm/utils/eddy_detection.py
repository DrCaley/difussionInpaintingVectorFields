"""
Eddy detection and evaluation for 2-D ocean velocity fields.

Two detection methods are provided:

**Gamma1 method (default)** — Graftieaux et al. (2001)
    For each point P, compute
        Γ₁(P) = (1/N) Σ_M sin(angle(PM, v(M)))
    where M ranges over a disk of given radius around P.
    |Γ₁| ≈ 1 at the centre of a vortex, ≈ 0 at shear zones.
    This directly tests for *rotational* flow and is robust against
    boundary-layer vorticity, flow separation, and headland effects.

**Okubo–Weiss method (legacy)**
    W = s_n² + s_s² − ω²
    A connected vorticity-dominated region (W < W₀ < 0) is labelled
    an eddy.  This over-detects in coastal domains where strong shear
    creates vorticity-dominated regions that are not eddies.

References
----------
- Graftieaux, Michard & Grosjean 2001, Meas. Sci. Technol., 12, 1422
- Okubo 1970, Deep-Sea Res., 17, 445–454
- Weiss 1991, Physica D, 48, 273–294
- Isern-Fontanet et al. 2006, J. Geophys. Res., 111, C01004
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Eddy:
    """One detected eddy."""
    label: int                         # connected-component label
    center_y: float                    # centroid (row, fractional)
    center_x: float                    # centroid (col, fractional)
    area_pixels: int                   # number of pixels
    mean_vorticity: float              # signed mean ω inside the eddy
    mean_ow: float                     # mean OW value inside the eddy
    min_ow: float                      # most negative OW inside the eddy
    bbox: Tuple[int, int, int, int]    # (row_min, row_max, col_min, col_max)
    is_cyclonic: bool                  # ω > 0  (Northern Hemisphere convention)
    swirl_fraction: float = 0.0        # mean |v_tangential| / |v| in eddy region
    mask: Optional[torch.Tensor] = field(default=None, repr=False)  # (H,W) bool


# ---------------------------------------------------------------------------
# Velocity-gradient tensor components  (central differences)
# ---------------------------------------------------------------------------

def _velocity_gradients(vel: torch.Tensor, dx: float = 1.0, dy: float = 1.0):
    """
    Compute all four 2-D velocity gradient components.

    Parameters
    ----------
    vel : (B, 2, H, W)  or  (2, H, W)
        Channel 0 = u (x-velocity), channel 1 = v (y-velocity).
    dx, dy : grid spacing in x and y (pixels by default).

    Returns
    -------
    du_dx, du_dy, dv_dx, dv_dy : each (B, H, W), zero-padded at boundaries.
    """
    squeeze = False
    if vel.dim() == 3:
        vel = vel.unsqueeze(0)
        squeeze = True

    u = vel[:, 0]  # (B, H, W)
    v = vel[:, 1]

    du_dx = torch.zeros_like(u)
    du_dy = torch.zeros_like(u)
    dv_dx = torch.zeros_like(v)
    dv_dy = torch.zeros_like(v)

    # Central differences on interior
    du_dx[:, 1:-1, 1:-1] = (u[:, 1:-1, 2:] - u[:, 1:-1, :-2]) / (2.0 * dx)
    du_dy[:, 1:-1, 1:-1] = (u[:, 2:, 1:-1] - u[:, :-2, 1:-1]) / (2.0 * dy)
    dv_dx[:, 1:-1, 1:-1] = (v[:, 1:-1, 2:] - v[:, 1:-1, :-2]) / (2.0 * dx)
    dv_dy[:, 1:-1, 1:-1] = (v[:, 2:, 1:-1] - v[:, :-2, 1:-1]) / (2.0 * dy)

    if squeeze:
        du_dx, du_dy = du_dx.squeeze(0), du_dy.squeeze(0)
        dv_dx, dv_dy = dv_dx.squeeze(0), dv_dy.squeeze(0)

    return du_dx, du_dy, dv_dx, dv_dy


# ---------------------------------------------------------------------------
# Okubo–Weiss field
# ---------------------------------------------------------------------------

def okubo_weiss(vel: torch.Tensor, dx: float = 1.0, dy: float = 1.0):
    """
    Compute the Okubo–Weiss parameter field.

        W = s_n² + s_s² − ω²

    Parameters
    ----------
    vel : (B, 2, H, W)  or  (2, H, W)
    dx, dy : grid spacing

    Returns
    -------
    W        : Okubo–Weiss field, same leading dims as input but H,W spatial
    vorticity: ω = ∂v/∂x − ∂u/∂y
    s_n      : normal strain  ∂u/∂x − ∂v/∂y
    s_s      : shear strain   ∂v/∂x + ∂u/∂y
    """
    du_dx, du_dy, dv_dx, dv_dy = _velocity_gradients(vel, dx, dy)

    s_n = du_dx - dv_dy    # normal strain
    s_s = dv_dx + du_dy    # shear strain
    omega = dv_dx - du_dy  # vorticity

    W = s_n ** 2 + s_s ** 2 - omega ** 2
    return W, omega, s_n, s_s


# ---------------------------------------------------------------------------
# Eddy detection via connected components on thresholded OW field
# ---------------------------------------------------------------------------

def _connected_components_cpu(binary: np.ndarray) -> np.ndarray:
    """Label connected components using scipy (CPU fallback)."""
    from scipy import ndimage
    labels, _ = ndimage.label(binary)
    return labels


def detect_eddies(
    vel: torch.Tensor,
    dx: float = 1.0,
    dy: float = 1.0,
    threshold_sigma: float = 0.2,
    threshold_value: Optional[float] = None,
    min_area: int = 16,
    ocean_mask: Optional[torch.Tensor] = None,
    shore_buffer: int = 2,
    min_swirl_fraction: float = 0.6,
) -> Tuple[List[Eddy], torch.Tensor, torch.Tensor]:
    """
    Detect eddies in a 2-D velocity field using the OW parameter.

    Parameters
    ----------
    vel : (2, H, W)  — single velocity field
    dx, dy : grid spacing
    threshold_sigma : W₀ = −threshold_sigma × std(W).  Default 0.2 is the
        standard Isern-Fontanet (2006) choice.
    threshold_value : If given, override the sigma-based threshold with a
        fixed value (useful for comparing fields on the same scale).
    min_area : reject blobs smaller than this (pixels).  Default 16 filters
        out small noise clusters (a real eddy should be at least ~4×4).
    ocean_mask : (H, W) bool or float, True / 1 = valid ocean pixel.
        If provided, OW is computed only over ocean pixels and land is
        excluded from connected components.  When *None* the mask is
        auto-derived from the velocity field (land pixels have u=v=0).
    shore_buffer : number of pixels to erode from the ocean mask boundary.
        Velocity gradients near the land–ocean interface are noisy because
        central differences mix zero (land) and nonzero (ocean) values.
        Default 2 removes the two rows/columns closest to land, which the
        diagnostic sweep showed have ~4× the vorticity noise of the interior.
        Set to 0 to disable.
    min_swirl_fraction : minimum mean tangential velocity fraction to keep a
        candidate.  For each pixel in the candidate region, we decompose
        velocity into tangential (perpendicular to radius from center) and
        radial components.  swirl_fraction = mean(|v_tan| / |v|).  A true
        eddy has most velocity tangential (≈0.7–0.9); shear zones, boundary
        currents, and flow separation have lower values.  Default 0.6.

    Returns
    -------
    eddies : list of Eddy dataclasses
    W      : (H, W) Okubo–Weiss field
    omega  : (H, W) vorticity field
    """
    assert vel.dim() == 3 and vel.shape[0] == 2, \
        f"Expected (2, H, W), got {vel.shape}"

    W, omega, s_n, s_s = okubo_weiss(vel, dx, dy)
    # W, omega are (H, W) after squeeze inside okubo_weiss

    # --- build / refine ocean mask ---
    if ocean_mask is not None:
        ocean = ocean_mask.bool()
    else:
        # Auto-derive: land pixels have both u == 0 and v == 0
        speed = vel[0] ** 2 + vel[1] ** 2
        ocean = speed > 0

    # Erode the mask to exclude noisy shoreline pixels
    if shore_buffer > 0 and ocean.any():
        from scipy.ndimage import binary_erosion
        ocean_np = ocean.cpu().numpy()
        eroded = binary_erosion(ocean_np, iterations=shore_buffer)
        ocean = torch.from_numpy(eroded).to(vel.device)

    W_ocean = W[ocean] if ocean.any() else W

    if threshold_value is not None:
        W0 = threshold_value
    else:
        W0 = -threshold_sigma * W_ocean.std().item()

    eddy_binary = (W < W0) & ocean  # (H, W)

    # --- connected components (CPU, scipy) ---
    labels_np = _connected_components_cpu(eddy_binary.cpu().numpy())
    labels = torch.from_numpy(labels_np).to(vel.device)

    # --- extract eddy properties ---
    eddies: List[Eddy] = []
    unique_labels = torch.unique(labels)
    for lbl in unique_labels:
        if lbl == 0:
            continue  # background
        lbl_int = lbl.item()
        mask_lbl = labels == lbl_int
        area = mask_lbl.sum().item()
        if area < min_area:
            continue

        ys, xs = torch.where(mask_lbl)
        cy = ys.float().mean().item()
        cx = xs.float().mean().item()
        mean_vor = omega[mask_lbl].mean().item()
        mean_ow = W[mask_lbl].mean().item()
        min_ow = W[mask_lbl].min().item()
        bbox = (ys.min().item(), ys.max().item(),
                xs.min().item(), xs.max().item())

        # --- swirl fraction: tangential velocity / total velocity ---
        # For each pixel, decompose velocity into radial (toward center)
        # and tangential (perpendicular) components.  A true eddy has
        # high tangential fraction; shear zones and boundary currents don't.
        dy_c = ys.float() - cy  # radius vector y-component
        dx_c = xs.float() - cx  # radius vector x-component
        r = (dx_c**2 + dy_c**2).sqrt().clamp(min=0.5)
        # Unit radial vector
        rx = dx_c / r
        ry = dy_c / r

        u_eddy = vel[0][ys, xs]
        v_eddy = vel[1][ys, xs]
        speed = (u_eddy**2 + v_eddy**2).sqrt().clamp(min=1e-10)

        # Radial component: dot(vel, r_hat)
        v_radial = u_eddy * rx + v_eddy * ry
        # Tangential fraction = |v_tangential| / |v|
        v_tan_sq = speed**2 - v_radial**2
        v_tan_mag = v_tan_sq.clamp(min=0).sqrt()
        swirl_frac = (v_tan_mag / speed).mean().item()

        if swirl_frac < min_swirl_fraction:
            continue  # not a real eddy — velocity doesn't rotate around center

        eddies.append(Eddy(
            label=lbl_int,
            center_y=cy,
            center_x=cx,
            area_pixels=area,
            mean_vorticity=mean_vor,
            mean_ow=mean_ow,
            min_ow=min_ow,
            bbox=bbox,
            is_cyclonic=(mean_vor > 0),
            swirl_fraction=swirl_frac,
            mask=mask_lbl,
        ))

    # Sort by area (largest first)
    eddies.sort(key=lambda e: e.area_pixels, reverse=True)
    return eddies, W, omega


# ---------------------------------------------------------------------------
# Gamma1 vortex identification (Graftieaux et al. 2001)
# ---------------------------------------------------------------------------

def gamma1_field(
    vel: torch.Tensor,
    radius: int = 10,
    ocean_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute the Γ₁ scalar field for vortex centre identification.

    For each point P in the domain,
        Γ₁(P) = (1/N) Σ_{M ∈ disk(P,r)} sin(θ_M)
    where θ_M = angle between the vector PM and the velocity v(M).

    |Γ₁| ≈ 1 at the centre of an ideal point vortex, ≈ 0 in shear flow.
    Γ₁ > 0 → counter-clockwise (cyclonic in NH), Γ₁ < 0 → clockwise.

    Parameters
    ----------
    vel : (2, H, W)   — single velocity field, channel 0=u, 1=v
    radius : int       — neighbourhood radius (pixels).  Should roughly match
                         the expected eddy radius.  For large eddies in a
                         44×94 domain, 8–12 is a good choice.
    ocean_mask : (H, W) bool — True = valid ocean pixel.  If None, derived
                 from velocity (speed > 0).

    Returns
    -------
    gamma1 : (H, W)  — the Γ₁ field in [-1, +1].
    """
    assert vel.dim() == 3 and vel.shape[0] == 2
    u = vel[0]   # (H, W)
    v = vel[1]
    H, W = u.shape
    device = vel.device

    if ocean_mask is None:
        ocean_mask = (u ** 2 + v ** 2) > 0

    speed = (u ** 2 + v ** 2).sqrt().clamp(min=1e-10)

    sin_accum = torch.zeros(H, W, device=device)
    count = torch.zeros(H, W, device=device)

    # Vectorised over all offsets (dy, dx) within the circular annulus.
    # For each offset, we shift the velocity grid and accumulate the
    # sin(angle) contribution.  Total iterations ≈ π·radius².
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            dist = (dy ** 2 + dx ** 2) ** 0.5
            if dist < 1 or dist > radius:
                continue

            # Valid P-range: P=(py,px) such that M=(py+dy,px+dx) is in bounds
            py_s = max(0, -dy)
            py_e = min(H, H - dy)
            px_s = max(0, -dx)
            px_e = min(W, W - dx)
            if py_e <= py_s or px_e <= px_s:
                continue

            p_ocean = ocean_mask[py_s:py_e, px_s:px_e]
            m_ocean = ocean_mask[py_s + dy:py_e + dy, px_s + dx:px_e + dx]
            valid = p_ocean & m_ocean

            um = u[py_s + dy:py_e + dy, px_s + dx:px_e + dx]
            vm = v[py_s + dy:py_e + dy, px_s + dx:px_e + dx]
            sm = speed[py_s + dy:py_e + dy, px_s + dx:px_e + dx]

            # sin(angle(PM, v(M))) = (PM × v) / (|PM| · |v|)
            # PM = (dx, dy),  v = (um, vm)
            cross = dx * vm - dy * um          # scalar cross product
            sin_theta = cross / (dist * sm)    # normalised

            sin_accum[py_s:py_e, px_s:px_e] += torch.where(valid, sin_theta, torch.zeros_like(sin_theta))
            count[py_s:py_e, px_s:px_e] += valid.float()

    gamma1 = sin_accum / count.clamp(min=1)
    # Mask out land
    gamma1[~ocean_mask] = 0.0
    return gamma1


def detect_eddies_gamma(
    vel: torch.Tensor,
    radius: int = 10,
    gamma_threshold: float = 0.7,
    min_area: int = 50,
    ocean_mask: Optional[torch.Tensor] = None,
    shore_buffer: int = 3,
    smooth_sigma: float = 2.0,
    min_mean_speed_ratio: float = 0.3,
    min_vorticity: float = 0.0,
) -> Tuple[List[Eddy], torch.Tensor, torch.Tensor]:
    """
    Detect eddies using the Gamma1 vortex-identification method.

    This is better suited to large-scale circulation features than OW
    because it directly tests whether velocity *rotates around* a point.

    Parameters
    ----------
    vel : (2, H, W) — single velocity field
    radius : neighbourhood radius for the Gamma1 computation (pixels).
        Should roughly match the expected eddy radius.  Default 10.
    gamma_threshold : |Γ₁| must exceed this to qualify as eddy-like.
        Default 0.7 (Graftieaux et al. recommend 0.9 for ideal vortices;
        real ocean data is noisier, so 0.7 is a reasonable start).
    min_area : minimum connected-component area in pixels.  Default 50.
    ocean_mask : (H, W) bool.  If None, derived from vel.
    shore_buffer : erode ocean mask by this many pixels before analysis.
    smooth_sigma : Gaussian smoothing σ (pixels) applied to vel before
        computing Gamma1.  Larger values isolate larger features.
        Default 2.0.  Set 0 to disable.
    min_mean_speed_ratio : minimum ratio of the eddy's mean speed to the
        field's mean ocean speed.  Gamma1 only measures rotation direction —
        it's blind to magnitude, so near-zero-speed regions with rotating
        vectors score high.  This filter rejects candidates whose mean
        speed is too low relative to the overall flow.  Default 0.3 (30%).
        Set to 0 to disable.
    min_vorticity : minimum absolute mean vorticity |ω| inside the eddy.
        Rejects candidates with weak rotational intensity.  Default 0.0
        (disabled).  A value of ~0.03 eliminates most false positives
        while preserving strong real eddies.

    Returns
    -------
    eddies : list[Eddy]   — sorted by area, largest first
    gamma1 : (H, W)       — the Γ₁ field
    vorticity : (H, W)    — ω field (for downstream plotting compat)
    """
    assert vel.dim() == 3 and vel.shape[0] == 2, \
        f"Expected (2, H, W), got {vel.shape}"
    from scipy.ndimage import binary_erosion, gaussian_filter

    H, W = vel.shape[1], vel.shape[2]

    # --- build / refine ocean mask ---
    if ocean_mask is None:
        spd = vel[0] ** 2 + vel[1] ** 2
        ocean = spd > 0
    else:
        ocean = ocean_mask.bool()

    if shore_buffer > 0 and ocean.any():
        ocean_np = ocean.cpu().numpy()
        eroded = binary_erosion(ocean_np, iterations=shore_buffer)
        ocean = torch.from_numpy(eroded).to(vel.device)

    # --- compute field-wide speed stats for min-speed filter ---
    speed_field = (vel[0] ** 2 + vel[1] ** 2).sqrt()
    ocean_speeds = speed_field[ocean]
    field_mean_speed = ocean_speeds.mean().item() if ocean_speeds.numel() > 0 else 0.0

    # --- optional Gaussian smoothing ---
    vel_s = vel.clone()
    if smooth_sigma > 0:
        for ch in range(2):
            arr = vel_s[ch].cpu().numpy().copy()
            arr[~ocean.cpu().numpy()] = 0.0
            smoothed = gaussian_filter(arr, sigma=smooth_sigma)
            vel_s[ch] = torch.from_numpy(smoothed).to(vel.device)
            # Zero out land again
            vel_s[ch][~ocean] = 0.0

    # --- compute Gamma1 ---
    g1 = gamma1_field(vel_s, radius=radius, ocean_mask=ocean)

    # --- also compute vorticity for downstream compat ---
    _, omega, _, _ = okubo_weiss(vel)   # unsmoothed

    # --- threshold and find connected components ---
    eddy_binary = (g1.abs() > gamma_threshold) & ocean
    labels_np = _connected_components_cpu(eddy_binary.cpu().numpy())
    labels = torch.from_numpy(labels_np).to(vel.device)

    # --- extract eddy properties ---
    min_speed = min_mean_speed_ratio * field_mean_speed
    eddies: List[Eddy] = []
    for lbl in torch.unique(labels):
        if lbl == 0:
            continue
        lbl_int = lbl.item()
        mask_lbl = labels == lbl_int
        area = mask_lbl.sum().item()
        if area < min_area:
            continue

        ys, xs = torch.where(mask_lbl)

        # --- min speed filter: reject low-magnitude phantom eddies ---
        eddy_speed = speed_field[ys, xs].mean().item()
        if min_mean_speed_ratio > 0 and eddy_speed < min_speed:
            continue

        cy = ys.float().mean().item()
        cx = xs.float().mean().item()
        mean_vor = omega[mask_lbl].mean().item()

        # --- min vorticity filter: reject weak-rotation candidates ---
        if min_vorticity > 0 and abs(mean_vor) < min_vorticity:
            continue
        mean_g1 = g1[mask_lbl].mean().item()
        peak_g1 = g1[mask_lbl].abs().max().item()
        bbox = (ys.min().item(), ys.max().item(),
                xs.min().item(), xs.max().item())

        # Eddy centre = peak |Gamma1| location within the component
        g1_in_comp = g1[mask_lbl].abs()
        peak_idx = g1_in_comp.argmax()
        cy_peak = ys[peak_idx].item()
        cx_peak = xs[peak_idx].item()

        eddies.append(Eddy(
            label=lbl_int,
            center_y=float(cy_peak),
            center_x=float(cx_peak),
            area_pixels=area,
            mean_vorticity=mean_vor,
            mean_ow=float(mean_g1),   # store mean Gamma1 in OW field slot
            min_ow=float(-peak_g1),   # store -peak |Gamma1| for compat
            bbox=bbox,
            is_cyclonic=(mean_g1 > 0),   # Γ₁ > 0 → counter-clockwise
            swirl_fraction=float(peak_g1),  # peak |Γ₁| as quality score
            mask=mask_lbl,
        ))

    eddies.sort(key=lambda e: e.area_pixels, reverse=True)
    return eddies, g1, omega


# ---------------------------------------------------------------------------
# Unified dispatcher — selects method by name
# ---------------------------------------------------------------------------

def detect_eddies_dispatch(
    vel: torch.Tensor,
    method: str = "gamma",
    **kwargs,
) -> Tuple[List[Eddy], torch.Tensor, torch.Tensor]:
    """Select eddy detection method by name.

    method='gamma' (default) → detect_eddies_gamma(vel, **kwargs)
    method='ow'              → detect_eddies(vel, **kwargs)   (legacy OW)
    """
    if method == "gamma":
        return detect_eddies_gamma(vel, **kwargs)
    elif method == "ow":
        return detect_eddies(vel, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'gamma' or 'ow'.")


# ---------------------------------------------------------------------------
# Eddy-aware evaluation metrics
# ---------------------------------------------------------------------------

def eddy_metrics(
    pred: torch.Tensor,
    true: torch.Tensor,
    dx: float = 1.0,
    dy: float = 1.0,
    threshold_sigma: float = 0.2,
    ocean_mask: Optional[torch.Tensor] = None,
    min_area: int = 16,
    shore_buffer: int = 2,
) -> dict:
    """
    Compute eddy-specific evaluation metrics comparing a predicted velocity
    field against ground truth.

    We detect eddies in **ground truth** and then evaluate how well the
    prediction reconstructs those eddy regions.

    Parameters
    ----------
    pred, true : (2, H, W)  velocity fields (unstandardized, physical units)
    dx, dy     : grid spacing
    threshold_sigma : OW threshold for eddy detection (applied to GT)
    ocean_mask : optional ocean mask
    min_area   : minimum eddy area in pixels

    Returns
    -------
    dict with keys:
        # --- OW-field-level metrics ---
        ow_mse           : MSE of OW fields (pred vs true)
        ow_correlation   : Pearson r of OW fields over ocean
        ow_mae           : MAE of OW fields

        # --- vorticity-field-level metrics ---
        vort_mse         : MSE of vorticity fields
        vort_correlation : Pearson r of vorticity fields over ocean
        vort_mae         : MAE of vorticity fields

        # --- eddy detection agreement ---
        n_eddies_true    : number of eddies in ground truth
        n_eddies_pred    : number of eddies in prediction
        eddy_iou         : IoU of combined eddy masks (all eddies, binary)
        eddy_detection_rate : fraction of true eddies that have ≥50% overlap
                              with a predicted eddy

        # --- per-eddy reconstruction quality ---
        eddy_velocity_mse     : MSE of velocity inside true-eddy regions
        eddy_angular_error    : mean angular error (degrees) inside eddies
        eddy_vorticity_mse    : MSE of vorticity inside true-eddy regions
        eddy_center_distance  : mean distance between matched eddy centres
        non_eddy_velocity_mse : MSE of velocity outside eddy regions (for contrast)

        # --- shared threshold (for cross-field comparison) ---
        ow_threshold     : the W₀ value used
    """
    assert pred.dim() == 3 and true.dim() == 3
    device = pred.device
    results = {}

    # 1) Compute OW and vorticity for both fields
    W_true, omega_true, _, _ = okubo_weiss(true, dx, dy)
    W_pred, omega_pred, _, _ = okubo_weiss(pred, dx, dy)

    if ocean_mask is not None:
        ocean = ocean_mask.bool()
    else:
        ocean = torch.ones(true.shape[1:], dtype=torch.bool, device=device)

    # 2) OW field-level metrics
    w_t = W_true[ocean]
    w_p = W_pred[ocean]
    results['ow_mse'] = ((w_t - w_p) ** 2).mean().item()
    results['ow_mae'] = (w_t - w_p).abs().mean().item()
    if w_t.std() > 0 and w_p.std() > 0:
        results['ow_correlation'] = torch.corrcoef(
            torch.stack([w_t, w_p])
        )[0, 1].item()
    else:
        results['ow_correlation'] = 0.0

    # 3) Vorticity field-level metrics
    v_t = omega_true[ocean]
    v_p = omega_pred[ocean]
    results['vort_mse'] = ((v_t - v_p) ** 2).mean().item()
    results['vort_mae'] = (v_t - v_p).abs().mean().item()
    if v_t.std() > 0 and v_p.std() > 0:
        results['vort_correlation'] = torch.corrcoef(
            torch.stack([v_t, v_p])
        )[0, 1].item()
    else:
        results['vort_correlation'] = 0.0

    # 4) Detect eddies in ground truth (defines the "answer key")
    #    Use the same threshold for both to enable fair comparison.
    W0 = -threshold_sigma * W_true[ocean].std().item()
    results['ow_threshold'] = W0

    eddies_true, _, _ = detect_eddies(
        true, dx, dy, threshold_value=W0,
        min_area=min_area, ocean_mask=ocean_mask,
        shore_buffer=shore_buffer,
    )
    eddies_pred, _, _ = detect_eddies(
        pred, dx, dy, threshold_value=W0,
        min_area=min_area, ocean_mask=ocean_mask,
        shore_buffer=shore_buffer,
    )
    results['n_eddies_true'] = len(eddies_true)
    results['n_eddies_pred'] = len(eddies_pred)

    # 5) Binary eddy masks
    H, W_dim = true.shape[1], true.shape[2]
    true_eddy_mask = torch.zeros(H, W_dim, dtype=torch.bool, device=device)
    for e in eddies_true:
        true_eddy_mask |= e.mask.to(device)

    pred_eddy_mask = torch.zeros(H, W_dim, dtype=torch.bool, device=device)
    for e in eddies_pred:
        pred_eddy_mask |= e.mask.to(device)

    # 6) IoU of eddy regions
    intersection = (true_eddy_mask & pred_eddy_mask).sum().float()
    union = (true_eddy_mask | pred_eddy_mask).sum().float()
    results['eddy_iou'] = (intersection / union).item() if union > 0 else 0.0

    # 7) Eddy detection rate — fraction of true eddies overlapping a pred eddy
    if len(eddies_true) > 0:
        detected = 0
        center_dists = []
        for et in eddies_true:
            et_mask = et.mask.to(device)
            overlap = (et_mask & pred_eddy_mask).sum().float()
            if overlap / et.area_pixels >= 0.5:
                detected += 1
            # Find closest predicted eddy centre
            if len(eddies_pred) > 0:
                dists = [
                    ((et.center_y - ep.center_y) ** 2
                     + (et.center_x - ep.center_x) ** 2) ** 0.5
                    for ep in eddies_pred
                ]
                center_dists.append(min(dists))
        results['eddy_detection_rate'] = detected / len(eddies_true)
        results['eddy_center_distance'] = (
            np.mean(center_dists) if center_dists else float('nan')
        )
    else:
        results['eddy_detection_rate'] = float('nan')
        results['eddy_center_distance'] = float('nan')

    # 8) Velocity quality inside true-eddy regions
    if true_eddy_mask.sum() > 0:
        eddy_mask_2ch = true_eddy_mask.unsqueeze(0).expand(2, -1, -1)
        vel_diff = pred - true
        results['eddy_velocity_mse'] = (
            (vel_diff[eddy_mask_2ch] ** 2).mean().item()
        )

        # Angular error inside eddies
        u_p, v_p = pred[0], pred[1]
        u_t, v_t = true[0], true[1]
        dot = u_p * u_t + v_p * v_t
        mag_p = (u_p ** 2 + v_p ** 2).sqrt() + 1e-8
        mag_t = (u_t ** 2 + v_t ** 2).sqrt() + 1e-8
        cos_a = (dot / (mag_p * mag_t)).clamp(-1, 1)
        angle = torch.acos(cos_a) * (180.0 / np.pi)
        results['eddy_angular_error'] = angle[true_eddy_mask].mean().item()

        # Vorticity MSE inside eddies
        vor_diff = omega_pred - omega_true
        results['eddy_vorticity_mse'] = (
            (vor_diff[true_eddy_mask] ** 2).mean().item()
        )
    else:
        results['eddy_velocity_mse'] = float('nan')
        results['eddy_angular_error'] = float('nan')
        results['eddy_vorticity_mse'] = float('nan')

    # 9) Non-eddy velocity MSE (for contrast — how much worse are eddies?)
    non_eddy = ocean & ~true_eddy_mask
    if non_eddy.sum() > 0:
        ne_2ch = non_eddy.unsqueeze(0).expand(2, -1, -1)
        results['non_eddy_velocity_mse'] = (
            ((pred - true)[ne_2ch] ** 2).mean().item()
        )
    else:
        results['non_eddy_velocity_mse'] = float('nan')

    return results


# ---------------------------------------------------------------------------
# Convenience: batch version
# ---------------------------------------------------------------------------

def batch_eddy_metrics(
    preds: torch.Tensor,
    trues: torch.Tensor,
    **kwargs,
) -> dict:
    """
    Compute eddy metrics over a batch, returning means and stds.

    Parameters
    ----------
    preds, trues : (B, 2, H, W)
    **kwargs : forwarded to eddy_metrics

    Returns
    -------
    dict  like {metric_name: value, metric_name_std: value}
    """
    B = preds.shape[0]
    all_metrics = [
        eddy_metrics(preds[i], trues[i], **kwargs) for i in range(B)
    ]
    keys = all_metrics[0].keys()
    agg = {}
    for k in keys:
        vals = [m[k] for m in all_metrics if not np.isnan(m[k])]
        if vals:
            agg[k] = float(np.mean(vals))
            agg[f'{k}_std'] = float(np.std(vals))
        else:
            agg[k] = float('nan')
            agg[f'{k}_std'] = float('nan')
    return agg


# ---------------------------------------------------------------------------
# Pretty-print helper
# ---------------------------------------------------------------------------

def print_eddy_metrics(metrics: dict, title: str = "Eddy Metrics") -> None:
    """Pretty-print eddy evaluation results."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")

    sections = [
        ("OW Field", ['ow_mse', 'ow_mae', 'ow_correlation']),
        ("Vorticity Field", ['vort_mse', 'vort_mae', 'vort_correlation']),
        ("Eddy Detection", ['n_eddies_true', 'n_eddies_pred',
                            'eddy_iou', 'eddy_detection_rate',
                            'eddy_center_distance']),
        ("Eddy-Region Quality", ['eddy_velocity_mse', 'eddy_angular_error',
                                  'eddy_vorticity_mse']),
        ("Non-Eddy Contrast", ['non_eddy_velocity_mse']),
    ]

    for section_name, keys in sections:
        print(f"\n  {section_name}")
        print(f"  {'-' * 40}")
        for k in keys:
            v = metrics.get(k, float('nan'))
            std_k = f'{k}_std'
            if std_k in metrics:
                print(f"    {k:30s}  {v:10.6f}  (±{metrics[std_k]:.6f})")
            elif isinstance(v, float):
                print(f"    {k:30s}  {v:10.6f}")
            else:
                print(f"    {k:30s}  {v}")

    print(f"\n  OW threshold used: {metrics.get('ow_threshold', '?')}")
    print(f"{'=' * 60}\n")
