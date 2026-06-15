import torch
import torch.nn.functional as f
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import datetime

from data_prep.data_initializer import DDInitializer
dd = DDInitializer()


def _projection_debug_enabled():
    return bool(dd.get_attribute("debug_projection"))

def _projection_debug_every():
    try:
        configured = dd.get_attribute("debug_projection_every")
        # Keep terminal noise low during denoising: log at most once per 100 steps.
        if configured is None:
            return 100
        return max(100, int(configured))
    except (ValueError, TypeError):
        return 100


def _projection_debug_save_midpoint_images():
    return bool(dd.get_attribute("debug_projection_save_midpoint_images"))


def _projection_debug_dir():
    configured = dd.get_attribute("debug_projection_dir")
    if configured:
        return Path(configured)
    return Path("results") / "debug_projection"


def _vector_field_to_display_np(field, sample_idx=0):
    standardizer = dd.get_standardizer()
    disp = standardizer.unstandardize(field[sample_idx].detach().cpu())
    return disp.numpy()


def _plot_quiver_panel(ax, u, v, mask_np, title, vector_scale=0.15):
    H, W = u.shape
    xx, yy = np.meshgrid(np.arange(W), np.arange(H))
    ax.quiver(xx, yy, u, v, scale=1.0 / vector_scale)
    ax.contour(mask_np, levels=[0.5], colors="red", linewidths=2.0)
    ax.set_title(title)
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(-0.5, H - 0.5)
    ax.margins(x=0, y=0)
    ax.set_aspect('equal', adjustable='box')


def _save_projection_iteration_trace(reference_field, before_projection_field, projection_iter_fields,
                                    denoise_t, resample_idx, mask, sample_idx=0):
    out_dir = _projection_debug_dir() / "projection_traces"
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_np = _vector_field_to_display_np(reference_field, sample_idx=sample_idx)
    before_np = _vector_field_to_display_np(before_projection_field, sample_idx=sample_idx)
    iter_np_list = [_vector_field_to_display_np(f, sample_idx=sample_idx) for f in projection_iter_fields]

    mask_np = mask[sample_idx, 0].detach().cpu().numpy()

    # Match plotting frame used elsewhere.
    crop_h = min(44, ref_np.shape[1])
    crop_w = min(94, ref_np.shape[2])

    mask_np = mask_np[:crop_h, :crop_w]
    ref_u, ref_v = ref_np[0][:crop_h, :crop_w], ref_np[1][:crop_h, :crop_w]
    bef_u, bef_v = before_np[0][:crop_h, :crop_w], before_np[1][:crop_h, :crop_w]

    n_cols = 2 + len(iter_np_list)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

    _plot_quiver_panel(axes[0], ref_u, ref_v, mask_np, f"reference_t{denoise_t}")
    _plot_quiver_panel(axes[1], bef_u, bef_v, mask_np, f"before_projection_t{denoise_t}")

    for idx, arr in enumerate(iter_np_list):
        u_i, v_i = arr[0][:crop_h, :crop_w], arr[1][:crop_h, :crop_w]
        _plot_quiver_panel(axes[2 + idx], u_i, v_i, mask_np, f"after_poisson_iter{idx + 1}")

    fig.suptitle(f"Projection Trace t={denoise_t} resample={resample_idx}")
    fig.tight_layout()

    path = out_dir / f"projection_trace_t{denoise_t:03d}_resample{resample_idx}_sample{sample_idx}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_projection_progress_plot(progress_rows, total_steps, resample_steps):
    if not progress_rows:
        return None

    out_dir = _projection_debug_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    denoise_t = [row["denoise_t"] for row in progress_rows]

    div_rms_before = [row["div_rms_before"] for row in progress_rows]
    div_rms_after = [row["div_rms_after"] for row in progress_rows]
    div_max_before = [row["div_max_before"] for row in progress_rows]
    div_max_after = [row["div_max_after"] for row in progress_rows]

    proj_delta_unknown = [row["proj_delta_unknown_rms"] for row in progress_rows]
    proj_delta_known = [row["proj_delta_known_rms"] for row in progress_rows]
    proj_delta_total = [row["proj_delta_total_rms"] for row in progress_rows]

    drift_unknown_before = [row["drift_unknown_before"] for row in progress_rows]
    drift_unknown_after = [row["drift_unknown_after"] for row in progress_rows]
    drift_known_before = [row["drift_known_before"] for row in progress_rows]
    drift_known_after = [row["drift_known_after"] for row in progress_rows]
    drift_total_before = [row["drift_total_before"] for row in progress_rows]
    drift_total_after = [row["drift_total_after"] for row in progress_rows]

    fig, axes = plt.subplots(3, 2, figsize=(15, 12), sharex=True)
    axes = axes.flatten()

    axes[0].plot(denoise_t, div_rms_before, label="before", color="#1f77b4")
    axes[0].plot(denoise_t, div_rms_after, label="after", color="#ff7f0e")
    axes[0].set_title("Divergence RMS")
    axes[0].set_ylabel("value")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(denoise_t, div_max_before, label="before", color="#2ca02c")
    axes[1].plot(denoise_t, div_max_after, label="after", color="#d62728")
    axes[1].set_title("Divergence Max Abs")
    axes[1].set_ylabel("value")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(denoise_t, proj_delta_unknown, label="unknown", color="#9467bd")
    axes[2].plot(denoise_t, proj_delta_known, label="known", color="#8c564b")
    axes[2].plot(denoise_t, proj_delta_total, label="total", color="#17becf")
    axes[2].set_title("Projection Delta RMS")
    axes[2].set_ylabel("value")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    axes[3].plot(denoise_t, drift_unknown_before, label="before", color="#bcbd22")
    axes[3].plot(denoise_t, drift_unknown_after, label="after", color="#7f7f7f")
    axes[3].set_title("Drift to Noised: Unknown RMS")
    axes[3].set_ylabel("value")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()

    axes[4].plot(denoise_t, drift_known_before, label="before", color="#1f77b4")
    axes[4].plot(denoise_t, drift_known_after, label="after", color="#ff9896")
    axes[4].set_title("Drift to Noised: Known RMS")
    axes[4].set_ylabel("value")
    axes[4].grid(True, alpha=0.3)
    axes[4].legend()

    axes[5].plot(denoise_t, drift_total_before, label="before", color="#2ca02c")
    axes[5].plot(denoise_t, drift_total_after, label="after", color="#d62728")
    axes[5].set_title("Drift to Noised: Total RMS")
    axes[5].set_ylabel("value")
    axes[5].grid(True, alpha=0.3)
    axes[5].legend()

    for ax in axes:
        ax.set_xlabel("denoise step t")
        ax.invert_xaxis()

    fig.suptitle(f"Projection Debug Progress (n_steps={total_steps}, resample={resample_steps})")
    fig.tight_layout()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    save_path = out_dir / f"projection_progress_{timestamp}.png"
    fig.savefig(save_path, dpi=180)
    plt.close(fig)
    return save_path


def _save_field_drift_comparison(
    noised_field,
    denoised_field,
    denoise_t,
    mask,
    sample_idx=0,
    before_projection_field=None,
    projection_iter_fields=None,
    resample_idx=None,
):
    """Save one per-step field drift PNG with a fixed 3-row layout.

    Row 1: reference at t.
    Row 2: before (after DDPM + snap) and projection iterations.
    Row 3: final field after iterative projection, ready for next denoise step.
    """
    out_dir = _projection_debug_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    
    noised_np = _vector_field_to_display_np(noised_field, sample_idx=sample_idx)
    denoised_np = _vector_field_to_display_np(denoised_field, sample_idx=sample_idx)
    mask_np = mask[sample_idx, 0].detach().cpu().numpy()
    
    u_noised, v_noised = noised_np[0], noised_np[1]
    u_denoised, v_denoised = denoised_np[0], denoised_np[1]

    # Match pt_predictions framing from saved tensors (top-left crop ~44x94).
    crop_h = min(44, u_noised.shape[0])
    crop_w = min(94, u_noised.shape[1])
    u_noised = u_noised[:crop_h, :crop_w]
    v_noised = v_noised[:crop_h, :crop_w]
    u_denoised = u_denoised[:crop_h, :crop_w]
    v_denoised = v_denoised[:crop_h, :crop_w]
    mask_np = mask_np[:crop_h, :crop_w]

    H, W = u_noised.shape
    xx, yy = np.meshgrid(np.arange(W), np.arange(H))
    
    extra_fields = []
    if before_projection_field is not None:
        before_np = _vector_field_to_display_np(before_projection_field, sample_idx=sample_idx)
        extra_fields.append((before_np[0][:crop_h, :crop_w], before_np[1][:crop_h, :crop_w], "before (after ddpm + snap)"))

    if projection_iter_fields:
        for idx, arr_field in enumerate(projection_iter_fields):
            arr_np = _vector_field_to_display_np(arr_field, sample_idx=sample_idx)
            extra_fields.append((arr_np[0][:crop_h, :crop_w], arr_np[1][:crop_h, :crop_w], f"after_poisson_projection_{idx + 1}"))

    n_cols = max(1, len(extra_fields))
    fig, axes = plt.subplots(3, n_cols, figsize=(6 * n_cols, 14), squeeze=False)

    for row in range(3):
        for col in range(n_cols):
            axes[row, col].set_axis_off()

    def _draw_panel(ax, u, v, title):
        ax.set_axis_on()
        ax.quiver(xx, yy, u, v, scale=1.0 / 0.15)
        ax.contour(mask_np, levels=[0.5], colors="red", linewidths=2.0)
        ax.set_title(title)
        ax.set_xlim(-0.5, W - 0.5)
        ax.set_ylim(-0.5, H - 0.5)
        ax.margins(x=0, y=0)
        ax.set_aspect('equal', adjustable='box')

    center_col = n_cols // 2
    _draw_panel(axes[0, center_col], u_noised, v_noised, f"reference_t{denoise_t}")

    if extra_fields:
        _draw_panel(axes[1, 0], extra_fields[0][0], extra_fields[0][1], f"{extra_fields[0][2]}_t{denoise_t}")
        for idx, (u_extra, v_extra, title) in enumerate(extra_fields[1:], start=1):
            _draw_panel(axes[1, idx], u_extra, v_extra, f"{title}_t{denoise_t}")

    _draw_panel(axes[2, center_col], u_denoised, v_denoised, f"ready_for_next_denoise_t{denoise_t}")

    if resample_idx is not None:
        fig.suptitle(f"Field Drift t={denoise_t} resample={resample_idx}")

    fig.tight_layout()
    
    save_path = out_dir / f"field_drift_t{denoise_t}_sample{sample_idx}.png"
    fig.savefig(save_path, dpi=180)
    plt.close(fig)
    return save_path


def _region_rms_delta(new_field, old_field, region_mask, eps=1e-8):
    diff = new_field - old_field
    diff_sq = diff[:, 0:1] ** 2 + diff[:, 1:2] ** 2
    denom = region_mask.sum(dim=(2, 3), keepdim=True) + eps
    rms = torch.sqrt((diff_sq * region_mask).sum(dim=(2, 3), keepdim=True) / denom)
    return float(rms.mean().item())


def _debug_projection_change(tag, before_field, after_field, mask):
    unknown = mask[:, 0:1]
    known = 1 - unknown
    diff = after_field - before_field
    diff_sq = diff[:, 0:1] ** 2 + diff[:, 1:2] ** 2
    max_delta = float(torch.sqrt(diff_sq + 1e-12).amax().item())
    total_rms = float(torch.sqrt(diff_sq.mean()).item())
    known_rms = _region_rms_delta(after_field, before_field, known)
    unknown_rms = _region_rms_delta(after_field, before_field, unknown)
    div_before = float(div_rms(before_field).mean().item())
    div_after = float(div_rms(after_field).mean().item())
    div_max_before = float(div_max_abs(before_field).mean().item())
    div_max_after = float(div_max_abs(after_field).mean().item())
    print(
        f"[DEBUG_PROJ] {tag} | total_rms={total_rms:.6e} "
        f"known_rms={known_rms:.6e} unknown_rms={unknown_rms:.6e} "
        f"max_delta={max_delta:.6e} div_rms_before={div_before:.6e} "
        f"div_rms_after={div_after:.6e} "
        f"div_max_abs_before={div_max_before:.6e} "
        f"div_max_abs_after={div_max_after:.6e}",
        file=sys.stderr,
    )


def _summarize_projection_change(before_field, after_field, mask):
    unknown = mask[:, 0:1]
    known = 1 - unknown
    return {
        "total_rms": float(torch.sqrt(((after_field - before_field) ** 2).mean()).item()),
        "known_rms": _region_rms_delta(after_field, before_field, known),
        "unknown_rms": _region_rms_delta(after_field, before_field, unknown),
        "div_before": float(div_rms(before_field).mean().item()),
        "div_after": float(div_rms(after_field).mean().item()),
        "div_max_abs_before": float(div_max_abs(before_field).mean().item()),
        "div_max_abs_after": float(div_max_abs(after_field).mean().item()),
    }


def _summarize_field_drift(current_field, reference_field, mask):
    unknown = mask[:, 0:1]
    known = 1 - unknown
    return {
        "total_rms": float(torch.sqrt(((current_field - reference_field) ** 2).mean()).item()),
        "known_rms": _region_rms_delta(current_field, reference_field, known),
        "unknown_rms": _region_rms_delta(current_field, reference_field, unknown),
    }


def _save_vector_field_snapshot(before_field, after_field, mask, denoise_t, sample_idx=0, resample_idx=0):
    out_dir = _projection_debug_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    before_np = before_field[sample_idx].detach().cpu().numpy()
    after_np = after_field[sample_idx].detach().cpu().numpy()
    mask_np = mask[sample_idx, 0].detach().cpu().numpy()

    u_b, v_b = before_np[0], before_np[1]
    u_a, v_a = after_np[0], after_np[1]
    mag_b = np.sqrt(u_b ** 2 + v_b ** 2)
    mag_a = np.sqrt(u_a ** 2 + v_a ** 2)
    step = max(1, min(u_b.shape) // 24)
    H, W = u_b.shape
    xx, yy = np.meshgrid(np.arange(0, W, step), np.arange(0, H, step))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    im0 = axes[0].imshow(mag_b, cmap="viridis", origin="upper")
    axes[0].quiver(
        xx,
        yy,
        u_b[::step, ::step],
        v_b[::step, ::step],
        color="white",
        alpha=0.75,
        scale=None,
    )
    axes[0].contour(mask_np, levels=[0.5], colors="red", linewidths=2.0)
    axes[0].set_title(f"Before Iterative Projection (t={denoise_t})", fontweight='bold')
    axes[0].set_xlabel("Width (columns)")
    axes[0].set_ylabel("Height (rows)")
    axes[0].set_xlim(-0.5, W - 0.5)
    axes[0].set_ylim(H - 0.5, -0.5)
    axes[0].margins(x=0, y=0)
    axes[0].set_aspect('equal', adjustable='box')
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="Magnitude")

    im1 = axes[1].imshow(mag_a, cmap="viridis", origin="upper")
    axes[1].quiver(
        xx,
        yy,
        u_a[::step, ::step],
        v_a[::step, ::step],
        color="white",
        alpha=0.75,
        scale=None,
    )
    axes[1].contour(mask_np, levels=[0.5], colors="red", linewidths=2.0)
    axes[1].set_title(f"After Iterative Projection (t={denoise_t})", fontweight='bold')
    axes[1].set_xlabel("Width (columns)")
    axes[1].set_ylabel("Height (rows)")
    axes[1].set_xlim(-0.5, W - 0.5)
    axes[1].set_ylim(H - 0.5, -0.5)
    axes[1].margins(x=0, y=0)
    axes[1].set_aspect('equal', adjustable='box')
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="Magnitude")

    fig.suptitle("Halfway Denoise Snapshot: Projection Impact")
    fig.tight_layout()

    save_path = out_dir / f"halfway_t{denoise_t}_sample{sample_idx}_resample{resample_idx}.png"
    fig.savefig(save_path, dpi=180)
    plt.close(fig)
    print(f"[DEBUG_PROJ] saved_halfway_snapshot={save_path}", file=sys.stderr)

def _iterative_projection_with_snap(x, mask, known_pixels, max_iters=20, tol=1e-5, debug=False):
    """Perform repeated divergence‑free projection with snapping until known region stabilizes.

    The procedure mirrors the user's description: after a single denoise step we
    snap the known pixels (which destroys divergence) then project back to a
    divergence‑free field.  The projection is applied repeatedly until the values
    in the known region stop changing by more than ``tol``.  To prevent gradual
    shrinking of the field magnitude we renormalize the **unknown** (masked)
    region after each projection so that its RMS magnitude stays close to the
    pre‑projection value (clamped to avoid large jumps).

    Args:
        x:            current field (N,2,H,W);
        mask:         binary mask (N,2,H,W), 1=unknown region to update;
        known_pixels: original field at time step t (N,2,H,W) used for snapping;
        max_iters:    upper bound on projection iterations;
        tol:          relative tolerance on known‑region change.

    Returns:
        x after stabilization (divergence free within mask, known region unchanged).
    """
    known_mask = 1 - mask[:, 0:1]
    debug_proj = debug
    x_before_snap = x.clone()

    # Initial snap to the observed values.
    x = known_pixels * (1 - mask) + x * mask
    if debug_proj:
        _debug_projection_change("iterative_snap:init_snap", x_before_snap, x, mask)

    projection_iter_fields = []

    for iter_idx in range(max_iters):
        x_old = x

        # Global projection gives the cleanest divergence-free field.
        x_proj = global_poisson_projection_consistent(x)

        # Use both RMS and max statistics from the unknown region to stop
        # projection from inflating vector magnitudes.
        pre_rms = rms_magnitude(x, mask)
        post_rms = rms_magnitude(x_proj, mask)
        pre_max = max_magnitude(x, mask)
        post_max = max_magnitude(x_proj, mask)

        rms_scale = pre_rms / (post_rms + 1e-8)
        max_scale = pre_max / (post_max + 1e-8)
        s = torch.minimum(rms_scale, max_scale)
        s = torch.clamp(s, 0.85, 1.0).view(-1, 1, 1, 1)

        # Global scaling preserves divergence-free structure. Snapping known
        # pixels afterwards re-imposes the observation constraint.
        x_proj = x_proj * s
        x = known_pixels * (1 - mask) + x_proj * mask
        projection_iter_fields.append(x.detach().clone())

        if debug_proj:
            _debug_projection_change(
                f"iterative_snap:iter={iter_idx}",
                x_old,
                x,
                mask,
            )

        diff = (x - x_old) * known_mask
        rel_known = torch.norm(diff) / (torch.norm(x_old * known_mask) + 1e-8)
        if torch.max(rel_known) < tol:
            if debug_proj:
                print(
                    f"[DEBUG_PROJ] iterative_snap:converged iter={iter_idx} rel_known={float(rel_known.item()):.6e}",
                    file=sys.stderr,
                )
            break

    # Return a divergence-free field. One last global rescale keeps projection
    # from re-introducing large magnitudes.
    x_proj = global_poisson_projection_consistent(x)
    final_rms = rms_magnitude(x, mask)
    proj_rms = rms_magnitude(x_proj, mask)
    final_max = max_magnitude(x, mask)
    proj_max = max_magnitude(x_proj, mask)
    s = torch.minimum(final_rms / (proj_rms + 1e-8), final_max / (proj_max + 1e-8))
    s = torch.clamp(s, 0.85, 1.0).view(-1, 1, 1, 1)
    x_final = known_pixels * (1 - mask) + (x_proj * s) * mask
    if debug_proj:
        _debug_projection_change("iterative_snap:final", x, x_final, mask)
    return x_final, projection_iter_fields


def inpaint_generate_new_images(ddpm, input_image, mask, n_samples=16, device=None,
                                resample_steps=1, channels=2, height=64, width=128, noise_strategy = dd.get_noise_strategy()):
    """
    Given a DDPM model, an input image, and a mask, generates in-painted samples.
    """
    debug_proj = _projection_debug_enabled()
    debug_every = _projection_debug_every()
    save_midpoint_images = debug_proj and _projection_debug_save_midpoint_images()
    projection_progress_rows = []
    midpoint_saved = False
    halfway_idx = ddpm.n_steps // 2
    noised_images = [None] * (ddpm.n_steps + 1)
    device = dd.get_device()
    if debug_proj:
        print(
            f"[DEBUG_PROJ] enabled=True n_steps={ddpm.n_steps} resample_steps={resample_steps} log_every={debug_every}",
            file=sys.stderr,
        )

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
            # One reverse transition uses one stochastic draw; keep it divergence-free
            # while avoiding heavy multi-layer generation tied to large t.
            step_t = torch.ones((n_samples,), device=device, dtype=torch.long)
            z = noise_strat(tensor_size, step_t)
            beta_t = ddpm.betas[t].to(device)
            sigma_t = beta_t.sqrt()
            less_noised_img = less_noised_img + sigma_t * z

        return less_noised_img

    def noise_one_step(unnoised_img, t, noise_strat):
        batch_n = unnoised_img.shape[0]
        step_t = torch.ones((batch_n,), device=unnoised_img.device, dtype=torch.long)
        epsilon = noise_strat(unnoised_img, step_t)
        noised_img = ddpm(unnoised_img, t, epsilon, one_step=True)
        return noised_img

    with torch.no_grad():
        noise_strat = noise_strategy

        input_img = input_image.clone().to(device)
        mask = mask.to(device)
  
        noise = None

        # Step-by-step forward noising
        noised_images[0] = input_img
        for t in range(ddpm.n_steps):
            noised_images[t + 1] = noise_one_step(noised_images[t], t, noise_strat)

        doing_the_thing = False # Usually false

        if doing_the_thing:
            noise = noise_strat(input_img, torch.tensor([ddpm.n_steps] , device=device))
            x = noised_images[ddpm.n_steps] * (1 - mask) + (noise * mask)
        else:
            x = masked_poisson_projection(noised_images[ddpm.n_steps], mask)
        final_noised_image = x

        for idx, t in enumerate(range(ddpm.n_steps - 1, -1, -1)):
            for i in range(resample_steps):
                x = denoise_one_step(x, noise_strat, t) # temp used to be noise but wasn't be used at all

                x = noised_images[t] * (1 - mask) + (x * mask) # divergence caused by this snapping
                should_log_this_step = debug_proj and (t % debug_every == 0)
                
                """
                x_original = x 
                x_prev = x
                x = global_poisson_projection(x)
                
                change = known_region_change(x, x_original, mask)
                iterations = 0
                MAX_ITERATIONS = 5
                
                while (change.mean() > 1e-4 and iterations < MAX_ITERATIONS) :
                    scale_prev = rms_magnitude(x_prev, mask)
                    scale_cur = rms_magnitude(x, mask)
                    x = x * (scale_prev/scale_cur)
                    
                    x = global_poisson_projection(x)
                    change = known_region_change(x, x_original, mask)
                    x = noised_images[t] * (1 - mask) + (x * mask)
                    x_prev = x
                    
                    iterations += 1
                """
                
                # After each denoising step we must snap the known pixels and then
                # project back to a divergence-free field repeatedly until the
                # snapping no longer alters the known region.  The helper below
                # handles the iterative projection and rescales the unknown region
                # to prevent gradual shrinkage.
                # Reduced from 20 to 5 for faster MPS execution
                MAX_ITERS = 5
                tol = 1e-4
                x_after_snap = x
                x, projection_iter_fields = _iterative_projection_with_snap(
                    x,
                    mask,
                    noised_images[t],
                    max_iters=MAX_ITERS,
                    tol=tol,
                    debug=False,
                )
                if debug_proj:
                    stats = _summarize_projection_change(x_after_snap, x, mask)
                    pre_drift = _summarize_field_drift(x_after_snap, noised_images[t], mask)
                    post_drift = _summarize_field_drift(x, noised_images[t], mask)

                    projection_progress_rows.append({
                        "denoise_t": int(t),
                        "div_rms_before": float(stats["div_before"]),
                        "div_rms_after": float(stats["div_after"]),
                        "div_max_before": float(stats["div_max_abs_before"]),
                        "div_max_after": float(stats["div_max_abs_after"]),
                        "proj_delta_unknown_rms": float(stats["unknown_rms"]),
                        "proj_delta_known_rms": float(stats["known_rms"]),
                        "proj_delta_total_rms": float(stats["total_rms"]),
                        "drift_unknown_before": float(pre_drift["unknown_rms"]),
                        "drift_unknown_after": float(post_drift["unknown_rms"]),
                        "drift_known_before": float(pre_drift["known_rms"]),
                        "drift_known_after": float(post_drift["known_rms"]),
                        "drift_total_before": float(pre_drift["total_rms"]),
                        "drift_total_after": float(post_drift["total_rms"]),
                    })
                    
                    # Save one consolidated field drift figure per denoise timestamp.
                    if debug_proj and (i == resample_steps - 1) and (t != 0):
                        _save_field_drift_comparison(
                            noised_images[t],
                            x,
                            t,
                            mask,
                            sample_idx=0,
                            before_projection_field=x_after_snap,
                            projection_iter_fields=projection_iter_fields,
                            resample_idx=i,
                        )

                if should_log_this_step:
                    print(
                        f"[DEBUG_PROJ] denoise_t={t} resample={i} "
                        f"div_rms_before={stats['div_before']:.6e} "
                        f"div_rms_after={stats['div_after']:.6e} "
                        f"div_max_abs_before={stats['div_max_abs_before']:.6e} "
                        f"div_max_abs_after={stats['div_max_abs_after']:.6e} "
                        f"proj_delta_unknown_rms={stats['unknown_rms']:.6e} "
                        f"proj_delta_known_rms={stats['known_rms']:.6e} "
                        f"proj_delta_total_rms={stats['total_rms']:.6e} "
                        f"drift_to_noised_unknown_rms:{pre_drift['unknown_rms']:.6e}->{post_drift['unknown_rms']:.6e} "
                        f"drift_to_noised_known_rms:{pre_drift['known_rms']:.6e}->{post_drift['known_rms']:.6e} "
                        f"drift_to_noised_total_rms:{pre_drift['total_rms']:.6e}->{post_drift['total_rms']:.6e}",
                        file=sys.stderr,
                    )

                if save_midpoint_images and (not midpoint_saved) and idx == halfway_idx and i == 0:
                    _save_vector_field_snapshot(
                        x_after_snap,
                        x,
                        mask,
                        denoise_t=t,
                        sample_idx=0,
                        resample_idx=i,
                    )
                    midpoint_saved = True
            
                if (i + 1) < resample_steps: # adds stochastic noise per denoise step  
                    x = noise_one_step(x, t, noise_strat)
            # Final hard snap only at t=0 so observed pixels (including land) are exact.
            if t == 0:
                x = noised_images[0] * (1 - mask) + x * mask
                if debug_proj:
                    _save_field_drift_comparison(
                        noised_images[0],
                        x,
                        0,
                        mask,
                        sample_idx=0,
                        before_projection_field=x_after_snap,
                        projection_iter_fields=projection_iter_fields,
                        resample_idx=resample_steps - 1,
                    )

        if debug_proj and projection_progress_rows:
            progress_path = _save_projection_progress_plot(
                projection_progress_rows,
                total_steps=ddpm.n_steps,
                resample_steps=resample_steps,
            )
            if progress_path is not None:
                print(f"[DEBUG_PROJ] saved_progress_plot={progress_path}", file=sys.stderr)
    return x, noised_images[ddpm.n_steps]

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


import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F


def masked_poisson_projection(vector_field, mask, num_iter=50, tol=1e-5):
    """
    Performs divergence-free projection of a 2D vector field with masked inpainting regions.

    Args:
        vector_field: (N, 2, H, W) torch tensor (vx, vy)
        mask:         (N, 2, H, W) binary tensor, 1 = region to inpaint
        num_iter:     max Jacobi iterations (reduced from 500 to 50 for MPS speed)
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


# Henry new functions


def global_poisson_projection(vector_field, num_iter=25, tol=1e-4):
    """
    Global divergence-free projection of a 2D vector field.

    Args:
        vector_field: (N, 2, H, W) tensor (vx, vy)
        num_iter: max Jacobi iterations (reduced from 50 to 25 for MPS speed)
        tol: early stopping tolerance on residual (RMS update of phi)

    Returns:
        projected_field: (N, 2, H, W) approximately divergence-free
    """
    N, _, H, W = vector_field.shape
    device = vector_field.device

    vx, vy = vector_field[:, 0], vector_field[:, 1]

    # divergence: forward diff (same pattern you used)
    div = torch.zeros(N, H, W, device=device)
    div[:, :, :-1] += vx[:, :, 1:] - vx[:, :, :-1]
    div[:, :-1, :] += vy[:, 1:, :] - vy[:, :-1, :]

    # solve Laplacian(phi) = div  via Jacobi iterations
    phi = torch.zeros(N, H, W, device=device)

    for _ in range(num_iter):
        # neighbor sum (up, down, left, right)
        neighbor_sum = torch.zeros_like(phi)
        neighbor_sum[:, 1:, :]  += phi[:, :-1, :]   # up
        neighbor_sum[:, :-1, :] += phi[:, 1:, :]    # down
        neighbor_sum[:, :, 1:]  += phi[:, :, :-1]   # left
        neighbor_sum[:, :, :-1] += phi[:, :, 1:]    # right

        phi_new = (div + neighbor_sum) / 4.0

        # RMS update as residual
        diff = phi_new - phi
        residual = torch.sqrt((diff * diff).mean()).item()

        phi = phi_new

        if residual < tol:
            break

    # grad(phi): forward diff (same as you used)
    dphix = torch.zeros_like(vx)
    dphiy = torch.zeros_like(vy)
    dphix[:, :, :-1] = phi[:, :, 1:] - phi[:, :, :-1]
    dphiy[:, :-1, :] = phi[:, 1:, :] - phi[:, :-1, :]

    vx_proj = vx - dphix
    vy_proj = vy - dphiy

    return torch.stack([vx_proj, vy_proj], dim=1)

def vector_magnitude(field, eps=1e-12):
    # field: (N, 2, H, W)
    return torch.sqrt((field[:, 0]**2 + field[:, 1]**2) + eps)

def rms_magnitude(field, mask=None, eps=1e-12):
    mag_sq = field[:, 0]**2 + field[:, 1]**2

    if mask is not None:
        # use single-channel mask
        m = mask[:, 0]
        mag_sq = mag_sq * m
        denom = m.sum(dim=(1, 2)) + eps
    else:
        denom = torch.tensor(field.shape[2] * field.shape[3], device=field.device)

    mean_mag_sq = mag_sq.sum(dim=(1, 2)) / denom
    return torch.sqrt(mean_mag_sq + eps)   # (N,)

def max_magnitude(field, mask=None):
    mag = vector_magnitude(field)

    if mask is not None:
        m = mask[:, 0]
        mag = mag.masked_fill(m == 0, float("-inf"))

    return mag.amax(dim=(1, 2))   # (N,)

def known_region_change(field_a, field_b, mask, mode="rms", eps=1e-8):
    """
    Measures how much the KNOWN region changed between two vector fields.

    Args:
        field_a, field_b : (N, 2, H, W)
        mask             : (N, 2, H, W)  1 = unknown, 0 = known
        mode             : "rms", "mae", or "max"

    Returns:
        change_per_sample : (N,) tensor
    """

    # known region = mask == 0
    known = 1 - mask[:, 0:1]   # single-channel mask

    diff = field_a - field_b
    diff_sq = diff[:, 0:1]**2 + diff[:, 1:2]**2   # |u|^2 per pixel

    if mode == "rms":
        num = (diff_sq * known).sum(dim=(2, 3))
        denom = known.sum(dim=(2, 3)) + eps
        return torch.sqrt(num / denom).squeeze(1)

    elif mode == "mae":
        mag = torch.sqrt(diff_sq + eps)
        num = (mag * known).sum(dim=(2, 3))
        denom = known.sum(dim=(2, 3)) + eps
        return (num / denom).squeeze(1)

    elif mode == "max":
        mag = torch.sqrt(diff_sq + eps)
        mag = mag.masked_fill(known == 0, 0)
        return mag.amax(dim=(2, 3)).squeeze(1)

    else:
        raise ValueError("mode must be 'rms', 'mae', or 'max'")
    
    
def global_poisson_projection_consistent(v, num_iter=100, tol=1e-5):
    N, _, H, W = v.shape
    device = v.device
    vx, vy = v[:, 0], v[:, 1]

    # backward divergence
    div = torch.zeros(N, H, W, device=device)
    div[:, :, 1:] += vx[:, :, 1:] - vx[:, :, :-1]
    div[:, 1:, :] += vy[:, 1:, :] - vy[:, :-1, :]

    # IMPORTANT: solvability for Neumann-ish solve
    div = div - div.mean(dim=(1,2), keepdim=True)

    phi = torch.zeros(N, H, W, device=device)
    for iter_idx in range(num_iter):
        neighbor_sum = torch.zeros_like(phi)
        neighbor_sum[:, 1:, :]  += phi[:, :-1, :]
        neighbor_sum[:, :-1, :] += phi[:, 1:, :]
        neighbor_sum[:, :, 1:]  += phi[:, :, :-1]
        neighbor_sum[:, :, :-1] += phi[:, :, 1:]

        phi_new = (neighbor_sum - div) / 4.0

        # remove mean to prevent drift
        phi_new = phi_new - phi_new.mean(dim=(1,2), keepdim=True)

        res = torch.sqrt(((phi_new - phi) ** 2).mean()).item()
        phi = phi_new
        if res < tol:
            break

    # forward grad
    dphix = torch.zeros_like(vx)
    dphiy = torch.zeros_like(vy)
    dphix[:, :, :-1] = phi[:, :, 1:] - phi[:, :, :-1]
    dphiy[:, :-1, :] = phi[:, 1:, :] - phi[:, :-1, :]

    return torch.stack([vx - dphix, vy - dphiy], dim=1)

def div_backward(v):
    vx, vy = v[:, 0], v[:, 1]

    div = torch.zeros(v.shape[0], v.shape[2], v.shape[3], device=v.device)
    div[:, :, 1:] += vx[:, :, 1:] - vx[:, :, :-1]
    div[:, 1:, :] += vy[:, 1:, :] - vy[:, :-1, :]

    return div

def div_rms(v):
    d = div_backward(v)
    return torch.sqrt((d ** 2).mean(dim=(1, 2)))


def div_max_abs(v):
    d = div_backward(v)
    return d.abs().amax(dim=(1, 2))