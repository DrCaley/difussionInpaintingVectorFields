import torch
import torch.nn.functional as f

from data_prep.data_initializer import DDInitializer
dd = DDInitializer()

def _iterative_projection_with_snap(x, mask, known_pixels, max_iters=20, tol=1e-5):
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

    # Initial snap to the observed values.
    x = known_pixels * (1 - mask) + x * mask

    for _ in range(max_iters):
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

        diff = (x - x_old) * known_mask
        rel_known = torch.norm(diff) / (torch.norm(x_old * known_mask) + 1e-8)
        if torch.max(rel_known) < tol:
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
    return x_final


def inpaint_generate_new_images(ddpm, input_image, mask, n_samples=16, device=None,
                                resample_steps=1, channels=2, height=64, width=128, noise_strategy = dd.get_noise_strategy()):
    """
    Given a DDPM model, an input image, and a mask, generates in-painted samples.
    """
    import sys
    print("[DEBUG] inpaint_generate_new_images: Starting", file=sys.stderr)
    noised_images = [None] * (ddpm.n_steps + 1)
    device = dd.get_device()
    print(f"[DEBUG] Device: {device}, n_steps: {ddpm.n_steps}", file=sys.stderr)

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
        print(f"[DEBUG] Starting forward noising loop ({ddpm.n_steps} steps)", file=sys.stderr)
        noised_images[0] = input_img
        for t in range(ddpm.n_steps):
            if t % 10 == 0:
                print(f"[DEBUG] Forward noising step {t}/{ddpm.n_steps}", file=sys.stderr)
            noised_images[t + 1] = noise_one_step(noised_images[t], t, noise_strat)
        
        print(f"[DEBUG] Forward noising complete", file=sys.stderr)

        doing_the_thing = False # Usually false

        if doing_the_thing:
            print(f"[DEBUG] Creating noise for step {ddpm.n_steps}", file=sys.stderr)
            noise = noise_strat(input_img, torch.tensor([ddpm.n_steps] , device=device))
            x = noised_images[ddpm.n_steps] * (1 - mask) + (noise * mask)
        else:
            print(f"[DEBUG] Starting masked_poisson_projection on heavily noised image", file=sys.stderr)
            x = masked_poisson_projection(noised_images[ddpm.n_steps], mask)
            print(f"[DEBUG] Masked poisson projection complete", file=sys.stderr)
        final_noised_image = x

        print(f"[DEBUG] Starting reverse denoising loop ({ddpm.n_steps} steps, resample_steps={resample_steps})", file=sys.stderr)
        for idx, t in enumerate(range(ddpm.n_steps - 1, -1, -1)):
            if t % 10 == 0:
                print(f"[DEBUG] Reverse denoise step {ddpm.n_steps-1-idx}/{ddpm.n_steps}", file=sys.stderr)
            for i in range(resample_steps):
                x = denoise_one_step(x, noise_strat, t) # temp used to be noise but wasn't be used at all
                
                x = noised_images[t] * (1 - mask) + (x * mask) # divergence caused by this snapping
                
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
                x = _iterative_projection_with_snap(x, mask, noised_images[t],
                                                   max_iters=MAX_ITERS, tol=tol)
            
                if (i + 1) < resample_steps: # adds stochastic noise per denoise step  
                    x = noise_one_step(x, t, noise_strat)
            # Final hard snap at t=0 so observed pixels (including land) are exact.
            x = noised_images[0] * (1 - mask) + x * mask
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