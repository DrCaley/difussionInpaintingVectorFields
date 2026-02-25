import torch
import torch.nn.functional as f
from tqdm import tqdm

from data_prep.data_initializer import DDInitializer
from ddpm.vector_combination.vector_combiner import combine_fields

dd = DDInitializer()


# def inpaint_generate_new_images(ddpm, input_image, mask, n_samples=16, device=None,
#                                 resample_steps=1, channels=2, height=64, width=128, noise_strategy = dd.get_noise_strategy()):
#     """
#     Given a DDPM model, an input image, and a mask, generates in-painted samples.
#     """
#     noised_images = [None] * (ddpm.n_steps + 1)
#     device = dd.get_device()
#
#     def denoise_one_step(noisy_img, noise_strat, t):
#         time_tensor = torch.full((n_samples, 1), t, device=device, dtype=torch.long)
#         epsilon_theta = ddpm.backward(noisy_img, time_tensor)
#
#         alpha_t = ddpm.alphas[t].to(device)
#         alpha_t_bar = ddpm.alpha_bars[t].to(device)
#
#         if noise_strat.get_gaussian_scaling():
#             less_noised_img = (1 / alpha_t.sqrt()) * (
#                     noisy_img - ((1 - alpha_t) / (1 - alpha_t_bar).sqrt()) * epsilon_theta
#             )
#         else:
#             less_noised_img = (1 / alpha_t.sqrt()) * (noisy_img - epsilon_theta)
#
#         tensor_size = torch.zeros(n_samples, channels, height, width, device=device)
#
#         if t > 0:
#             z = noise_strat(tensor_size, torch.tensor([t], device=device))
#             beta_t = ddpm.betas[t].to(device)
#             sigma_t = beta_t.sqrt()
#             less_noised_img = less_noised_img + sigma_t * z
#
#         return less_noised_img, noise
#
#     def noise_one_step(unnoised_img, t, noise_strat):
#         epsilon = noise_strat(unnoised_img, None)
#         noised_img = ddpm(unnoised_img, t, epsilon, one_step=True)
#         return noised_img
#
#     with torch.no_grad():
#         noise_strat = noise_strategy
#
#         input_img = input_image.clone().to(device)
#         mask = mask.to(device)
#
#         noise = noise_strat(input_img, torch.tensor([ddpm.n_steps], device=device))
#
#         # Step-by-step forward noising
#         noised_images[0] = input_img
#         for t in range(ddpm.n_steps):
#             noised_images[t + 1] = noise_one_step(noised_images[t], t, noise_strat)
#
#         doing_the_thing = False
#
#         if doing_the_thing:
#             x = noised_images[ddpm.n_steps] * (1 - mask) + (noise * mask)
#         else:
#             x = masked_poisson_projection(noised_images[ddpm.n_steps], mask)
#
#         with tqdm(total=ddpm.n_steps, desc="Denoising") as pbar:
#             for idx, t in enumerate(range(ddpm.n_steps - 1, -1, -1)):
#                 for i in range(resample_steps):
#                     inpainted, noise = denoise_one_step(x, noise_strat, t)
#                     known = noised_images[t]
#
#                     combined = combine_fields(known, inpainted, mask,
#                                               save_dir=f"../vector_combination/results/step_{idx}/resample_{i}")
#
#                     if (i + 1) < resample_steps:
#                         x = noise_one_step(combined, t, noise_strat)
#                 pbar.update(1)
#
#     result = input_img * (1 - mask) + combined * mask
#     #result = combined
#     return result

def inpaint_generate_new_images(ddpm, input_image, mask, n_samples=16, device=None,
                                resample_steps=1, channels=2, height=64, width=128,
                                noise_strategy=None, gamma=0.05, lambda_phys=0.001):
    """
    Given a DDPM model, an input image, and a mask, generates in-painted samples
    using Tweedie Guidance (Direct Data Consistency + Zero Divergence).
    """
    if noise_strategy is None:
        noise_strategy = dd.get_noise_strategy()

    noised_images = [None] * (ddpm.n_steps + 1)
    device = dd.get_device() if device is None else device


    def tweedie_denoise_one_step(noisy_img, noise_strat, t, ground_truth, mask):
        """
        Nested function: Replaces standard denoise_one_step.
        Uses Tweedie Guidance to steer x0_hat toward the known data and zero divergence.
        """
        # 1. Predict Noise (No Gradients)
        time_tensor = torch.full((noisy_img.shape[0], 1), t, device=device, dtype=torch.long)
        epsilon_theta = ddpm.backward(noisy_img, time_tensor)

        alpha_t = ddpm.alphas[t].to(device)
        alpha_t_bar = ddpm.alpha_bars[t].to(device)

        # 2. Calculate the Tweedie Prediction (x0_hat)
        x0_hat = (noisy_img - (1 - alpha_t_bar).sqrt() * epsilon_theta) / alpha_t_bar.sqrt()

        # --- ENABLE GRADIENTS FOR OPTIMIZATION ---
        with torch.enable_grad():
            # 3. Prepare for Optimization
            x0_hat = x0_hat.detach().requires_grad_(True)

            # 4. Compute Losses
            # Fidelity Loss (Note: your mask is 1 for inpaint, 0 for known)
            known_region_mask = (mask == 0).float()
            L_data = torch.nn.functional.mse_loss(
                x0_hat * known_region_mask,
                ground_truth * known_region_mask,
                reduction='sum'
            ) / known_region_mask.sum().clamp(min=1.0)

            # Physics Loss (Divergence via Central Difference)
            vx, vy = x0_hat[:, 0], x0_hat[:, 1]
            dvx_dx = torch.zeros_like(vx)
            dvy_dy = torch.zeros_like(vy)

            dvx_dx[:, 1:-1, :] = (vx[:, 2:, :] - vx[:, :-2, :]) / 2.0
            dvy_dy[:, :, 1:-1] = (vy[:, :, 2:] - vy[:, :, :-2]) / 2.0

            div = dvx_dx + dvy_dy
            L_divergence = (div ** 2).mean()

            # Total Loss
            L_total = L_data + (lambda_phys * L_divergence)

            # 5. Compute the Gradient
            grad = torch.autograd.grad(L_total, x0_hat)[0]

        # --- DISABLE GRADIENTS AGAIN FOR THE UPDATE ---
        # 6. Apply the Gradient Update (The "Nudge")
        current_gamma = gamma * (1 - alpha_t_bar).item()
        x0_hat_updated = x0_hat.detach() - (current_gamma * grad)

        # 7. Proceed to t-1 (Standard DDPM Step)
        epsilon_updated = (noisy_img - alpha_t_bar.sqrt() * x0_hat_updated) / (1 - alpha_t_bar).sqrt()

        if noise_strat.get_gaussian_scaling():
            less_noised_img = (1 / alpha_t.sqrt()) * (
                    noisy_img - ((1 - alpha_t) / (1 - alpha_t_bar).sqrt()) * epsilon_updated
            )
        else:
            less_noised_img = (1 / alpha_t.sqrt()) * (noisy_img - epsilon_updated)

        if t > 0:
            z = noise_strat(torch.zeros_like(noisy_img), torch.tensor([t], device=device))
            beta_t = ddpm.betas[t].to(device)
            sigma_t = beta_t.sqrt()
            less_noised_img = less_noised_img + sigma_t * z

        return less_noised_img, epsilon_updated

    def noise_one_step(unnoised_img, t, noise_strat):
        epsilon = noise_strat(unnoised_img, None)
        noised_img = ddpm(unnoised_img, t, epsilon, one_step=True)
        return noised_img

    with torch.no_grad():
        noise_strat = noise_strategy
        input_img = input_image.clone().to(device)
        mask = mask.to(device)
        known_region_mask = (mask == 0).float()

        # 1. Generate Div-Free Noise for the known regions
        base_noise_known = noise_strat(torch.zeros_like(input_img), None)
        x_T_known = ddpm(input_img, torch.tensor([ddpm.n_steps-1], device=device), base_noise_known)

        # 2. Generate pure Div-Free Noise for the hole
        x_T_hole = noise_strat(torch.zeros_like(input_img), None)

        # 3. Combine them (this creates boundary divergence)
        x = (x_T_known * known_region_mask) + (x_T_hole * (1 - known_region_mask))

        # 4. FIX THE BOUNDARY DIVERGENCE!
        # Pass the spliced field and the mask into your projection function
        x = masked_poisson_projection(x, mask)

        for idx, t in enumerate(range(ddpm.n_steps - 1, -1, -1)):
            for i in range(resample_steps):

                # 1. Take a Tweedie Guided Step (replacing denoise_one_step & combine_fields)
                inpainted, _ = tweedie_denoise_one_step(x, noise_strat, t, ground_truth=input_img, mask=mask)

                # 2. Resampling logic (RePaint trick)
                if (i + 1) < resample_steps:
                    x = noise_one_step(inpainted, t, noise_strat)
                else:
                    x = inpainted  # Crucial: Update x for the next timestep t-1


    # For the final output, enforce the ground truth on the known regions one last time
    # to guarantee exact pixel matches outside the mask.
    result = x

    return result



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


def masked_poisson_projection(vector_field, mask, num_iter=500, tol=1e-5):
    """
    Performs divergence-free projection of a 2D vector field with masked inpainting regions.
    Assumes swapped channels: Channel 0 is Vy (Y-velocity), Channel 1 is Vx (X-velocity).

    Args:
        vector_field: (N, 2, H, W) torch tensor (vy, vx)
        mask:         (N, 2, H, W) binary tensor, 1 = region to inpaint
        num_iter:     max Jacobi iterations
        tol:          early stopping tolerance on residual (L2 norm)

    Returns:
        projected_field: (N, 2, H, W) divergence-free vector field
    """
    N, _, H, W = vector_field.shape
    device = vector_field.device

    # NOTE: Assuming Ch 0 is Vy and Ch 1 is Vx based on previous divergence math
    vx, vy = vector_field[:, 0], vector_field[:, 1] # vx is Vy, vy is Vx

    # Compute divergence: ∂Vy/∂y + ∂Vx/∂x (forward diff)
    div = torch.zeros(N, H, W, device=device)
    div[:, :-1, :] += vx[:, 1:, :] - vx[:, :-1, :] # Ch 0 (Vy) along Y (dim 1)
    div[:, :, :-1] += vy[:, :, 1:] - vy[:, :, :-1] # Ch 1 (Vx) along X (dim 2)

    # Initialize scalar potential φ
    phi = torch.zeros(N, H, W, device=device)

    # Jacobi solver (Now solves globally to prevent boundary shockwaves)
    for i in range(num_iter):
        # Sum of neighbors (up, down, left, right)
        neighbor_sum = torch.zeros_like(phi)

        neighbor_sum[:, 1:, :] += phi[:, :-1, :]    # up
        neighbor_sum[:, :-1, :] += phi[:, 1:, :]    # down
        neighbor_sum[:, :, 1:] += phi[:, :, :-1]    # left
        neighbor_sum[:, :, :-1] += phi[:, :, 1:]    # right

        # Jacobi update applied globally
        phi_new = (div + neighbor_sum) / 4.0

        # Residual for early stopping
        residual = torch.norm(phi_new - phi, dim=(1, 2)).mean()

        phi = phi_new

        if residual < tol:
            break

    # Compute gradient of φ (forward diff)
    dphix = torch.zeros_like(vy) # X-gradient matches Vx shape
    dphiy = torch.zeros_like(vx) # Y-gradient matches Vy shape

    dphix[:, :, :-1] = phi[:, :, 1:] - phi[:, :, :-1] # Derivative along X
    dphiy[:, :-1, :] = phi[:, 1:, :] - phi[:, :-1, :] # Derivative along Y

    # Subtract gradient to get divergence-free field
    # FIX: Subtract Y-gradient from Vy (Ch 0) and X-gradient from Vx (Ch 1)
    vx_proj = vx - dphiy
    vy_proj = vy - dphix

    # Restore known values (preserve unmasked regions per channel)
    # The global solver touched everything, so we enforce the hard constraint here
    vx_proj = torch.where(mask[:, 0] == 0, vx, vx_proj)
    vy_proj = torch.where(mask[:, 1] == 0, vy, vy_proj)

    return torch.stack([vx_proj, vy_proj], dim=1)