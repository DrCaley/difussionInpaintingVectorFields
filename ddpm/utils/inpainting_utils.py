import torch
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from data_prep.data_initializer import DDInitializer
from ddpm.helper_functions.compute_divergence import compute_divergence
dd = DDInitializer()


def inpaint_generate_new_images(ddpm, input_image, mask, n_samples=16, device=None,
                                resample_steps=1, channels=2, height=64, width=128, 
                                noise_strategy=None, return_final_noised=False,
                                save_final_noised_path=None):
    """
    Given a DDPM model, an input image, and a mask, generates in-painted samples
    using simple noise and denoise steps.
    """
    if noise_strategy is None:
        noise_strategy = dd.get_noise_strategy()
    
    if device is None:
        device = dd.get_device()

    noised_images = [None] * (ddpm.n_steps + 1)

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
        input_img = input_image.clone().to(device)
        mask = mask.to(device)

        # Step-by-step forward noising
        noised_images[0] = input_img
        for t in range(ddpm.n_steps):
            noised_images[t + 1] = noise_one_step(noised_images[t], t, noise_strategy)

        # Initialize with fully noised image
        x = noised_images[ddpm.n_steps]
        # Keep a reference to the fully noised vector field (before denoising)
        final_noised_vector_field = noised_images[ddpm.n_steps].detach().cpu()

        # Denoise steps
        for t in range(ddpm.n_steps - 1, -1, -1):
            for i in range(resample_steps):
                x = denoise_one_step(x, noise_strategy, t)
                x = noised_images[t] * (1 - mask) + (x * mask) # The snapping of known pixels
                
                if (i + 1) < resample_steps:
                    x = noise_one_step(x, t, noise_strategy)
            
            # Final hard snap at t=0 to ensure observed pixels are exact
            if t == 0:
                x = noised_images[0] * (1 - mask) + x * mask
                pass

        # Optionally save the fully noised image to disk
        if save_final_noised_path is not None:
            import os
            try:
                os.makedirs(os.path.dirname(save_final_noised_path), exist_ok=True)
            except Exception:
                pass

            # Save raw tensor as numpy
            try:
                np.save(save_final_noised_path + '.npy', final_noised_vector_field.cpu().numpy())
            except Exception:
                pass

            # Try to create visualizations if plotting utilities are available
            try:
                from plots.visualization_tools.plot_vector_field_tool import plot_vector_field, make_heatmap

                # Use first sample
                sample = final_noised_vector_field[0]
                if sample.shape[0] >= 2:
                    vx = sample[0]
                    vy = sample[1]
                    try:
                        plot_vector_field(vx, vy, file=save_final_noised_path + '_vectorfield.png')
                    except Exception:
                        pass

                # Save heatmaps for each channel
                for c in range(sample.shape[0]):
                    try:
                        make_heatmap(sample[c], save_path=save_final_noised_path + f'_ch{c}.png')
                    except Exception:
                        pass
            except Exception:
                # plotting tools not available or failed; ignore gracefully
                pass

        # Optionally return the fully noised image along with the final prediction
        if return_final_noised:
            return x, final_noised_vector_field

        return x

    # unreachable guard (kept for clarity) -- function returns earlier normally


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

    Args:
        original_image: (1, 2, H, W)
        predicted_image: (1, 2, H, W)
        mask: (1, 2, H, W)

    Returns:
        Scalar percent error
    """
    single_mask = mask[:, 0:1, :, :]  # shape (1, 1, H, W)

    percent_error = torch.abs((predicted_image - original_image) / original_image)  # (1, 2, H, W)
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
    """
    Calculate average pixel difference as a percentage.
    """
    avg_pixel_value = torch.sum(torch.abs(original_image * mask)) / mask.sum()
    avg_diff = torch.sum(torch.abs((predicted_image * mask) - (original_image * mask))) / mask.sum()
    return avg_diff * (100 / avg_pixel_value)


def _split_vector_field(vector_field: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Return the two components of a 2D vector field and its layout."""
    if vector_field.ndim < 3:
        raise ValueError("Vector field must have at least 3 dimensions.")

    if vector_field.shape[-3] == 2:
        return vector_field[..., 0, :, :], vector_field[..., 1, :, :], "channel_first"

    if vector_field.shape[-1] == 2:
        return vector_field[..., 0], vector_field[..., 1], "channel_last"

    raise ValueError("Expected a vector field with a 2-channel dimension.")


def total_divergence(vector_field: torch.Tensor, reduction: str = "sum") -> torch.Tensor:
    """
    Compute the divergence of a 2D vector field and reduce it to a scalar.

    Args:
        vector_field: Tensor shaped as (2, H, W), (B, 2, H, W), (H, W, 2), or (B, H, W, 2).
        reduction: One of "sum", "mean", "abs_sum", or "abs_mean".

    Returns:
        Reduced divergence value as a tensor.
    """
    u, v, _ = _split_vector_field(vector_field)

    if u.ndim == 2:
        divergence_map = compute_divergence(u, v)
    else:
        flat_u = u.reshape(-1, u.shape[-2], u.shape[-1])
        flat_v = v.reshape(-1, v.shape[-2], v.shape[-1])
        divergence_map = torch.stack(
            [compute_divergence(flat_u[i], flat_v[i]) for i in range(flat_u.shape[0])],
            dim=0,
        ).reshape(*u.shape)

    if reduction == "sum":
        return divergence_map.sum()
    if reduction == "mean":
        return divergence_map.mean()
    if reduction == "abs_sum":
        return divergence_map.abs().sum()
    if reduction == "abs_mean":
        return divergence_map.abs().mean()

    raise ValueError("reduction must be one of: sum, mean, abs_sum, abs_mean")


def poisson_projection(
    vector_field: torch.Tensor,
    return_potential: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Project a 2D vector field onto its divergence-free component with a Poisson solve.

    The projection assumes periodic boundary conditions and works for tensors shaped
    as (2, H, W), (B, 2, H, W), (H, W, 2), or (B, H, W, 2).
    """
    u, v, layout = _split_vector_field(vector_field)

    def project_single(single_u: torch.Tensor, single_v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        divergence = compute_divergence(single_u, single_v)

        height, width = divergence.shape
        row_freq = 2 * torch.pi * torch.fft.fftfreq(height, d=1.0, device=divergence.device)
        col_freq = 2 * torch.pi * torch.fft.fftfreq(width, d=1.0, device=divergence.device)
        row_freq = row_freq.reshape(height, 1)
        col_freq = col_freq.reshape(1, width)

        laplacian_symbol = row_freq.square() + col_freq.square()
        laplacian_symbol = laplacian_symbol.to(divergence.dtype)
        laplacian_symbol = laplacian_symbol.clone()
        laplacian_symbol[0, 0] = torch.tensor(float("inf"), device=divergence.device, dtype=divergence.dtype)

        divergence_hat = torch.fft.fftn(divergence)
        potential_hat = -divergence_hat / laplacian_symbol
        potential_hat = potential_hat.clone()
        potential_hat[0, 0] = 0

        potential = torch.fft.ifftn(potential_hat).real
        grad_row = torch.fft.ifftn(1j * row_freq.to(potential_hat.dtype) * potential_hat).real
        grad_col = torch.fft.ifftn(1j * col_freq.to(potential_hat.dtype) * potential_hat).real

        projected_u = single_u - grad_row
        projected_v = single_v - grad_col
        return projected_u, projected_v, potential

    if u.ndim == 2:
        projected_u, projected_v, potential = project_single(u, v)
    else:
        flat_u = u.reshape(-1, u.shape[-2], u.shape[-1])
        flat_v = v.reshape(-1, v.shape[-2], v.shape[-1])

        projected_u_list = []
        projected_v_list = []
        potential_list = []
        for index in range(flat_u.shape[0]):
            proj_u, proj_v, pot = project_single(flat_u[index], flat_v[index])
            projected_u_list.append(proj_u)
            projected_v_list.append(proj_v)
            potential_list.append(pot)

        projected_u = torch.stack(projected_u_list, dim=0).reshape(*u.shape)
        projected_v = torch.stack(projected_v_list, dim=0).reshape(*v.shape)
        potential = torch.stack(potential_list, dim=0).reshape(*u.shape)

    if layout == "channel_first":
        stack_dim = 0 if projected_u.ndim == 2 else 1
        projected_field = torch.stack([projected_u, projected_v], dim=stack_dim)
    else:
        projected_field = torch.stack([projected_u, projected_v], dim=-1)

    if return_potential:
        return projected_field, potential

    return projected_field




