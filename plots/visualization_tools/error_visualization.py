import torch
import matplotlib.pyplot as plt
import numpy as np
import os



DEFAULT_SAVE_DIR = "pt_visualizer_images"

def ensure_save_path(filename_or_path):
    if os.path.dirname(filename_or_path):  # path already includes directories
        return filename_or_path
    else:
        os.makedirs(DEFAULT_SAVE_DIR, exist_ok=True)
        return os.path.join(DEFAULT_SAVE_DIR, filename_or_path)


def save_mse_heatmap(tensor1, tensor2, mask, save_path="mse_heatmap.png",
                     title="Masked Pixel-wise MSE Heatmap", mask_color="lightgray",
                     crop_shape=(44, 94), cmap_name="viridis"):
    save_path = ensure_save_path(save_path)

    assert tensor1.shape == tensor2.shape, "Tensors must be the same shape"
    assert tensor1.shape[1] == 2, "Expected tensors with shape (1, 2, H, W)"
    assert mask.shape == (1, 2, tensor1.shape[2], tensor1.shape[3]), "Mask must be shape (1, 2, H, W)"

    single_mask = mask[:, 0:1, :, :]

    squared_error = (tensor1 - tensor2) ** 2
    pixel_mse = squared_error.sum(dim=1, keepdim=True)
    masked_mse = pixel_mse * single_mask

    masked_mse_np = masked_mse.squeeze().cpu().numpy()
    mask_np = single_mask.squeeze().cpu().numpy()

    crop_h, crop_w = crop_shape
    cropped_mse = masked_mse_np[:crop_h, :crop_w]
    cropped_mask = mask_np[:crop_h, :crop_w]

    valid_pixels = cropped_mask == 1
    avg_mse = cropped_mse[valid_pixels].mean() if np.any(valid_pixels) else float('nan')
    print(f"Average MSE per pixel over masked area in crop: {avg_mse:.6f}")

    masked_array = np.where(cropped_mask == 1, cropped_mse, np.nan)

    plt.figure(figsize=(8, 6))
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color=mask_color)

    plt.imshow(masked_array, cmap=cmap, interpolation='nearest')
    plt.colorbar(label="Pixel MSE")
    full_title = f"{title}\nAverage MSE: {avg_mse:.6f}"
    plt.title(full_title)
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def save_angular_error_heatmap(tensor1, tensor2, mask, save_path="angular_error_heatmap.png",
                               crop_shape=(44, 94), mask_color="lightgray", cmap_name="viridis",
                               title="Masked Angular Error Heatmap"):
    save_path = ensure_save_path(save_path)

    assert tensor1.shape == tensor2.shape, "Tensors must be the same shape"
    assert tensor1.shape[1] == 2, "Expected tensors with shape (1, 2, H, W)"
    assert mask.shape == (1, 2, tensor1.shape[2], tensor1.shape[3]), "Mask must be shape (1, 2, H, W)"

    single_mask = mask[:, 0:1, :, :]

    u_pred = tensor1[:, 0, :, :]
    v_pred = tensor1[:, 1, :, :]
    u_true = tensor2[:, 0, :, :]
    v_true = tensor2[:, 1, :, :]

    dot = u_pred * u_true + v_pred * v_true
    norm_pred = torch.sqrt(u_pred ** 2 + v_pred ** 2) + 1e-8
    norm_true = torch.sqrt(u_true ** 2 + v_true ** 2) + 1e-8

    cos_angle = dot / (norm_pred * norm_true)
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
    angle = torch.acos(cos_angle)
    angle_deg = angle * (180.0 / np.pi)

    angle_np = angle_deg.squeeze().cpu().numpy()
    mask_np = single_mask.squeeze().cpu().numpy()

    crop_h, crop_w = crop_shape
    cropped_angle = angle_np[:crop_h, :crop_w]
    cropped_mask = mask_np[:crop_h, :crop_w]

    masked_angle = np.where(cropped_mask == 1, cropped_angle, np.nan)
    avg_angle_error = cropped_angle[cropped_mask == 1].mean() if np.any(cropped_mask == 1) else float('nan')
    print(f"Average angular error (degrees) over masked crop: {avg_angle_error:.3f}")

    plt.figure(figsize=(8, 6))
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color=mask_color)

    plt.imshow(masked_angle, cmap=cmap, interpolation='nearest')
    plt.colorbar(label="Angular error (degrees)")
    full_title = f"{title}\nAverage Angular Error: {avg_angle_error:.3f}°"
    plt.title(full_title)
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def save_scaled_error_vectors_scalar_field(tensor1, tensor2, mask, save_path="scaled_errors_scalar_field_heatmap.png",
                                            crop_shape=(44, 94), mask_color="lightgray", cmap_name="viridis",
                                            title="Masked Pixel-wise MSE Heatmap",):
    save_path = ensure_save_path(save_path)
    assert tensor1.shape == tensor2.shape, "Tensors must be the same shape"
    assert tensor1.shape[1] == 2, "Expected tensors with shape (1, 2, H, W)"
    assert mask.shape == (1, 2, tensor1.shape[2], tensor1.shape[3]), "Mask must be shape (1, 2, H, W)"

    single_mask = mask[:, 0:1, :, :]

    u_pred = tensor1[:, 0, :, :]
    v_pred = tensor1[:, 1, :, :]
    u_true = tensor2[:, 0, :, :]
    v_true = tensor2[:, 1, :, :]

    # Compute the error vectors
    error_u = u_pred - u_true
    error_v = v_pred - v_true

    # Magnitude of the error vectors
    error_magnitude = torch.sqrt(error_u ** 2 + error_v ** 2)

    # Magnitude of the real vectors
    real_magnitude = torch.sqrt(u_true ** 2 + v_true ** 2) # + 1e-8 # avoid divide-by-zero

    # Scale error magnitude by real magnitude
    scaled_error_magnitude = error_magnitude / real_magnitude

    # Apply mask
    masked_scaled_error = scaled_error_magnitude * single_mask.squeeze(1)

    # Convert to numpy
    masked_scaled_error_np = masked_scaled_error.squeeze().cpu().numpy()
    mask_np = single_mask.squeeze().cpu().numpy()

    # Crop
    crop_h, crop_w = crop_shape
    cropped_error = masked_scaled_error_np[:crop_h, :crop_w]
    cropped_mask = mask_np[:crop_h, :crop_w]

    # Mask out invalid values
    masked_array = np.where(cropped_mask == 1, cropped_error, np.nan)

    # Compute average over valid pixels
    valid_pixels = cropped_mask == 1
    avg_scaled_error = cropped_error[valid_pixels].mean() if np.any(valid_pixels) else float('nan')
    print(f"Average scaled error magnitude over masked crop: {avg_scaled_error:.6f}")

    # Plot
    plt.figure(figsize=(8, 6))
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color=mask_color)

    plt.imshow(masked_array, cmap=cmap, interpolation='nearest')
    plt.colorbar(label="Scaled Error Magnitude")
    full_title = f"{title}\nAverage Scaled Error: {avg_scaled_error:.6f}"
    plt.title(full_title)
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()