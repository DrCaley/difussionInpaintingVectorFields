import torch
import matplotlib.pyplot as plt
import numpy as np
import os

def save_mse_heatmap(tensor1, tensor2, mask, save_path="mse_heatmap.png",
                     title="Masked Pixel-wise MSE Heatmap", mask_color="lightgray",
                     crop_shape=(44, 94), cmap_name="viridis"):
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

    # Calculate average MSE over valid (masked=1) pixels in crop
    valid_pixels = cropped_mask == 1
    if np.any(valid_pixels):
        avg_mse = cropped_mse[valid_pixels].mean()
    else:
        avg_mse = float('nan')

    print(f"Average MSE per pixel over masked area in crop: {avg_mse:.6f}")

    masked_array = np.where(cropped_mask == 1, cropped_mse, np.nan)

    plt.figure(figsize=(8, 6))
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color=mask_color)

    plt.imshow(masked_array, cmap=cmap, interpolation='nearest')
    plt.colorbar(label="Pixel MSE")
    plt.title(title)
    plt.xlabel("Width")
    plt.ylabel("Height")

    dir_name = os.path.dirname(save_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def save_angular_error_heatmap(tensor1, tensor2, mask, save_path="angular_error_heatmap.png",
                               crop_shape=(44, 94), mask_color="lightgray", cmap_name="viridis", title="Masked Angular Error Heatmap"):
    # tensor shape: (1, 2, H, W)
    single_mask = mask[:, 0:1, :, :]

    u_pred = tensor1[:, 0, :, :]
    v_pred = tensor1[:, 1, :, :]
    u_true = tensor2[:, 0, :, :]
    v_true = tensor2[:, 1, :, :]

    # Compute dot product and norms
    dot = u_pred * u_true + v_pred * v_true  # (1, H, W)
    norm_pred = torch.sqrt(u_pred ** 2 + v_pred ** 2) + 1e-8
    norm_true = torch.sqrt(u_true ** 2 + v_true ** 2) + 1e-8

    cos_angle = dot / (norm_pred * norm_true)
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
    angle = torch.acos(cos_angle)  # radians

    angle_deg = angle * (180.0 / 3.14159265)  # convert to degrees

    angle_np = angle_deg.squeeze().cpu().numpy()
    mask_np = single_mask.squeeze().cpu().numpy()

    crop_h, crop_w = crop_shape
    cropped_angle = angle_np[:crop_h, :crop_w]
    cropped_mask = mask_np[:crop_h, :crop_w]

    masked_angle = np.where(cropped_mask == 1, cropped_angle, np.nan)

    avg_angle_error = cropped_angle[cropped_mask == 1].mean()
    print(f"Average angular error (degrees) over masked crop: {avg_angle_error:.3f}")

    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color=mask_color)

    plt.imshow(masked_angle, cmap=cmap, interpolation='nearest')
    plt.colorbar(label="Angular error (degrees)")
    plt.title(title)
    plt.xlabel("Width")
    plt.ylabel("Height")

    import os
    dir_name = os.path.dirname(save_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

