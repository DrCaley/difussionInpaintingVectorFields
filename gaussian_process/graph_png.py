import numpy as np
from PIL import Image
from plots.adapt_visualize_data import plot_png


def normalize_image(img):
    norm_image = (img / 127.5) - 1.0
    return norm_image


def calculate_mse(original_image, reconstructed_image, mask):
    mse = np.mean((original_image - reconstructed_image) ** 2)
    return mse


image = Image.open('images/input_img_cropped.png').convert('RGB')
image = np.array(image, dtype=np.float32)
normalized_image = normalize_image(image)

mask = Image.open('images/mask_cropped.png').convert('L')
mask = np.array(mask, dtype=np.float32)
missing_pixel_mask = mask > 0
missing_pixel_mask = ~missing_pixel_mask

masked_image = normalized_image.copy()
masked_image[missing_pixel_mask] = np.nan

expanded_mask = np.repeat(missing_pixel_mask[..., np.newaxis], 3, axis=2)

# gp_image = Image.open('images/gp_image.png')
# gp_image = np.array(gp_image, dtype=np.float32)
# normalized_gp_image = normalize_image(gp_image)
# mse_gp = calculate_mse(normalized_image, normalized_gp_image, expanded_mask)
# print(f"GP MSE: {mse_gp}")
#
# ddpm_image = Image.open('images/ddpm_img_cropped.png')
# ddpm_image = np.array(ddpm_image, dtype=np.float32)
# normalized_ddpm_image = normalize_image(ddpm_image)
# mse_ddpm = calculate_mse(normalized_image, normalized_ddpm_image, expanded_mask)
# print(f"DDPM MSE: {mse_ddpm}")

plot_png('images/0_ddpm_image_0_sample0_resample20.png', './images/land_mask_cropped.png')
# plot_png('./images/gp_image.png', './images/land_mask_cropped.png')
plot_png('images/input_img_cropped.png', './images/land_mask_cropped.png')
