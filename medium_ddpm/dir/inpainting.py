import imageio
import numpy as np
import torch
from einops import rearrange
import torch.nn.functional as F

def inpaint_generate_new_images(ddpm, input_image, mask, n_samples=16, device=None, frames_per_gif=100,
                                gif_name="sampling.gif", c=1, h=64, w=128):
    """Given a DDPM model, an input image, and a mask, generates inpainted samples"""
    frame_idxs = np.linspace(0, ddpm.n_steps - 1, frames_per_gif).astype(np.uint)
    frames = []
    noised_imgs = [None] * (ddpm.n_steps + 1)

    with torch.no_grad():
        if device is None:
            device = ddpm.device

        input_img = input_image.clone().to(device)
        noise = torch.randn_like(input_img).to(device)

        # Adding noise step by step
        noised_imgs[0] = input_img
        for t in range(ddpm.n_steps):
            eta = torch.randn_like(input_img).to(device)
            noised_imgs[t + 1] = ddpm(noised_imgs[t], t, eta, one_step=True)

        x = noised_imgs[ddpm.n_steps] * (1 - mask) + (noise * mask)

        for idx, t in enumerate(range(ddpm.n_steps - 1, -1, -1)):
            time_tensor = (torch.ones(n_samples, 1) * t).to(device).long()
            eta_theta = ddpm.backward(x, time_tensor)

            alpha_t = ddpm.alphas[t]
            alpha_t_bar = ddpm.alpha_bars[t]

            # Partially denoising the image
            less_noised_img = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta)

            if t > 0:
                z = torch.randn(n_samples, c, h, w).to(device)
                beta_t = ddpm.betas[t]
                sigma_t = beta_t.sqrt()
                less_noised_img = less_noised_img + sigma_t * z

            x = noised_imgs[t] * (1 - mask) + (less_noised_img * mask)

            # Adding frames to the GIF
            if idx in frame_idxs or t == 0:
                # Normalize and prepare the frame for the GIF
                normalized = x.clone()
                for i in range(len(normalized)):
                    min_val = torch.min(normalized[i])
                    max_val = torch.max(normalized[i])
                    normalized[i] = (normalized[i] - min_val) / (max_val - min_val) * 255

                frame = rearrange(normalized, "b c h w -> b h w c")
                frame = frame.cpu().numpy().astype(np.uint8)

                grid_frame = rearrange(frame, "(b1 b2) h w c -> (b1 h) (b2 w) c", b1=int(n_samples ** 0.5))
                frames.append(grid_frame)

    # Save the frames as a GIF
    with imageio.get_writer(gif_name, mode="I") as writer:
        for idx, frame in enumerate(frames):
            rgb_frame = np.repeat(frame, 3, axis=2)  # Ensure RGB format
            writer.append_data(rgb_frame)

            # Show the last frame for a longer time
            if idx == len(frames) - 1:
                last_rgb_frame = np.repeat(frames[-1], 3, axis=2)
                for _ in range(frames_per_gif // 3):
                    writer.append_data(last_rgb_frame)

    return x


def naive_inpaint(input_image, mask):
    inpainted_image = input_image.clone()

    _, _, h, w = input_image.shape

    for y in range(h):
        for x in range(w):
            if mask[0, 0, y, x] == 1:  # If in the masked region
                # Get neighboring pixels
                neighbors = []
                if y > 0:
                    neighbors.append(input_image[0, 0, y - 1, x])
                if y < h - 1:
                    neighbors.append(input_image[0, 0, y + 1, x])
                if x > 0:
                    neighbors.append(input_image[0, 0, y, x - 1])
                if x < w - 1:
                    neighbors.append(input_image[0, 0, y, x + 1])

                if neighbors:
                    inpainted_image[0, 0, y, x] = torch.mean(torch.stack(neighbors))
                else:
                    inpainted_image[0, 0, y, x] = 0
    return inpainted_image



def calculate_mse(original_image, predicted_image, mask):
    """Calculate Mean Squared Error between the original and predicted images for the masked area only."""
    masked_original = original_image * mask
    masked_predicted = predicted_image * mask
    mse = F.mse_loss(masked_predicted, masked_original, reduction='sum') / mask.sum()
    return mse

def avg_pixel_value(original_image, predicted_image, mask):
    avg_pixel_value = torch.sum(torch.abs(original_image * mask)) / mask.sum()

    avg_diff = (torch.sum(torch.abs((predicted_image * mask) - (original_image * mask))) / mask.sum())

    return avg_diff * (100 / avg_pixel_value)

