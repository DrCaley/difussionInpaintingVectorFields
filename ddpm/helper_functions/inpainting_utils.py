import torch
import torch.nn.functional as f

from data_prep.data_initializer import DDInitializer
from ddpm.helper_functions.model_evaluation import noise_strat

dd = DDInitializer()

def inpaint_generate_new_images(ddpm, input_image, mask, n_samples=16, device=None,
                                resample_steps=1, channels=1, height=64, width=128):
    """
    Given a DDPM model, an input image, and a mask, generates in-painted samples.
    """
    noised_images = [None] * (ddpm.n_steps + 1)
    device = dd.get_device()

    def denoise_one_step(noisy_img, noise_strat, t):
        time_tensor = torch.full((n_samples, 1), t, device=device, dtype=torch.long)
        epsilon_theta = ddpm.backward(noisy_img, time_tensor)

        alpha_t = ddpm.alphas[t].to(device)
        alpha_t_bar = ddpm.alpha_bars[t].to(device)

        if dd.get_attribute('gaussian_scaling'):
            less_noised_img = (1 / alpha_t.sqrt()) * (
                    noisy_img - ((1 - alpha_t) / (1 - alpha_t_bar).sqrt()) * epsilon_theta
            )
        else:
            less_noised_img = (1 / alpha_t.sqrt()) * (noisy_img - epsilon_theta)

        tensor_size = torch.zeros(n_samples, channels, height, width, device=device)

        if t > 0:
            z = noise_strat(tensor_size, torch.tensor([t], device=device))
            beta_t = ddpm.betas[t].to(device)
            sigma_t = beta_t.sqrt()
            less_noised_img = less_noised_img + sigma_t * z

        return less_noised_img

    def noise_one_step(unnoised_img, t, noise_strat):
        epsilon = noise_strat(unnoised_img, torch.tensor([t], device=device))
        noised_img = ddpm(unnoised_img, t, epsilon, one_step=True)
        return noised_img

    with torch.no_grad():

        noise_strat = dd.get_noise_strategy()

        input_img = input_image.clone().to(device)

        noise = noise_strat(input_img, torch.tensor([0], device=device))

        # Step-by-step forward noising
        noised_images[0] = input_img
        for t in range(ddpm.n_steps):
            noised_images[t + 1] = noise_one_step(noised_images[t], t, noise_strat)

        x = noised_images[ddpm.n_steps] * (1 - mask) + (noise * mask)

        for idx, t in enumerate(range(ddpm.n_steps - 1, -1, -1)):
            for i in range(resample_steps):
                x = denoise_one_step(x, noise_strat, t)
                x = noised_images[t] * (1 - mask) + (x * mask)

                if (i + 1) < resample_steps:
                    x = noise_one_step(x, t, noise_strat)
    return x


def calculate_mse(original_image, predicted_image, mask, flow=False):
    masked_original = original_image * mask
    masked_predicted = predicted_image * mask
    mse = f.mse_loss(masked_predicted, masked_original, reduction='sum') / mask.sum()
    return mse


def avg_pixel_value(original_image, predicted_image, mask):
    avg_pixel_value = torch.sum(torch.abs(original_image * mask)) / mask.sum()
    avg_diff = torch.sum(torch.abs((predicted_image * mask) - (original_image * mask))) / mask.sum()
    return avg_diff * (100 / avg_pixel_value)
