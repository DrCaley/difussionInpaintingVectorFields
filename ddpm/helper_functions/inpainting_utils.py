import torch
import torch.nn.functional as f


def inpaint_generate_new_images(ddpm, input_image, mask, n_samples=16, device=None,
                                resample_steps=1, channels=1, height=64, width=128):
    """
    Based on the medium article. This does a bunch of hard math we don't understand - Feb 2025
    Given a DDPM model, an input image, and a mask, generates in-painted samples.

    Args:
        ddpm (DDPM): The DDPM model
        input_image (torch.Tensor): The input image
        mask (torch.Tensor): The mask
        n_samples (int): The number of samples to generate
        device (torch.device): The device to use
        resample_steps (int): The number of steps to resample the image
        channels (int): The number of channels to use
        height (int): The height of the image
        width (int): The width of the image
    Returns:
        Tensor: The generated samples
    """
    noised_images = [None] * (ddpm.n_steps + 1)

    def denoise_one_step(noisy_img):
        """
        Denoises image by one step.

        Returns:
            Slightly less noised image.
        """
        time_tensor = (torch.ones(n_samples, 1) * t).to(device).long()
        eta_theta = ddpm.backward(noisy_img, time_tensor)

        alpha_t = ddpm.alphas[t]
        alpha_t_bar = ddpm.alpha_bars[t]

        # Partially denoising the image
        less_noised_img = (1 / alpha_t.sqrt()) * (noisy_img - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta)

        if t > 0:
            z = torch.randn(n_samples, channels, height, width).to(device)
            beta_t = ddpm.betas[t]
            sigma_t = beta_t.sqrt()
            less_noised_img = less_noised_img + sigma_t * z

        return less_noised_img

    def noise_one_step(unnoised_img):
        """
        Noises image by one step.

        Returns:
            Slightly more noised image.
        """
        eta = torch.randn_like(unnoised_img).to(device)
        noised_img = ddpm(unnoised_img, t, eta, one_step=True)
        return noised_img

    with torch.no_grad():
        if device is None:
            device = ddpm.device

        input_img = input_image.clone().to(device)
        noise = torch.randn_like(input_img).to(device)

        # Adding noise step by step
        noised_images[0] = input_img
        for t in range(ddpm.n_steps):
            noised_images[t + 1] = noise_one_step(noised_images[t])

        x = noised_images[ddpm.n_steps] * (1 - mask) + (noise * mask)

        for idx, t in enumerate(range(ddpm.n_steps - 1, -1, -1)):

            for i in range(resample_steps):
                x = denoise_one_step(x)
                #combine parts
                x = noised_images[t] * (1 - mask) + (x * mask)
                #renoise and repeat if not the last step
                if not (i + 1) >= resample_steps:
                    x = noise_one_step(x)

    return x


def calculate_mse(original_image, predicted_image, mask, flow=False):
    """
    Calculate Mean Squared Error (MSE) between the original and predicted images for the masked area only.

    Args:
        original_image (torch.Tensor): The original image
        predicted_image (torch.Tensor): The predicted image
        mask (torch.Tensor): A binary mask tensor indicating valid regions same shape as image
        flow (bool, optional): what is this for?
    """
    masked_original = original_image * mask
    masked_predicted = predicted_image * mask
    mse = f.mse_loss(masked_predicted, masked_original, reduction='sum') / mask.sum()
    return mse


def avg_pixel_value(original_image, predicted_image, mask):
    """
    Computes the average percentage difference in pixel values between the original and predicted images
    within the masked region.

    Parameters:
        original_image (torch.Tensor): The ground truth image tensor.
        predicted_image (torch.Tensor): The predicted image tensor.
        mask (torch.Tensor): A binary mask tensor indicating valid regions (same shape as image).

    Returns:
        torch.Tensor: The average percentage pixel difference over the masked region.
    """
    avg_pixel_value = torch.sum(torch.abs(original_image * mask)) / mask.sum()

    avg_diff = (torch.sum(torch.abs((predicted_image * mask) - (original_image * mask))) / mask.sum())

    return avg_diff * (100 / avg_pixel_value)
