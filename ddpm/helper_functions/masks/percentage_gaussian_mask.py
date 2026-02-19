import torch
from ddpm.helper_functions.masks.abstract_mask import MaskGenerator

class GaussianNoisePercentageMaskGenerator(MaskGenerator):
    def __init__(self, mask_fraction=0.05, mean=0.0, std=1.0):
        """
        mask_fraction: float between 0 and 1, fraction of pixels to mask
        mean: mean of Gaussian noise
        std: standard deviation of Gaussian noise
        """
        if not 0 <= mask_fraction <= 1:
            raise ValueError("mask_fraction must be between 0 and 1")
        self.mask_fraction = mask_fraction
        self.mean = mean
        self.std = std

    def generate_mask(self, image_shape=None):
        if image_shape is None:
            raise ValueError("image_shape must be provided")
        
        noise = torch.normal(mean=self.mean, std=self.std, size=image_shape)

        # Flatten noise to easily compute top fraction
        flat_noise = noise.flatten()
        num_pixels = flat_noise.numel()
        k = int(self.mask_fraction * num_pixels)

        if k == 0:
            # If mask_fraction is extremely small, return all zeros
            return torch.zeros_like(noise)

        # Find threshold to mask exactly k pixels
        threshold_value = torch.topk(flat_noise, k, largest=True).values.min()

        # Mask pixels above threshold
        binary_mask = (noise >= threshold_value).float()

        return binary_mask

    def __str__(self):
        return (f"GaussianNoisePercentageMaskGenerator(mask_fraction={self.mask_fraction}")

    def get_num_lines(self):
        return 0