import torch
from ddpm.helper_functions.masks.abstract_mask import MaskGenerator


class GaussianNoiseBinaryMaskGenerator(MaskGenerator):
    def __init__(self, threshold=0.95, mean=0.0, std=1.0):
        self.threshold = threshold
        self.mean = mean
        self.std = std

    def generate_mask(self, image_shape=None, land_mask=None):

        if image_shape is None:
            raise ValueError("image_shape must be provided")

        noise = torch.normal(mean=self.mean, std=self.std, size=image_shape)

        binary_mask = (noise > self.threshold).float()

        if land_mask is not None:
            binary_mask *= land_mask

        return binary_mask

    def __str__(self):
        return (f"GaussianNoiseBinaryMaskGenerator(threshold={self.threshold}, "
                f"mean={self.mean}, std={self.std})")

    def get_num_lines(self):
        return 0
