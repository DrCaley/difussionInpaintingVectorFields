import random
import torch
import numpy as np

from ddpm.helper_functions.mask_factory.masks.abstract_mask import MaskGenerator
from ddpm.helper_functions.mask_factory.masks.border_mask import BorderMaskGenerator


class StraightLineMaskGenerator(MaskGenerator):
    def __init__(self, num_lines=10, line_thickness=5):
        self.num_lines = num_lines
        self.line_thickness = line_thickness

    def generate_mask(self, image_shape = None, land_mask = None):
        if image_shape is None:
            print("image_shape is None")
        if land_mask is None:
            print("land_mask is None")

        num_lines = self.num_lines
        line_thickness = self.line_thickness

        _, _, h, w = image_shape

        mask = np.ones((h, w), dtype=np.float32)

        # Exclude border
        area_height = 44
        area_width = 94

        for _ in range(num_lines):
            y = random.randint(0, area_height - line_thickness)
            mask[y:y + line_thickness, 0:0 + area_width] = 0.0

        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        border_mask = BorderMaskGenerator().generate_mask(image_shape, land_mask)
        # Exclude land
        mask = land_mask * mask * border_mask

        return mask

    def __str__(self):
        return "StraightLineMaskGenerator"