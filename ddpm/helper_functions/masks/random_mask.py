import random
import torch

from data_prep.data_initializer import DDInitializer
from ddpm.helper_functions.masks.abstract_mask import MaskGenerator

dd = DDInitializer()

class RandomMaskGenerator(MaskGenerator):

    def __init__(self, max_mask_size=32):
        self.max_mask_size = max_mask_size

    def generate_mask(self, image_shape=None):
        if image_shape is None:
            print("image_shape is None")

        _, _, h, w = image_shape
        max_mask_size = self.max_mask_size

        # Mask convention: 1.0 = missing (to inpaint), 0.0 = known.
        mask = torch.ones((1, 1, h, w), dtype=torch.float32)

        while True:
            mask_h = random.randint(1, max_mask_size)
            mask_w = random.randint(1, max_mask_size)
            start_h = random.randint(0, 44 - mask_h)
            start_w = random.randint(0,94 - mask_w)
            break

        mask[:, :, start_h:start_h + mask_h, start_w:start_w + mask_w] = 0.0

        device = dd.get_device()

        mask = mask.to(device)

        return mask

    def __str__(self):
        return "RandomMask"

    def get_num_lines(self):
        return super().get_num_lines()