import random
import torch

from ddpm.helper_functions.mask_factory.masks.abstract_mask import MaskGenerator

class RandomMaskGenerator(MaskGenerator):

    def __init__(self, input_image_original, max_mask_size=32):
        self.input_image_original = input_image_original
        self.max_mask_size = max_mask_size

    def generate_mask(self, image_shape=None, land_mask=None):
        if image_shape is None:
            print("image_shape is None")

        _, _, h, w = image_shape
        max_mask_size = self.max_mask_size
        input_image_original = self.input_image_original

        mask = torch.zeros((1, 1, h, w), dtype=torch.float32)

        while True:
            mask_h = random.randint(1, max_mask_size)
            mask_w = random.randint(1, max_mask_size)
            start_h = random.randint(0, 44 - mask_h)
            start_w = random.randint(0,94 - mask_w)

            if torch.all(input_image_original[:, :, start_h:start_h + mask_h, start_w:start_w + mask_w] != 0):
                mask[:, :, start_h:start_h + mask_h, start_w:start_w + mask_w] = 1
                break

        return mask

    def __str__(self):
        return "RandomMask"

    def get_num_lines(self):
        return super().get_num_lines()