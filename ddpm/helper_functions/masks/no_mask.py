import torch

from data_prep.data_initializer import DDInitializer
from ddpm.helper_functions.masks.abstract_mask import MaskGenerator


class NoMask(MaskGenerator):
    def __init__(self):
        pass

    def generate_mask(self, image_shape = None):

        dd = DDInitializer()

        if image_shape is None:
            print("image_shape is None")

        _, _, h, w = image_shape

        # Mask convention: 1.0 = missing (to inpaint), 0.0 = known.
        mask = torch.zeros((1, 1, h, w), dtype=torch.float32).to(dd.get_device())

        return mask

    def __str__(self):
        return "NoMask"

    def get_num_lines(self):
        return 0
