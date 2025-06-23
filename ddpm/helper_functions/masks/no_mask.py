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

        mask = torch.zeros((h, w)).to(dd.get_device())

        return mask

    def __str__(self):
        return "NoMask"

    def get_num_lines(self):
        return 0
