import random
import torch
import numpy as np
from torch.xpu import device

from data_prep.data_initializer import DDInitializer
from ddpm.helper_functions.mask_factory.masks.abstract_mask import MaskGenerator
from ddpm.helper_functions.mask_factory.masks.border_mask import BorderMaskGenerator

dd = DDInitializer

class NoMask(MaskGenerator):
    def __init__(self):
        pass

    def generate_mask(self, image_shape = None, land_mask = None):
        if image_shape is None:
            print("image_shape is None")
        if land_mask is None:
            print("land_mask is None")

        _, _, h, w = image_shape

        mask = torch.zeros((h, w)).to(device)

        return mask

    def __str__(self):
        return "NoMask"

    def get_num_lines(self):
        return 0
