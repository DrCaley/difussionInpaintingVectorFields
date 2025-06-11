import torch
import numpy as np

from data_prep.data_initializer import DDInitializer
from ddpm.helper_functions.mask_factory.masks.abstract_mask import MaskGenerator

dd = DDInitializer()

class BorderMaskGenerator(MaskGenerator):
    def __init__(self, area_height=44, area_width=94, offset_top=0, offset_left=0):
        self.border_mask = None
        self.area_height = area_height
        self.area_width = area_width
        self.offset_top = offset_top
        self.offset_left = offset_left

    def generate_mask(self, image_shape=None, land_mask=None):
        if image_shape is None:
            print("image_shape is None")

        offset_top = self.offset_top
        area_height = self.area_height
        offset_left = self.offset_left
        area_width = self.area_width
        _, _, h, w = image_shape


        self.border_mask = np.zeros((h, w), dtype=np.float32)
        self.border_mask[offset_top:offset_top + area_height, offset_left:offset_left + area_width] = 1.0

        device = dd.get_device()

        mask = torch.tensor(self.border_mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        mask = mask.to(device)
        return mask

    def __str__(self):
        return "BorderMask"

    def get_num_lines(self):
        return super().get_num_lines()
