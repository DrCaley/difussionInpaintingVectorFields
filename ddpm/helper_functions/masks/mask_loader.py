from ddpm.helper_functions.masks.abstract_mask import MaskGenerator
import torch

class MaskLoader(MaskGenerator):
    def __init__(self, file_path):
        self.mask = torch.load(file_path)

    def generate_mask(self, image_shape = None):
        return self.mask

    def __str__(self):
        return 'MaskLoader'

    def get_num_lines(self):
        return 0