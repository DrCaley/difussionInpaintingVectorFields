import random
import torch
import numpy as np

from data_prep.data_initializer import DDInitializer
from ddpm.helper_functions.masks.abstract_mask import MaskGenerator
from ddpm.helper_functions.masks.border_mask import BorderMaskGenerator

dd = DDInitializer()

class RandomPathMaskGenerator(MaskGenerator):

    def __init__(self, num_lines = 10, line_thickness = 2, line_length = 5 ):
        self.num_lines = num_lines
        self.line_thickness = line_thickness
        self.line_length = line_length


    def generate_mask(self, image_shape = None):
        if image_shape is None:
            print("image_shape is None")

        _, _, h, w = image_shape
        line_thickness = self.line_thickness
        num_lines = self.num_lines
        line_length = self.line_length

        # Mask convention: 1.0 = missing (to inpaint), 0.0 = known.
        mask = np.ones((h, w), dtype=np.float32)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        area_top = 64 - 44
        area_left = 0
        area_bottom = 64
        area_right = 94

        current_y = random.randint(0, 44)
        current_x = random.randint(0,94)


        directions = [(0, line_thickness),
                      (0, -line_thickness),
                      (line_thickness, 0),
                      (-line_thickness, 0)]  # (dy, dx) for (N, S, E, W)

        for _ in range(num_lines):
            current_y = max(area_top, min(area_bottom - line_thickness, current_y))
            current_x = max(area_left, min(area_right - line_thickness, current_x))

            direction = random.choice(directions)

            for i in range(line_length):

                mask[current_y:current_y + line_thickness, current_x:current_x + line_thickness] = 0.0

                new_y = current_y + direction[0]
                new_x = current_x + direction[1]

                new_y = max(area_top, min(area_bottom - line_thickness, new_y))
                new_x = max(area_left, min(area_right - line_thickness, new_x))

                current_y = new_y
                current_x = new_x

        border_mask = BorderMaskGenerator().generate_mask(image_shape=image_shape)

        device = dd.get_device()

        mask = mask.to(device)
        border_mask = border_mask.to(device)

        mask = mask * border_mask

        return mask

    def get_num_lines(self):
        return self.num_lines

    def __str__(self):
        return "RandomPath"
