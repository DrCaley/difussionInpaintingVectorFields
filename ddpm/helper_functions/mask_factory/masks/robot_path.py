import random
import torch
import numpy as np

from ddpm.helper_functions.mask_factory.masks.abstract_mask import MaskGenerator
from ddpm.helper_functions.mask_factory.masks.border_mask import BorderMaskGenerator


class RobotPathMaskGenerator(MaskGenerator):

    def __init__(self, num_squares=10, square_size=10, line_thickness=1):
        self.line_thickness = line_thickness
        self.square_size = square_size
        self.num_squares = num_squares

    def generate_mask(self, image_shape = None, land_mask = None):
        if image_shape is None:
            print("image_shape is None")
        if land_mask is None:
            print("land_mask is None")

        num_squares = self.num_squares
        square_size = self.square_size

        _, _, h, w = image_shape

        mask = np.ones((h, w), dtype=np.float32)

        area_top = 0
        area_left = 0
        area_bottom = 44
        area_right = 94

        grid_rows = 4  # 44 / 10 = 4.4
        grid_cols = 9  # 94 / 10 = 9.4
        grid_height = area_bottom // grid_rows
        grid_width = area_right // grid_cols

        for row in range(grid_rows):
            for col in range(grid_cols):
                if num_squares <= 0:
                    break

                current_y = random.randint(area_top + row * grid_height, area_top + (row + 1) * grid_height - square_size)
                current_x = random.randint(area_left + col * grid_width, area_left + (col + 1) * grid_width - square_size)

                for i in range(square_size):
                    if current_x + i < w:
                        mask[current_y, current_x + i] = 0.0

                for i in range(square_size):
                    if current_y + i < h:
                        mask[current_y + i, current_x + square_size - 1] = 0.0

                for i in range(square_size):
                    if current_x + square_size - 1 - i >= 0:
                        mask[current_y + square_size - 1, current_x + square_size - 1 - i] = 0.0

                for i in range(square_size):
                    if current_y + square_size - 1 - i >= 0:
                        mask[current_y + square_size - 1 - i, current_x] = 0.0

                num_squares -= 1

        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # (dy, dx) for (N, S, E, W)
        current_y = random.randint(area_top, area_bottom - square_size)
        current_x = random.randint(area_left, area_right - square_size)

        for _ in range(num_squares):
            for i in range(square_size):
                if current_x + i < w:
                    mask[current_y, current_x + i] = 0.0

            for i in range(square_size):
                if current_y + i < h:
                    mask[current_y + i, current_x + square_size - 1] = 0.0

            for i in range(square_size):
                if current_x + square_size - 1 - i >= 0:
                    mask[current_y + square_size - 1, current_x + square_size - 1 - i] = 0.0

            for i in range(square_size):
                if current_y + square_size - 1 - i >= 0:
                    mask[current_y + square_size - 1 - i, current_x] = 0.0

            direction = random.choice(directions)
            new_y = current_y + direction[0] * square_size
            new_x = current_x + direction[1] * square_size

            new_y = max(area_top, min(area_bottom - square_size, new_y))
            new_x = max(area_left, min(area_right - square_size, new_x))

            current_y = new_y
            current_x = new_x

        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        border_mask = BorderMaskGenerator().generate_mask(image_shape=image_shape, land_mask=land_mask)

        mask = mask * land_mask * border_mask

        return mask

    def __str__(self):
        return "RobotPath"