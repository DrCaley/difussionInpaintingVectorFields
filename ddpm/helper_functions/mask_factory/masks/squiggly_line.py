import random
import torch
import numpy as np

from ddpm.helper_functions.mask_factory.masks.abstract_mask import MaskGenerator


class SquigglyLineMaskGenerator(MaskGenerator):
    def __init__(self, input_image_original, num_lines=5, line_thickness=2):
        self.input_image_original = input_image_original
        self.num_lines = num_lines
        self.line_thickness = line_thickness

    def generate_mask(self, image_shape=None, land_mask=None):
        if image_shape is None:
            print("image_shape is None")

        _, _, h, w = image_shape
        input_image_original = self.input_image_original
        num_lines = self.num_lines
        line_thickness = self.line_thickness

        mask = np.zeros((h, w), dtype=np.float32)

        def draw_line(mask, start_point, end_point, thickness):
            x0, y0 = start_point
            x1, y1 = end_point
            num_points = max(abs(x1 - x0), abs(y1 - y0)) * 2
            x_values = np.linspace(x0, x1, num_points).astype(int)
            y_values = np.linspace(y0, y1, num_points).astype(int)
            for x, y in zip(x_values, y_values):
                for i in range(-thickness//2, thickness//2 + 1):
                    for j in range(-thickness//2, thickness//2 + 1):
                        if 0 <= x+i < w and 0 <= y+j < h and input_image_original[0, 0, y+j, x+i] != 0:
                            mask[y+j, x+i] = 1.0

        for _ in range(num_lines):
            start_point = (random.randint(0, w - 1), random.randint(0, h - 1))
            while input_image_original[0, 0, start_point[1], start_point[0]] == 0:
                start_point = (random.randint(0, w - 1), random.randint(0, h - 1))
            num_segments = random.randint(5, 10)
            points = [start_point]

            for _ in range(num_segments):
                next_point = (random.randint(0, w - 1), random.randint(0, h - 1))
                while input_image_original[0, 0, next_point[1], next_point[0]] == 0:
                    next_point = (random.randint(0, w - 1), random.randint(0, h - 1))
                points.append(next_point)

            for i in range(len(points) - 1):
                draw_line(mask, points[i], points[i + 1], line_thickness)

        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return mask

    def __str__(self):
        return "SquigglyLine"