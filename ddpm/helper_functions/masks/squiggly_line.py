import random
import torch
import numpy as np

from data_prep.data_initializer import DDInitializer
from ddpm.helper_functions.masks.abstract_mask import MaskGenerator

dd = DDInitializer()

class SquigglyLineMaskGenerator(MaskGenerator):
    def __init__(self, num_lines=2, line_thickness=2):
        self.num_lines = num_lines
        self.line_thickness = line_thickness

    def generate_mask(self, image_shape=None):
        if image_shape is None:
            print("image_shape is None")

        _, _, h, w = image_shape
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
                        if 0 <= x+i < w and 0 <= y+j < h:
                            mask[y+j, x+i] = 1.0

        for _ in range(num_lines):
            start_point = (random.randint(0, w - 1), random.randint(0, h - 1))
            num_segments = random.randint(5, 10)
            points = [start_point]

            for _ in range(num_segments):
                next_point = (random.randint(0, w - 1), random.randint(0, h - 1))
                points.append(next_point)

            for i in range(len(points) - 1):
                draw_line(mask, points[i], points[i + 1], line_thickness)

        device = dd.get_device()

        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        mask = mask.to(device)

        return mask

    def __str__(self):
        return "SquigglyLine"

    def get_num_lines(self):
        return self.num_lines