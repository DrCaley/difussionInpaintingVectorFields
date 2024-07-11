import random
import torch
import numpy as np
from matplotlib import pyplot as plt


def generate_random_mask(image_shape, input_image_original, max_mask_size=32):
    _, _, h, w = image_shape
    mask = torch.zeros((1, 1, h, w), dtype=torch.float32)

    while True:
        mask_h = random.randint(1, max_mask_size)
        mask_w = random.randint(1, max_mask_size)
        start_h = random.randint(0, h - mask_h)
        start_w = random.randint(0, w - mask_w)

        if torch.all(input_image_original[:, :, start_h:start_h + mask_h, start_w:start_w + mask_w] != 0):
            mask[:, :, start_h:start_h + mask_h, start_w:start_w + mask_w] = 1
            break

    return mask

def generate_straight_line_mask(image_shape, input_image_original, num_lines=5, line_thickness=2, orientation='horizontal'):
    _, _, h, w = image_shape
    mask = np.zeros((h, w), dtype=np.float32)

    for _ in range(num_lines):
        if orientation == 'horizontal':
            y = random.randint(0, h - 1)
            while torch.any(input_image_original[:, :, y:y + line_thickness, :] == 0):
                y = random.randint(0, h - 1)
            mask[y:y + line_thickness, :] = 1.0
        elif orientation == 'vertical':
            x = random.randint(0, w - 1)
            while torch.any(input_image_original[:, :, :, x:x + line_thickness] == 0):
                x = random.randint(0, w - 1)
            mask[:, x:x + line_thickness] = 1.0

    mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return mask


def generate_squiggly_line_mask(image_shape, input_image_original, num_lines=5, line_thickness=2):
    _, _, h, w = image_shape
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
