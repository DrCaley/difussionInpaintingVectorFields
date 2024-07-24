import random
import torch
import numpy as np

def generate_random_mask(image_shape, input_image_original, max_mask_size=32):
    _, _, h, w = image_shape
    mask = torch.zeros((1, 1, h, w), dtype=torch.float32)

    while True:
        mask_h = random.randint(1, max_mask_size)
        mask_w = random.randint(1, max_mask_size)
        start_h = random.randint(0, 44 - mask_h)
        start_w = random.randint(0,94 - mask_w)

        if torch.all(input_image_original[:, :, start_h:start_h + mask_h, start_w:start_w + mask_w] != 0):
            mask[:, :, start_h:start_h + mask_h, start_w:start_w + mask_w] = 1
            break

    return mask


def generate_straight_line_mask(image_shape, land_mask, num_lines=10, line_thickness=5):
    _, _, h, w = image_shape

    mask = np.zeros((h, w), dtype=np.float32)

    # Exclude border
    area_height = 44
    area_width = 94

    for _ in range(num_lines):
        y = random.randint(0, area_height - line_thickness)
        mask[y:y + line_thickness, 0:0 + area_width] = 1.0

    mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # Exclude land
    mask = (land_mask != 0) * mask

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


def generate_robot_path_mask(image_shape, land_mask, num_lines=0, line_thickness=5):
    _, _, h, w = image_shape

    mask = np.ones((h, w), dtype=np.float32)

    area_top = 64 - 44
    area_left = 0
    area_bottom = 64
    area_right = 94

    current_y = 0
    current_x = 0

    directions = [(0, line_thickness), (0, -line_thickness), (line_thickness, 0), (-line_thickness, 0)]  # (dy, dx) for (N, S, E, W)

    for _ in range(num_lines):
        current_y = max(area_top, min(area_bottom - line_thickness, current_y))
        current_x = max(area_left, min(area_right - line_thickness, current_x))

        mask[current_y:current_y + line_thickness, current_x:current_x + line_thickness] = 0.0  # Unmask the path

        direction = random.choice(directions)
        new_y = current_y + direction[0]
        new_x = current_x + direction[1]

        new_y = max(area_top, min(area_bottom - line_thickness, new_y))
        new_x = max(area_left, min(area_right - line_thickness, new_x))

        current_y = new_y
        current_x = new_x

    mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    border_mask = create_border_mask(image_shape)

    mask = mask * land_mask * border_mask

    return mask


def create_border_mask(image_shape, area_height=44, area_width=94, offset_top=0, offset_left=0):
    _, _, h, w = image_shape
    border_mask = np.zeros((h, w), dtype=np.float32)
    border_mask[offset_top:offset_top + area_height, offset_left:offset_left + area_width] = 1.0
    return torch.tensor(border_mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

