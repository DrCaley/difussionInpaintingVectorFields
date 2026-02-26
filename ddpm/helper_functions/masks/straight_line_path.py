import math
import random
import numpy as np
import torch

from data_prep.data_initializer import DDInitializer
from ddpm.helper_functions.masks.abstract_mask import MaskGenerator
from ddpm.helper_functions.masks.border_mask import BorderMaskGenerator


dd = DDInitializer()


class StraightLinePathGenerator(MaskGenerator):
    def __init__(self, line_width=1):
        self.line_width = max(1, int(line_width))

    def _pick_start(self, valid_mask):
        valid_points = np.argwhere(valid_mask == 1)
        if len(valid_points) == 0:
            raise ValueError("No valid points to start line.")
        return random.choice(valid_points.tolist())

    def _pick_end(self, start_y, start_x, h, w):
        angle = random.uniform(0, 2 * math.pi)
        dx = math.cos(angle)
        dy = math.sin(angle)

        candidates = []
        if dx > 0:
            candidates.append((w - 1 - start_x) / dx)
        elif dx < 0:
            candidates.append((0 - start_x) / dx)
        if dy > 0:
            candidates.append((h - 1 - start_y) / dy)
        elif dy < 0:
            candidates.append((0 - start_y) / dy)

        candidates = [t for t in candidates if t > 0]
        if not candidates:
            return start_y, start_x

        t = min(candidates)
        end_x = int(round(start_x + t * dx))
        end_y = int(round(start_y + t * dy))
        end_x = max(0, min(w - 1, end_x))
        end_y = max(0, min(h - 1, end_y))
        return end_y, end_x

    def _draw_line(self, mask, valid_mask, start_y, start_x, end_y, end_x):
        length = int(max(abs(end_x - start_x), abs(end_y - start_y))) + 1
        if length <= 1:
            mask[start_y, start_x] = 0.0
            return

        ys = np.linspace(start_y, end_y, length)
        xs = np.linspace(start_x, end_x, length)
        radius = self.line_width // 2

        for y, x in zip(ys, xs):
            cy = int(round(y))
            cx = int(round(x))
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    ny = cy + dy
                    nx = cx + dx
                    if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1]:
                        if valid_mask[ny, nx] == 1:
                            mask[ny, nx] = 0.0

    def generate_mask(self, image_shape=None):
        if image_shape is None:
            raise ValueError("Missing required shape.")

        _, _, h, w = image_shape
        device = dd.get_device()
        # Mask convention: 1.0 = missing (to inpaint), 0.0 = known.
        mask = np.ones((h, w), dtype=np.float32)

        border_mask = BorderMaskGenerator().generate_mask(image_shape=image_shape)
        border_mask = border_mask.to(device)
        valid_mask = border_mask.squeeze().cpu().numpy()
        if valid_mask.ndim == 3:
            valid_mask = valid_mask[0]

        start_y, start_x = self._pick_start(valid_mask)
        end_y, end_x = self._pick_end(start_y, start_x, h, w)
        self._draw_line(mask, valid_mask, start_y, start_x, end_y, end_x)

        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        mask = mask * border_mask
        return mask

    def __str__(self):
        return "StraightLinePath"

    def get_num_lines(self):
        return self.line_width
