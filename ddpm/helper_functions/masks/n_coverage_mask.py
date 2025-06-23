import numpy as np
import torch
import random
from collections import deque

from data_prep.data_initializer import DDInitializer
from ddpm.helper_functions.masks.abstract_mask import MaskGenerator
from ddpm.helper_functions.masks.border_mask import BorderMaskGenerator

dd = DDInitializer()

class CoverageMaskGenerator(MaskGenerator):
    def __init__(self, coverage_ratio=0.2):
        self.coverage_ratio = coverage_ratio

    def generate_mask(self, image_shape=None):
        if image_shape is None:
            raise ValueError("image_shape is None")

        _, _, h, w = image_shape
        device = dd.get_device()

        # Use first channel of land mask only
        border_mask = BorderMaskGenerator().generate_mask(image_shape=image_shape).to(device)

        # Compute valid area
        valid_area_tensor = border_mask  # [1,1,H,W]
        valid_area = valid_area_tensor.squeeze().cpu().numpy()  # (H, W)

        visited = np.zeros((h, w), dtype=bool)
        mask = np.zeros((h, w), dtype=np.float32)

        yx = np.argwhere(valid_area == 1)
        if len(yx) == 0:
            raise ValueError("No valid area to explore.")
        start_y, start_x = random.choice(yx)

        target_explore = int(self.coverage_ratio * np.sum(valid_area))
        explored = 0

        queue = deque([(start_y, start_x)])
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while queue and explored < target_explore:
            y, x = queue.popleft()
            if not (0 <= y < h and 0 <= x < w):
                continue
            if visited[y, x] or valid_area[y, x] != 1:
                continue

            visited[y, x] = True
            mask[y, x] = 1.0  # mark as missing (to be filled)
            explored += 1

            for dy, dx in directions:
                queue.append((y + dy, x + dx))

        # Final mask: shape [1, 2, H, W], mask == 1 => missing
        final_mask = np.stack([mask, mask], axis=0)  # shape [2, H, W]
        final_mask = torch.tensor(final_mask, dtype=torch.float32).unsqueeze(0).to(device)  # [1,2,H,W]

        return final_mask

    def __str__(self):
        return "coverage"

    def get_num_lines(self):
        return self.coverage_ratio
