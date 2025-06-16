import numpy as np
import torch
import random
from collections import deque

from data_prep.data_initializer import DDInitializer
from ddpm.helper_functions.masks.abstract_mask import MaskGenerator
from ddpm.helper_functions.masks.border_mask import BorderMaskGenerator

dd = DDInitializer()

class BetterRobotPathGenerator(MaskGenerator):
    def __init__(self, coverage_ratio=0.2):
        self.coverage_ratio = coverage_ratio  # How much of the area to explore

    def generate_mask(self, image_shape=None, land_mask=None):
        if image_shape is None:
            raise ValueError("image_shape is None")
        if land_mask is None:
            raise ValueError("land_mask is None")

        _, _, h, w = image_shape
        device = dd.get_device()

        land_mask = land_mask.to(device)
        border_mask = BorderMaskGenerator().generate_mask(image_shape=image_shape, land_mask=land_mask).to(device)

        valid_area = (land_mask * border_mask).squeeze().cpu().numpy()
        visited = np.zeros((h, w), dtype=bool)
        mask = np.zeros((h, w), dtype=np.float32)

        # Seed: start in the middle or any random valid spot
        yx = np.argwhere(valid_area == 1)
        if len(yx) == 0:
            raise ValueError("No valid area to explore.")
        start_y, start_x = random.choice(yx)
        queue = deque([(start_y, start_x)])
        visited[start_y, start_x] = True
        mask[start_y, start_x] = 1.0

        target_explore = int(self.coverage_ratio * np.sum(valid_area))
        explored = 1

        directions = [(-1,0), (1,0), (0,-1), (0,1)]

        while queue and explored < target_explore:
            y, x = queue.popleft()
            random.shuffle(directions)  # Randomize path

            for dy, dx in directions:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    if valid_area[ny, nx] == 1 and not visited[ny, nx]:
                        visited[ny, nx] = True
                        mask[ny, nx] = 1.0
                        queue.append((ny, nx))
                        explored += 1
                        if explored >= target_explore:
                            break

        final_mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        return final_mask

    def __str__(self):
        return "BetterRobotPath"

    def get_num_lines(self):
        return super().get_num_lines()
