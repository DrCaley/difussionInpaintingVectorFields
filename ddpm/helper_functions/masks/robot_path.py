import random
import torch
import numpy as np
from collections import deque

from data_prep.data_initializer import DDInitializer
from ddpm.helper_functions.masks.abstract_mask import MaskGenerator
from ddpm.helper_functions.masks.border_mask import BorderMaskGenerator

dd = DDInitializer()


class RobotPathGenerator(MaskGenerator):

    def __init__(self, coverage_ratio=0.2):
        self.coverage_ratio = coverage_ratio

    def flood_fill_area(self, start_y, start_x, valid_mask, visited, max_depth=10):
        """Returns the number of unexplored cells reachable within max_depth."""
        h, w = valid_mask.shape
        seen = np.zeros_like(valid_mask, dtype=bool)
        q = deque([(start_y, start_x, 0)])
        count = 0
        while q:
            y, x, d = q.popleft()
            if d > max_depth:
                continue
            if not (0 <= y < h and 0 <= x < w):
                continue
            if seen[y, x] or visited[y, x] or valid_mask[y, x] != 1:
                continue
            seen[y, x] = True
            count += 1
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                q.append((y + dy, x + dx, d + 1))
        return count

    def generate_mask(self, image_shape=None):
        if image_shape is None:
            raise ValueError("Missing required shape.")

        _, _, h, w = image_shape
        device = dd.get_device()
        mask = np.ones((h, w), dtype=np.float32)

        border_mask = BorderMaskGenerator().generate_mask(image_shape=image_shape)

        border_mask = border_mask.to(device)
        valid_mask = border_mask.squeeze().cpu().numpy()
        if valid_mask.ndim == 3:
            valid_mask = valid_mask[0]

        visited = np.zeros_like(valid_mask, dtype=bool)
        valid_points = np.argwhere(valid_mask == 1)
        if len(valid_points) == 0:
            raise ValueError("No valid points to explore.")

        start_y, start_x = random.choice(valid_points.tolist())
        queue = deque([(start_y, start_x)])
        visited[start_y, start_x] = True
        mask[start_y, start_x] = 0.0

        target_cells = int(self.coverage_ratio * np.sum(valid_mask))
        explored = 1
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while queue and explored < target_cells:
            y, x = queue.popleft()

            best_score = -1
            best_dir = None
            for dy, dx in directions:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    if valid_mask[ny, nx] == 1 and not visited[ny, nx]:
                        score = self.flood_fill_area(ny, nx, valid_mask, visited, max_depth=10)
                        if score > best_score:
                            best_score = score
                            best_dir = (ny, nx)

            if best_dir:
                ny, nx = best_dir
                visited[ny, nx] = True
                mask[ny, nx] = 0.0
                queue.append((ny, nx))
                explored += 1

        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        mask = mask * border_mask

        return mask

    def __str__(self):
        return "RobotPath"

    def get_num_lines(self):
        return super().get_num_lines()