import torch
from ddpm.helper_functions.masks.abstract_mask import MaskGenerator
import math

class OceanRobotPathMask(MaskGenerator):
    def __init__(self, steps=50, radius=10, start_pos=None, device="cpu"):
        self.steps = steps
        self.radius = radius
        self.start_pos = start_pos  # Tuple (y, x) or None
        self.device = device

    def __str__(self):
        return f"OceanRobotPathMask"

    def get_num_lines(self):
        return self.steps

    def generate_mask(self, image_shape=None):
        if image_shape is None:
            raise ValueError("Image shape must be provided")

        height, width = image_shape[-2:]
        # Mask convention: 1.0 = missing (to inpaint), 0.0 = known.
        mask = torch.ones((height, width), dtype=torch.float32, device=self.device)

        # Initial position
        if self.start_pos is None:
            pos = torch.tensor([
                torch.randint(0, height, (1,), device=self.device).item(),
                torch.randint(0, width, (1,), device=self.device).item()
            ], dtype=torch.float32)
        else:
            pos = torch.tensor(self.start_pos, dtype=torch.float32, device=self.device)

        # Initial random direction
        direction = torch.randn(2, device=self.device)
        direction = direction / torch.norm(direction)

        for _ in range(self.steps):
            self._reveal_circle(mask, pos, self.radius)

            # Try to step in the current direction
            for _ in range(10):
                new_pos = pos + direction * self.radius * 1.5
                lower_bound = torch.tensor([0, 0], device=self.device, dtype=new_pos.dtype)
                upper_bound = torch.tensor([height - 1, width - 1], device=self.device, dtype=new_pos.dtype)
                new_pos = torch.max(torch.min(new_pos, upper_bound), lower_bound)

                y = int(new_pos[0].item())
                x = int(new_pos[1].item())

                direction = torch.randn(2, device=self.device)
                direction = direction / torch.norm(direction)

            # Slight randomness
            direction += 0.3 * torch.randn(2, device=self.device)
            direction = direction / torch.norm(direction)

        return mask

    def _reveal_circle(self, mask, center, radius):
        """Mark a circular patch as known by setting mask to 0 in that region."""
        h, w = mask.shape
        y = torch.arange(h, device=self.device).view(-1, 1)
        x = torch.arange(w, device=self.device).view(1, -1)

        cy, cx = center[0], center[1]
        dist_squared = (y - cy)**2 + (x - cx)**2
        region = dist_squared <= radius**2

        mask[region] = 0.0