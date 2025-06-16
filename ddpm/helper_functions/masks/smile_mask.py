import numpy as np
import torch
from skimage.draw import ellipse, polygon
from ddpm.helper_functions.masks.abstract_mask import MaskGenerator  # Adjust import as needed

class SmileyMaskGenerator(MaskGenerator):
    def __init__(self):
        super().__init__()
        self.symbol_height = 44
        self.symbol_width = 94
        self.symbol_mask = self._create_symbol_mask()

    def _create_symbol_mask(self):
        mask = np.zeros((44, 94), dtype=np.uint8)

        # Face circle
        rr, cc = ellipse(22, 47, 20, 40, shape=mask.shape)
        mask[rr, cc] = 1

        # Eyes blank out (cut out)
        rr, cc = ellipse(12, 32, 4, 6, shape=mask.shape)
        mask[rr, cc] = 0
        rr, cc = ellipse(12, 62, 4, 6, shape=mask.shape)
        mask[rr, cc] = 0

        # Pupils filled in
        rr, cc = ellipse(12, 32, 2, 3, shape=mask.shape)
        mask[rr, cc] = 1
        rr, cc = ellipse(12, 62, 2, 3, shape=mask.shape)
        mask[rr, cc] = 1

        # Smile polygon (cut out shape to simulate a smile)
        smile_points = np.array([
            [32, 30],
            [36, 34],
            [38, 42],
            [36, 50],
            [32, 54],
            [30, 52],
            [34, 42],
            [30, 34],
        ])
        rr, cc = polygon(smile_points[:, 0], smile_points[:, 1], shape=mask.shape)
        mask[rr, cc] = 0

        return mask

    def generate_mask(self, image_shape=None, land_mask=None):
        if image_shape is None:
            raise ValueError("image_shape must be provided")
        if isinstance(image_shape, torch.Size):
            image_shape = tuple(image_shape)

        if len(image_shape) != 4 or image_shape[0] != 1 or image_shape[1] != 2:
            raise ValueError(f"Expected shape (1, 2, H, W), got {image_shape}")

        _, _, H, W = image_shape
        symbol_h, symbol_w = self.symbol_mask.shape

        if symbol_h > H or symbol_w > W:
            raise ValueError(f"Image shape too small to fit symbol: got {(H, W)}, symbol needs {(symbol_h, symbol_w)}")

        # Place symbol in the TOP-LEFT corner (start at 0,0)
        full_mask = np.zeros((H, W), dtype=np.uint8)
        full_mask[0:symbol_h, 0:symbol_w] = self.symbol_mask

        # Convert to tensor shape (1, 2, H, W)
        tensor_mask = torch.tensor(full_mask, dtype=torch.float32)
        tensor_mask = tensor_mask.unsqueeze(0).repeat(2, 1, 1)  # (2, H, W)
        return tensor_mask.unsqueeze(0)  # (1, 2, H, W)

    def __str__(self):
        return "SmileyMaskGenerator(size=44x94, top-left)"

    def get_num_lines(self):
        return 1
