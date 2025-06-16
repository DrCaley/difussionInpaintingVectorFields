import torch
import numpy as np
from PIL import Image

from data_prep.data_initializer import DDInitializer
from ddpm.helper_functions.masks.abstract_mask import MaskGenerator
from ddpm.helper_functions.masks.border_mask import BorderMaskGenerator

dd = DDInitializer()

class SmileyFaceMaskGenerator(MaskGenerator):

    def __init__(self, scale=1.0, center=True):
        self.scale = scale
        self.center = center  # center or place randomly

    def generate_mask(self, image_shape=None, land_mask=None):
        if image_shape is None:
            print("image_shape is None")
        if land_mask is None:
            print("land_mask is None")

        _, _, h, w = image_shape
        device = dd.get_device()

        # Initialize empty mask
        mask = np.zeros((h, w), dtype=np.float32)

        # Generate land and border masks
        border_mask = BorderMaskGenerator().generate_mask(image_shape=image_shape, land_mask=land_mask)
        land_mask_np = land_mask.squeeze().cpu().numpy()
        border_mask_np = border_mask.squeeze().cpu().numpy()
        valid_mask = land_mask_np * border_mask_np

        # Define pixel-art smiley pattern
        smiley_pattern = [
            "0" * 94, "0" * 94, "0" * 94, "0" * 94, "0" * 94, "0" * 94, "0" * 94,
            "000000000000111111111111111111100000000000000000000000000001111111111111111111000000000000",
            "000000000111111111111111111111111000000000000000000000000111111111111111111111110000000000",
            "000000001111111111111111111111111100000000000000000000011111111111111111111111111000000000",
            "000000011111111111111111111111111110000000000000000000111111111111111111111111111100000000",
            "000000011111111111111111111111111110000000000000000000111111111111111111111111111100000000",
            "000000111111111111111111111111111111000000000000000001111111111111111111111111111110000000",
            "000000011111111111111111111111111110000000000000000000111111111111111111111111111100000000",
            "000000011111111111111111111111111110000000000000000000111111111111111111111111111100000000",
            "000000001111111111111111111111111100000000000000000000011111111111111111111111111000000000",
            "000000000111111111111111111111111000000000000000000000001111111111111111111111110000000000",
            "000000000000111111111111111111100000000000000000000000000001111111111111111111000000000000",
            "0" * 94, "0" * 94, "0" * 94, "0" * 94, "0" * 94, "0" * 94,
            "000000000000000000000000000000000000111111111111111111111111111111000000000000000000000000",
            "000000000000000000000000000000001111111111111111111111111111111111111000000000000000000000",
            "000000000000000000000000000011111111111111111111111111111111111111111110000000000000000000",
            "000000000000000000000000111111111111111111111111111111111111111111111111000000000000000000",
            "000000000000000000001111111111111111111111111111111111111111111111111111100000000000000000",
            "000000000000000001111111111111111111111111111111111111111111111111111111111000000000000000",
            "000000000000000111111111111111111111111111111111111111111111111111111111111110000000000000",
            "000000000000001111111111111111111111111111111111111111111111111111111111111111000000000000",
            "000000000000001111111111111111111111111111111111111111111111111111111111111111000000000000",
            "000000000000000111111111111111111111111111111111111111111111111111111111111110000000000000",
            "000000000000000011111111111111111111111111111111111111111111111111111111111100000000000000",
            "000000000000000000111111111111111111111111111111111111111111111111111111110000000000000000",
            "000000000000000000001111111111111111111111111111111111111111111111111111000000000000000000",
            "000000000000000000000011111111111111111111111111111111111111111111111100000000000000000000",
            "000000000000000000000000111111111111111111111111111111111111111111110000000000000000000000",
            "000000000000000000000000001111111111111111111111111111111111111111000000000000000000000000",
            "0" * 94, "0" * 94, "0" * 94, "0" * 94,
        ]

        # Convert to binary array
        symbol = np.array([[int(c) for c in row] for row in smiley_pattern], dtype=np.float32)
        sym_h, sym_w = symbol.shape

        # Resize according to scale
        new_h, new_w = int(h * self.scale), int(w * self.scale)
        pil_img = Image.fromarray(symbol * 255)
        resized = pil_img.resize((new_w, new_h), Image.NEAREST)
        binary_symbol = (np.array(resized) > 127).astype(np.float32)

        sh, sw = binary_symbol.shape

        if self.center:
            top = (h - sh) // 2
            left = (w - sw) // 2
        else:
            valid_yx = np.argwhere(valid_mask == 1)
            if len(valid_yx) == 0:
                raise ValueError("No valid placement for smiley face.")
            top, left = valid_yx[np.random.choice(len(valid_yx))]
            top = max(0, min(top, h - sh))
            left = max(0, min(left, w - sw))

        # Apply symbol only on valid region
        subregion = valid_mask[top:top+sh, left:left+sw]
        symbol_mask = binary_symbol * subregion
        mask[top:top+sh, left:left+sw] = symbol_mask

        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        land_mask = land_mask.to(device)
        border_mask = border_mask.to(device)

        mask = mask * land_mask * border_mask
        mask = torch.cat([mask, mask.clone()], dim=1)  # (1, 2, H, W)
        return mask

    def __str__(self):
        return "SmileyFace"

    def get_num_lines(self):
        return super().get_num_lines()