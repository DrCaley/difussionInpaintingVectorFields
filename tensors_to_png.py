#tensors_to_png.py
from PIL import Image
import numpy as np
import torch

#input should be a 3x44x94 tensor
#outputs are normalized, so comparing the strength of currents between maps won't work
def generate_png(tensors):
    expected_shape = (3, 44, 94)
    if tensors.shape != expected_shape:
        raise ValueError(f"Expected tensor with shape {expected_shape}, but got {tensors.shape}.")

    img = Image.new('RGB', (94, 44), color='white')
    vectors_arr = tensors.detach().numpy()
    r_max, r_min = np.nanmax(vectors_arr[0]), np.nanmin(vectors_arr[0])
    g_max, g_min = np.nanmax(vectors_arr[1]), np.nanmin(vectors_arr[1])
    b_max, b_min = np.nanmax(vectors_arr[2]), np.nanmin(vectors_arr[2])
    for y in range(44):
        for x in range(94):
            #normalize so it works better with tensors that have been noised out of bounds
            r = (vectors_arr[0][y][x] - r_min) / (r_max - r_min)
            g = (vectors_arr[1][y][x] - g_min) / (g_max - g_min)
            b = (vectors_arr[2][y][x] - b_min) / (b_max - b_min)
            #scale to work with images and set nan to 0
            r = 0 if np.isnan(r) else int(r * 255.0)
            g = 0 if np.isnan(g) else int(g * 255.0)
            b = 0 if np.isnan(b) else int(b * 255.0)

            img.putpixel((x, -y), (r, g, b))
    scale = 9
    img.resize((94 * scale, 44 * scale), resample=Image.BOX).show()
