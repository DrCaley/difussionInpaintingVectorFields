#tensors_to_png.py
from PIL import Image
import numpy as np
import torch


#input should be a 3 by n by m tensor
#outputs are normalized, so comparing the strength of currents between maps won't work
def generate_png(tensors, scale=9):
    shape = tensors.shape

    img = Image.new('RGB', (shape[2], shape[1]), color='white')
    vectors_arr = tensors.detach().numpy()
    r_max, r_min = np.nanmax(vectors_arr[0]), np.nanmin(vectors_arr[0])
    g_max, g_min = np.nanmax(vectors_arr[1]), np.nanmin(vectors_arr[1])
    b_max, b_min = np.nanmax(vectors_arr[2]), np.nanmin(vectors_arr[2])
    for y in range(shape[1]):
        for x in range(shape[2]):
            #normalize so it works better with tensors that have been noised out of bounds
            r = (vectors_arr[0][y][x] - r_min) / (r_max - r_min)
            g = (vectors_arr[1][y][x] - g_min) / (g_max - g_min)
            b = (vectors_arr[2][y][x] - b_min) / (b_max - b_min)
            #scale to work with images and set nan to 0
            r = 0 if np.isnan(r) else int(r * 255.0)
            g = 0 if np.isnan(g) else int(g * 255.0)
            b = 0 if np.isnan(b) else int(b * 255.0)

            img.putpixel((x, -y), (r, g, b))
    img.resize((shape[2] * scale, shape[1] * scale), resample=Image.BOX).show()

#generate_png(torch.load("./data/tensors/0.pt"))