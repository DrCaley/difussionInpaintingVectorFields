from PIL import Image
import numpy as np
import torch
import os


# Input should be a 3 by n by m tensor
# Outputs are normalized, so comparing the strength of currents between maps won't work
def generate_png(tensors, scale=1, output_path="./results", filename="output.png"):
    shape = tensors.shape
    channels = shape[0]
    dim_num = len(shape)

    if len(shape) > 3:
        for i, tensor in enumerate(tensors):
            generate_png(tensor, scale, output_path, f"{i}_{filename}")
        return

    if len(shape) < 3:
        raise ValueError(f"Expected tensor with at least 3 dimensions,"
                         f" but got {len(shape)} dimensions for Tensor with shape {shape}.")

    img = Image.new('RGB', (shape[2], shape[1]), color='white')
    vectors_arr = tensors.detach().numpy()
    pixels_arr = np.zeros(shape)
    maxes = [np.nanmax(vectors_arr[i]) for i in range(channels)]
    mins = [np.nanmin(vectors_arr[i]) for i in range(channels)]

    for y in range(shape[1]):
        for x in range(shape[2]):
            # Normalize so it works better with tensors that have been noised out of bounds
            rgb = [0, 0, 0]
            for i in range(3):
                if i < channels:
                    denom = maxes[i] - mins[i]
                    if denom == 0:
                        rgb[i] = 0
                    else:
                        rgb[i] = (vectors_arr[i][y][x] - mins[i]) / denom
                if np.isnan(rgb[i]):
                    rgb[i] = 0
                rgb[i] *= 255.0
                rgb[i] = int(rgb[i])

            # -y-1 it's negative so that the orientation and alignment work out
            img.putpixel((x, -y - 1), tuple(rgb))

    # Resize and save the image
    img = img.resize((shape[2] * scale, shape[1] * scale), resample=Image.BOX)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    img.save(os.path.join(output_path, filename))

# Example for testing
# generate_png(torch.load("./data/tensors/0.pt"))
