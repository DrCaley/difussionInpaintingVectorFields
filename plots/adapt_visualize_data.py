import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from data_prep.ocean_image_dataset import OceanImageDataset
from plots.visualize_data import plotQuiverData



"""Adapted from WHOI. We use some of this to vizualize data."""

def revert_png(image_path, original_shape, scale=1, compare_to=None):
    """To change the original png visualization format back into data. No longer useful"""
    img = Image.open(image_path)
    img = img.resize((original_shape[2], original_shape[1]), resample=Image.BOX)
    img_array = np.array(img)

    reverted_tensor = np.zeros((original_shape[0], original_shape[1], original_shape[2]))

    if compare_to is None:
        raise ValueError("compare_to parameter is required to revert the image accurately.")
    else:
        compare_to = compare_to.detach().numpy()

    maxes = [np.nanmax(compare_to[i]) for i in range(original_shape[0])]
    mins = [np.nanmin(compare_to[i]) for i in range(original_shape[0])]

    for y in range(original_shape[1]):
        for x in range(original_shape[2]):
            for i in range(3):
                if i < original_shape[0]:
                    pixel_value = img_array[-y - 1, x, i]
                    denom = maxes[i] - mins[i]
                    if denom == 0:
                        reverted_value = mins[i]
                    else:
                        reverted_value = pixel_value / 255.0 * denom + mins[i]
                    reverted_tensor[i, y, x] = reverted_value

    reverted_tensor = torch.tensor(reverted_tensor)

    return reverted_tensor


def plot_png(image_path, land_mask_path):
    """
    To change the original png data vizualization into a plot. No longer useful
    :param image_path: Path to ocean image to be plotted, must be 94x44
    :param land_mask_path: Path to land mask image, must be 94x44
    """
    data = OceanImageDataset(num=1)
    train_loader = DataLoader(data, batch_size=1, shuffle=True)

    original_tensor = train_loader.dataset[0][0]
    original_shape = original_tensor.shape

    input_tensor = revert_png(image_path,
                              original_shape=original_shape,
                              compare_to=original_tensor)

    mask = Image.open(land_mask_path).convert('L')
    land_mask_tensor = np.array(mask, dtype=np.float32)

    plot_tensor(input_tensor, land_mask_tensor)


def plot_tensor(input_tensor, land_mask_tensor=None, filename = "results/plot.png"):
    """
    Plots a tensor. May still be useful
    :param input_tensor: (3, 94, 44), where third channel is binary representation of land mask
    :param land_mask_tensor: Optional land mask tensor, default is third channel
    """

    if land_mask_tensor is not None:
        mask = land_mask_tensor.T
        mask = np.fliplr(mask)
    else:
        mask = input_tensor[2].T

    missing_pixel_mask = mask > 0
    missing_pixel_mask = ~missing_pixel_mask

    masked_image = input_tensor.clone()
    for i in range(3):
        masked_image[i][missing_pixel_mask.T] = np.nan

    u_component = masked_image[0].T
    v_component = masked_image[1].T

    x_coords = np.arange(u_component.shape[0])
    y_coords = np.arange(u_component.shape[1])

    plotQuiverData(x_coords,  # (94,)
                   y_coords,  # (44,)
                   u_component,  # ([94, 44])
                   v_component,  # ([94, 44])
                   quiver_stride=3)
    plt.savefig(filename)


# Example Usage
# Load & plot tensor
# data = OceanImageDataset(num=1)
# train_loader = DataLoader(data, batch_size=1, shuffle=True)
# original_tensor = train_loader.dataset[0][0]
#
# plot_tensor(original_tensor)

# Plot PNG
# plot_png('../../plots/images/gp_image.png',
#          '../../plots/images/land_mask_cropped.png')
