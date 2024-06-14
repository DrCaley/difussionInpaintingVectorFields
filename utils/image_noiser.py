import os
import numpy as np
import skimage
from skimage.util import random_noise
import matplotlib.pyplot as plt
import torch


def save_noisy_image(tensor, mode, iteration, output_dir):
    filename = os.path.join(output_dir, f'noisy_image_{mode}_iteration_{iteration}.png')
    skimage.io.imsave(filename, (tensor * 255).astype('uint8'))


def plot_noise(tensor, mode, iteration, row, col, i):
    plt.subplot(row, col, i)
    plt.imshow(np.clip(tensor.numpy(), 0, 1), aspect='auto')
    plt.title(f'{mode} : Iteration {iteration}')
    plt.axis("off")


def generate_noised_tensor_iterative(tensor, iteration, variance, mode="gaussian"):
    """
    Applies noise to a given tensor for a specified number of iterations using
    skimage.util.random_noise()

    :param tensor: A tensor representing an image at a point in time
    :param iteration: The number of iterations for which noise will be applied.
    :param variance: The variance of the noise to be added per iteration.
                     Higher values result in noisier images.
    :param mode: The type of noise to be applied. Default is "gaussian".

    :return: A noised tensor applied with variance * iteration amount of noise
    """
    noisy_tensor = tensor.clone().numpy()
    for i in range(iteration):
        noisy_tensor = random_noise(noisy_tensor, mode, var=variance, clip=False)
    return torch.tensor(noisy_tensor)


def generate_noised_tensor_single_step(tensor, target_iteration, var_per_iteration, mode="gaussian"):
    """
    Applies noise to a given tensor for a specified number of iterations,
    simulating the noise accumulation over time using skimage.util.random_noise()

    :param tensor: A tensor representing an image at a point in time
    :param target_iteration: The number of iterations for which noise will be applied.
    :param var_per_iteration: The variance of the noise to be added per iteration.
                              Higher values result in noisier images.
    :param mode: The type of noise to be applied. Default is "gaussian".
    """
    total_var = var_per_iteration * target_iteration
    noisy_tensor = random_noise(tensor.numpy(), mode, var=total_var, clip=False)
    return torch.tensor(noisy_tensor)


# def main():
#     image_path = "./data/images/ocean_image0.png"
#     try:
#         image = skimage.io.imread(image_path) / 255.0
#     except FileNotFoundError:
#         print(f"Image file not found at path: {image_path}")
#         return
#     image_tensor = torch.tensor(image, dtype=torch.float32)
#
#     target_iteration = 1000
#     var_per_iteration = 0.005
#
#     noisy_tensor_iterative = generate_noised_tensor_iterative(image_tensor, iteration=target_iteration,
#                                                               variance=var_per_iteration)
#     noisy_tensor_single_step = generate_noised_tensor_single_step(image_tensor, target_iteration=target_iteration,
#                                                                   var_per_iteration=var_per_iteration)
#
#     fig, axes = plt.subplots(1, 3, figsize=(15, 5))
#     axes[0].imshow(np.clip(image_tensor.numpy(), 0, 1))
#     axes[0].set_title('Original Image')
#     axes[1].imshow(np.clip(noisy_tensor_iterative.numpy(), 0, 1))
#     axes[1].set_title(f'Iterative Noise (Iteration {target_iteration})')
#     axes[2].imshow(np.clip(noisy_tensor_single_step.numpy(), 0, 1))
#     axes[2].set_title(f'Single Step Noise (Iteration {target_iteration})')
#
#     for ax in axes:
#         ax.axis('off')
#
#     plt.show()
#
#
# if __name__ == "__main__":
#     main()
