import os
import skimage
import matplotlib.pyplot as plt


def add_noise(img, mode, base_var=0.0005):
    return skimage.util.random_noise(img, mode=mode, var=base_var)


def save_noisy_image(img, mode, iteration, output_dir):
    filename = os.path.join(output_dir, f'noisy_image_{mode}_iteration_{iteration}.png')
    skimage.io.imsave(filename, (img * 255).astype('uint8'))


def plot_noise(img, mode, iteration, row, col, i):
    plt.subplot(row, col, i)
    plt.imshow(img, aspect='auto')
    plt.title(f'{mode} : Iteration {iteration}')
    plt.axis("off")


def generate_noised_images(img, mode="gaussian", total_iterations=1000, display_iterations=7, save_final=False,
                           output_dir='./'):
    interval = total_iterations // display_iterations

    plt.figure(figsize=(18, 12))
    row = 2
    col = display_iterations // row + 1

    noisy_img = img
    # img with no noise
    plot_noise(img, mode, 0, row, col, 1)
    # save_noisy_image(noisy_img, mode, 0, output_dir)

    # noise images
    for i in range(1, display_iterations + 1):
        for _ in range(interval):
            noisy_img = add_noise(noisy_img, mode)
        plot_noise(noisy_img, mode, i * interval, row, col, i + 1)
        # save_noisy_image(noisy_img, mode, i * interval, output_dir)

    plt.tight_layout()
    plt.show()

    # if save_final:
    #     final_noisy_img_path = os.path.join(output_dir, "final_noisy_image.png")
    #     skimage.io.imsave(final_noisy_img_path, (noisy_img * 255).astype('uint8'))

    return noisy_img


def generate_noised_images(img, mode="gaussian", total_iterations=1000):
    interval = total_iterations // 100
    noisy_img = img.copy()

    for i in range(total_iterations):
        noisy_img = add_noise(noisy_img, mode)
        # save_noisy_image(noisy_img, mode, i * interval, output_dir='./noisy_images')

