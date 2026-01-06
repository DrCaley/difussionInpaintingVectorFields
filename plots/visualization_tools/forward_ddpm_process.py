import numpy as np
import torch
import matplotlib.pyplot as plt
import imageio
import io
import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from data_prep.data_initializer import DDInitializer

dd = DDInitializer()

def make_ddpm_vector_field_gif(alpha_bars, x0_orginal, noise_strategy, gif_path="ddpm_vectorfield_1.gif",
                                mag_path="avg_value_plot_1.png", var_path="variance_plot_1.png", every=1):
    """
    Create a gif visualizing the ddpm forward process on vector fields.
    Also plots average magnitude and variance of the vectors over time.

    x0 shape: (1, 2, H, W)
    noise_strategy: function taking (x0, t) and returning noise of shape (1, 2, H, W)
    """

    # Convert to numpy if tensor
    x0 = x0_orginal.cpu().numpy() if hasattr(x0_orginal, "cpu") else x0_orginal
    x0 = x0.transpose(1, 2, 0)  # (H, W, 2)

    frames = []
    avg_mags = []
    var_mags = []

    H, W, _ = x0.shape
    Y, X = np.mgrid[0:H, 0:W]

    steps = list(range(0, len(alpha_bars), every))
    print(f"Creating GIF with {len(steps)} frames...")

    for t in steps:
        t_tensor = torch.tensor([t])
        a_sqrt = np.sqrt(alpha_bars[t])
        one_minus_a_sqrt = np.sqrt(1 - alpha_bars[t])

        epsilon = noise_strategy(torch.unsqueeze(x0_orginal, 0), t_tensor)
        epsilon = epsilon.cpu().numpy() if hasattr(epsilon, "cpu") else epsilon
        epsilon = epsilon[0].transpose(1, 2, 0)

        noisy = a_sqrt * x0 + one_minus_a_sqrt * epsilon

        avg_mag = np.nanmean(noisy)
        var_mag = np.nanvar(noisy)

        avg_mags.append(avg_mag)
        var_mags.append(var_mag)

        # Visualization
        fig, axs = plt.subplots(1, 2, figsize=(2 * W / 10, H / 10))
        fig.suptitle(f"ddpm Forward Noising - Step {t}")

        axs[0].quiver(X, Y, x0[..., 0], x0[..., 1], color='blue', scale=20)
        axs[0].set_title("Original x₀")
        axs[0].axis('off')

        axs[1].quiver(X, Y, noisy[..., 0], noisy[..., 1], color='blue', scale=20)
        axs[1].set_title(f"Noisy t={t}")
        axs[1].axis('off')

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        frame = imageio.v2.imread(buf)
        frames.append(frame)

    # Save gif
    imageio.mimsave(gif_path, frames, duration=0.2, loop=0)
    print(f"Saved ddpm vector field forward noising GIF to {gif_path}")

    # Plot average magnitude
    plt.figure(figsize=(8, 4))
    plt.plot(steps, avg_mags, label="Average Value", color='purple', marker='o')
    plt.title("Average Value Over Time")
    plt.xlabel("Timestep t")
    plt.ylabel("Average Value")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(mag_path)
    plt.close()
    print(f"Saved average value plot to {mag_path}")

    # Plot variance
    plt.figure(figsize=(8, 4))
    plt.plot(steps, var_mags, label="Variance", color='orange', marker='o')
    plt.title("Variance of Vector Values Over Time")
    plt.xlabel("Timestep t")
    plt.ylabel("Variance")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(var_path)
    plt.close()
    print(f"Saved variance plot to {var_path}")


# Example usage
alpha_bars = dd.get_alpha_bars()
standardizer = dd.get_standardizer()
x0 = standardizer(dd.training_tensor[::, ::, ::].permute(3, 2, 1, 0)[0])
make_ddpm_vector_field_gif(alpha_bars, x0, dd.get_noise_strategy(), every=5)
