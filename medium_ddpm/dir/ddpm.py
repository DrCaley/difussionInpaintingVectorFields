import torch
import torch.nn as nn


class MyDDPM(nn.Module):
    def __init__(self, network, n_steps=200, min_beta=10 ** -4, max_beta=0.02, device=None, image_chw=(1, 64, 128)):
        super(MyDDPM, self).__init__()
        self.n_steps = n_steps
        self.device = device
        # The shape of input images
        self.image_chw = image_chw
        self.network = network.to(device)

        # Linearly spaced tensor of betas (the variance of the Gaussian noise added)
        # from min_beta to max_beta
        self.betas = torch.linspace(min_beta, max_beta, n_steps).to(device)

        # Represents the amount of noise to be removed at each step t during
        # the denoising process
        self.alphas = 1 - self.betas

        # Cumulative products of alphas (alpha_bars) to determine to how much of
        # the noise needs to be removed to progressively denoise the image.
        self.alpha_bars = torch.tensor([torch.prod(self.alphas[:i + 1]) for i in range(len(self.alphas))]).to(device)

    def forward(self, x0, t, eta=None, one_step=False):
        # Extract the shape of the input images
        n, c, h, w = x0.shape

        if one_step:
            a_bar = self.alphas[t]
        else:
            # Get the cumulative product of alphas for the given timestep t
            a_bar = self.alpha_bars[t]

        # If eta (noise) is not provided, generate Gaussian noise
        if eta is None:
            eta = torch.randn(n, c, h, w).to(self.device)

        # Add noise to the image:
        # - Scale the original image by sqrt(a_bar)
        # - Scale the noise by sqrt(1 - a_bar)
        # - Combine the scaled original image and scaled noise to get the noisy image
        noisy = a_bar.sqrt().reshape(n, 1, 1, 1) * x0 + (1 - a_bar).sqrt().reshape(n, 1, 1, 1) * eta

        # Return the noisy image
        return noisy

    def backward(self, x, t):
        # Run each image through the network for each timestep t in the vector t.
        # The network returns its estimation of the noise that was added.
        return self.network(x, t)
