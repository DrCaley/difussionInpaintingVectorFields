from random import randint, sample
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, random_split
import math
from dataloaders.dataloader import OceanImageDataset
from example_nn.inPaintingNetwork import Net
import numpy as np
from utils.resize_tensor import resize
from utils.tensors_to_png import generate_png
from utils.image_noiser import generate_noised_tensor_single_step, generate_noised_tensor_iterative


data = OceanImageDataset(
    mat_file="./data/rams_head/stjohn_hourly_5m_velocity_ramhead_v2.mat",
    boundaries="./data/rams_head/boundaries.yaml",
    num=17040
)

train_len = int(math.floor(len(data) * 0.7))
test_len = int(math.floor(len(data) * 0.15))
val_len = len(data) - train_len - test_len

torch.manual_seed(42)
training_data, test_data, validation_data = random_split(data, [train_len, test_len, val_len])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 15

train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size)
val_loader = DataLoader(validation_data, batch_size=batch_size)

# Sample
model = Net().to(device)
model_path = 'models/model_ep_11.pth'
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

num_samples = 1
image_size = 512
T = 1000
betas = np.linspace(0.0001, 0.02, T)

def generate_samples(model, num_samples, T, betas, shape, device, noised_samples, mask):
    """
    Generate samples from the trained DDPM model.

    Args:
        model: The trained DDPM model.
        num_samples: Number of samples to generate.
        T: Number of diffusion steps.
        betas: Beta values for each timestep.
        shape: Shape of the sample tensor to generate.
        device: Device to run the sampling on.
        noised_samples: Noised input samples.
        mask: Mask indicating known points.

    Returns:
        Generated sample tensor.
    """
    sample = noised_samples.clone().float().to(device)

    for t in reversed(range(T)):
        with torch.no_grad():
            # Model prediction
            predicted_noise = model(sample, mask)

            alpha_t = alphas[t]
            alpha_t_next = alphas[t + 1] if t + 1 < T else 1.0

            # Calculate posterior mean and variance
            mean = (sample - (1 - alpha_t).sqrt() * predicted_noise) / alpha_t.sqrt()
            variance = (1 - alpha_t_next) / (1 - alpha_t) * (1 - alpha_t / alpha_t_next)
            stddev = variance.sqrt()

            # Sample from the posterior distribution
            if t > 0:
                noise = torch.randn_like(sample)
                sample = mean + stddev * noise
            else:
                sample = mean

    return sample


def select_known_samples(tensor, num_known_points):
    """
    Randomly select known samples from the tensor.

    Args:
        tensor: Input tensor.
        num_known_points: Number of known points to select.

    Returns:
        Known samples and mask.
    """
    b, c, h, w = tensor.shape
    total_points = h * w

    known_indices = sample(range(total_points), num_known_points)
    known_indices = [(idx // w, idx % w) for idx in known_indices]

    mask = torch.zeros_like(tensor)
    for i, (y, x) in enumerate(known_indices):
        mask[:, :, y, x] = 1.0

    known_samples = tensor * mask

    return known_samples, mask


for num, (tensor, _) in enumerate(val_loader):
    if num == 0:
        val_tensor = tensor.to(device).float()
        num_known_points = 1000
        known_samples, mask = select_known_samples(val_tensor, num_known_points)

        known_samples = resize(known_samples, (3, 512, 512)).to(device)
        mask = resize(mask, (3, 512, 512)).to(device)

        noised_samples = generate_noised_tensor_iterative(known_samples, T, variance=0.005).float().to(device)

        print(f"Starting to sample {num}/{len(val_loader)} number of validation set")
        samples = generate_samples(model, val_tensor.shape[0], T, betas, val_tensor.shape[1:], device, noised_samples,
                                   mask)

        for i, sample in enumerate(samples):
            generate_png(sample.cpu(), output_path=f'./results/output_{i}.png')
        break
