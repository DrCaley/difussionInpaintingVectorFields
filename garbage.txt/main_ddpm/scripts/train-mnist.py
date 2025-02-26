import random
import numpy as np
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Lambda
from torchvision.datasets import FashionMNIST

from medium_ddpm.dir.ddpm import MyDDPM
from medium_ddpm.dir.resize_tensor import resize
from medium_ddpm.dir.unets.unet_resized_2_channel import MyUNet
from medium_ddpm.dir.util import show_images, generate_new_images

# Setting reproducibility
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Definitions
STORE_PATH_FASHION = "ddpm_model_fashion.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_steps, min_beta, max_beta = 1000, 0.0001, 0.02
ddpm = MyDDPM(MyUNet(n_steps), n_steps=n_steps, min_beta=min_beta, max_beta=max_beta, device=device)

no_train = False
batch_size = 64
n_epochs = 20
lr = 0.001


class ResizeTransform:
    def __init__(self, end_shape):
        self.end_shape = end_shape

    def __call__(self, tensor):
        return resize(tensor, self.end_shape)


# Initialize dataset with transformations
transform = Compose([
    ToTensor(),
    Lambda(lambda x: (x - 0.5) * 2),  # Normalize to range [-1, 1]
    ResizeTransform((1, 64, 128))  # Resized to (1, 64, 128)
])

# Select dataset and initialize DataLoader
store_path = "../../../models/ddpm_fashion.pt"
ds_fn = FashionMNIST
dataset = ds_fn("../datasets", download=True, train=True, transform=transform)
loader = DataLoader(dataset, batch_size, shuffle=True)


# Training loop function
def training_loop(ddpm, loader, n_epochs, optim, device, display=False, store_path="ddpm_model.pt"):
    mse = nn.MSELoss()
    best_loss = float("inf")
    n_steps = ddpm.n_steps

    for epoch in tqdm(range(n_epochs), desc="Training progress", colour="#00ff00"):
        epoch_loss = 0.0
        for step, batch in enumerate(tqdm(loader, leave=False, desc=f"Epoch {epoch + 1}/{n_epochs}", colour="#005500")):
            x0 = batch[0].to(device)  # Get batch of images and move to device
            n = len(x0)

            eta = torch.randn_like(x0).to(device)  # Generate noise
            t = torch.randint(0, n_steps, (n,)).to(device)  # Random time steps

            noisy_imgs = ddpm(x0, t, eta)  # Forward process: create noisy images

            eta_theta = ddpm.backward(noisy_imgs, t.reshape(n, -1))  # Predict noise

            loss = mse(eta_theta, eta)  # Compute MSE loss
            optim.zero_grad()
            loss.backward()
            optim.step()

            epoch_loss += loss.item() * len(x0) / len(loader.dataset)

        # Display images generated at this epoch
        if display:
            show_images(generate_new_images(ddpm, device=device), f"Images generated at epoch {epoch + 1}")

        log_string = f"Loss at epoch {epoch + 1}: {epoch_loss:.3f}"

        # Save the best model
        if best_loss > epoch_loss:
            best_loss = epoch_loss
            torch.save(ddpm.state_dict(), store_path)
            log_string += " --> Best model ever (stored)"

        print(log_string)


# Run training loop if not skipping training
if not no_train:
    training_loop(ddpm, loader, n_epochs, optim=Adam(ddpm.parameters(), lr), device=device, store_path=store_path)
