import math
import random
import numpy as np
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, Lambda

from dataloaders.dataloader import OceanImageDataset
from medium_ddpm.dir.ddpm import MyDDPM
from medium_ddpm.dir.resize_tensor import resize
from medium_ddpm.dir.unet_resized import MyUNet
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
        return resize(tensor, self.end_shape).float()


# Initialize dataset with transformations
transform = Compose([
    Lambda(lambda x: (x - 0.5) * 2),  # Normalize to range [-1, 1]
    ResizeTransform((1, 64, 128))
])

store_path = "../../../models/ddpm_ocean_v0.pt"

data = OceanImageDataset(
    mat_file="../../../data/rams_head/stjohn_hourly_5m_velocity_ramhead_v2.mat",
    boundaries="../../data/rams_head/boundaries.yaml",
    num=10,
    transform=transform
)

train_len = int(math.floor(len(data) * 0.7))
test_len = int(math.floor(len(data) * 0.15))
val_len = len(data) - train_len - test_len

training_data, test_data, validation_data = random_split(data, [train_len, test_len, val_len])

loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size)
val_loader = DataLoader(validation_data, batch_size=batch_size)


# Training loop function
def training_loop(ddpm, loader, n_epochs, optim, device, display=False, store_path="ddpm_ocean_model.pt"):
    mse = nn.MSELoss()
    best_loss = float("inf")
    n_steps = ddpm.n_steps

    for epoch in tqdm(range(n_epochs), desc="Training progress", colour="#00ff00"):
        epoch_loss = 0.0
        ddpm.train()
        for step, batch in enumerate(tqdm(loader, leave=False, desc=f"Epoch {epoch + 1}/{n_epochs}", colour="#005500")):
            x0 = batch[0].to(device).float()
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


        if display:
            show_images(generate_new_images(ddpm, device=device), f"Images generated at epoch {epoch + 1}")

        log_string = f"Loss at epoch {epoch + 1}: {epoch_loss:.3f}"

        # Save the best model
        if best_loss > epoch_loss:
            best_loss = epoch_loss
            torch.save(ddpm.state_dict(), store_path)
            log_string += " --> Best model ever (stored)"

        print(log_string)


# Create optimizer and scheduler
optimizer = Adam(ddpm.parameters(), lr=lr)

# Run training loop if not skipping training
if not no_train:
    training_loop(ddpm, loader, n_epochs, optim=optimizer, device=device, store_path=store_path)


def evaluate_model(ddpm, loader, device, description):
    ddpm.eval()
    mse = nn.MSELoss()
    total_loss = 0.0
    with torch.no_grad():
        for step, batch in enumerate(tqdm(loader, leave=False, desc=description, colour="#0000ff")):
            x0 = batch[0].to(device).float()
            n = len(x0)

            eta = torch.randn_like(x0).to(device)
            t = torch.randint(0, n_steps, (n,)).to(device)

            noisy_imgs = ddpm(x0, t, eta)
            eta_theta = ddpm.backward(noisy_imgs, t.reshape(n, -1))

            loss = mse(eta_theta, eta)
            total_loss += loss.item() * len(x0) / len(loader.dataset)

    print(f"{description} Loss: {total_loss:.3f}")

# evaluate_model(ddpm, val_loader, device, "Validation")
# evaluate_model(ddpm, test_loader, device, "Testing")
