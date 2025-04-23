import math
import os
import random

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, Lambda
from tqdm import tqdm

from dataloaders.dataloader import OceanImageDataset
from medium_ddpm.ddpm import MyDDPM
from medium_ddpm.resize_tensor import resize,ResizeTransform
from medium_ddpm.unets.unet_xl import MyUNet
from medium_ddpm.dir.util import show_images, generate_new_images

"""This file trains the most successful model as of Feb 2025."""

# Setting reproducibility
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Definitions
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_steps, min_beta, max_beta = 1000, 0.0001, 0.02
ddpm = MyDDPM(MyUNet(n_steps), n_steps=n_steps, min_beta=min_beta, max_beta=max_beta, device=device)

no_train = False
batch_size = 35
n_epochs = 100
lr = 0.001



# Initialize dataset with transformations
# minU, maxU = -0.8973235906436031, 1.0859991093945718
# minV, maxV = -0.6647028130174489, 0.5259408400292674
min = -0.8973235906436031 * 1.2
max = 1.0859991093945718 * 1.2
transform = Compose([
    Lambda(lambda x: (x - min) / (max - min) * 2),     # Normalize to range [-1, 1]
    ResizeTransform((2, 64, 128))        # Resized to (2, 64, 128)
])

store_path = "./ddpm_ocean_v0.pt"

data = OceanImageDataset(
    mat_file="../../data/rams_head/stjohn_hourly_5m_velocity_ramhead_v2.mat",
    boundaries="../../../data/rams_head/boundaries.yaml",
    num=100,
    transform=transform
)

train_len = int(math.floor(len(data) * 0.7))
test_len = int(math.floor(len(data) * 0.15))
val_len = len(data) - train_len - test_len

training_data, test_data, validation_data = random_split(data, [train_len, test_len, val_len])

train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size)
val_loader = DataLoader(validation_data, batch_size=batch_size)


class CustomLoss(nn.Module):
    def __init__(self, non_zero_weight=5.0, zero_weight=1.0):
        super(CustomLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.mae = nn.L1Loss(reduction='none')
        self.non_zero_weight = non_zero_weight
        self.zero_weight = zero_weight

    def forward(self, eta_theta, eta, x0):
        non_zero_mask = (x0 != 0).float()
        zero_mask = (x0 == 0).float()

        mse_loss = self.mse(eta_theta, eta)
        mae_loss = self.mae(eta_theta, eta)

        weighted_mse_loss = mse_loss * (self.non_zero_weight * non_zero_mask + self.zero_weight * zero_mask)
        weighted_mae_loss = mae_loss * (self.non_zero_weight * non_zero_mask + self.zero_weight * zero_mask)

        weighted_mse_loss = weighted_mse_loss.sum() / (non_zero_mask.sum() * self.non_zero_weight + zero_mask.sum() * self.zero_weight)
        weighted_mae_loss = weighted_mae_loss.sum() / (non_zero_mask.sum() * self.non_zero_weight + zero_mask.sum() * self.zero_weight)

        return weighted_mse_loss + weighted_mae_loss


def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0.0
    count = 0
    criterion = nn.MSELoss()

    with torch.no_grad():
        for batch in data_loader:
            x0 = batch[0].to(device).float()
            n = len(x0)

            eta = torch.randn_like(x0).to(device)
            t = torch.randint(0, model.n_steps, (n,)).to(device)

            noisy_imgs = model(x0, t, eta)
            eta_theta = model.backward(noisy_imgs, t.reshape(n, -1))
            loss = criterion(eta_theta, eta)
            total_loss += loss.item() * n
            count += n

    return total_loss / count


def training_loop(ddpm, train_loader, test_loader, n_epochs, optim, device, display=False, store_path="ddpm_ocean_model.pt"):
    """Trains the xl model (1 more layer than the original). The xl model is the main one we use as of early August, 2024"""

    custom_loss = CustomLoss()
    best_train_loss = float("inf")
    best_test_loss = float("inf")
    n_steps = ddpm.n_steps

    train_losses = []
    test_losses = []

    start_epoch = 0
    if os.path.exists(store_path):
        checkpoint = torch.load(store_path)
        ddpm.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        train_losses = checkpoint['train_losses']
        test_losses = checkpoint['test_losses']
        best_train_loss = checkpoint['best_train_loss']
        best_test_loss = checkpoint['best_test_loss']
        print(f"Resuming training from epoch {start_epoch}")

    for epoch in tqdm(range(start_epoch, start_epoch + n_epochs), desc="Training progress", colour="#00ff00"):
        epoch_loss = 0.0
        ddpm.train()
        for step, batch in enumerate(tqdm(train_loader, leave=False, desc=f"Epoch {epoch + 1}/{n_epochs}", colour="#005500")):
            x0 = batch[0].to(device).float()
            n = len(x0)

            eta = torch.randn_like(x0).to(device)  # Generate noise
            t = torch.randint(0, n_steps, (n,)).to(device)  # Random time steps

            noisy_imgs = ddpm(x0, t, eta)
            eta_theta = ddpm.backward(noisy_imgs, t.reshape(n, -1))

            loss = custom_loss(eta_theta, eta, x0)
            optim.zero_grad()
            loss.backward()
            optim.step()

            epoch_loss += loss.item() * len(x0) / len(train_loader.dataset)

        ddpm.eval()
        avg_train_loss = evaluate(ddpm, train_loader, device)
        avg_test_loss = evaluate(ddpm, test_loader, device)
        ddpm.train()

        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)

        log_string = f"Loss at epoch {epoch + 1}: {epoch_loss:.3f}"

        if best_test_loss > avg_test_loss:
            best_test_loss = avg_test_loss
            torch.save(ddpm.state_dict(), store_path)
            log_string += " --> Best model ever (stored based on test loss)"

        print(log_string)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': ddpm.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'train_losses': train_losses,
            'test_losses': test_losses,
            'best_train_loss': best_train_loss,
            'best_test_loss': best_test_loss
        }
        torch.save(checkpoint, store_path)

        if display:
            show_images(generate_new_images(ddpm, device=device), f"Images generated at epoch {epoch + 1}")

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Test Loss')
    plt.savefig('train_test_loss_xl.png')


optimizer = Adam(ddpm.parameters(), lr=lr)

if not no_train:
    training_loop(ddpm, train_loader, test_loader, n_epochs, optim=optimizer, device=device, store_path=store_path)
