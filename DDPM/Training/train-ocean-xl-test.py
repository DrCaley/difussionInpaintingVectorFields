import math
import os
import sys
import random
import yaml

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn, Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, Lambda
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from DataPrep.ocean_image_dataset import OceanImageDataset
from DDPM.Neural_Networks.ddpm import MyDDPM
from DDPM.Helper_Functions.resize_tensor import resize_transform
from DDPM.Helper_Functions.standardize_data import standardize_data
from DDPM.Neural_Networks.unets.unet_xl import MyUNet
from gaussian_process.incompressible_gp.adding_noise.divergence_free_noise import divergence_free_noise
from gaussian_process.incompressible_gp.adding_noise.compute_divergence import compute_divergence

# from medium_ddpm.dir.util import show_images, generate_new_images

"""
This file is being used to train the best model of all time baybee.
There's never been a model better than this one, we got he best epsilons, 
our loss function becomes our win function, truly remarkable stuff. 
Absolute model 
    | o |
    \ | /
      |
     / \
    |   |
"""

using_dumb_pycharm = True
# Load the YAML file
try:
    with open('../../data.yaml', 'r') as file: ## <- if you are running it on pycharm
        config = yaml.safe_load(file)
    print ("Why are u sing pycharm??")
except FileNotFoundError:
    using_dumb_pycharm = False # <-- congrats on NOT using that dumb IDE!
    print("I see you are using the Terminal")
    with open('data.yaml', 'r') as file: ## <-- if you are running it on the terminal
        config = yaml.safe_load(file)

# Setting reproducibility
SEED = config['testSeed']
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Definitions
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("we are running on the:", device)
n_steps, min_beta, max_beta = 1000, 0.0001, 0.02
ddpm = MyDDPM(MyUNet(n_steps), n_steps=n_steps, min_beta=min_beta, max_beta=max_beta, device=device)

training_mode = True
batch_size = config['batch_size']
n_epochs = config['n_epochs']
lr = config['lr']


transform = Compose([
    resize_transform((2, 64, 128)),        # Resized to (2, 64, 128)
    standardize_data(config['u_training_mean'], config['u_training_std'], config['v_training_mean'], config['v_training_std'])
])

store_path = "./ddpm_ocean_v0.pt"

if using_dumb_pycharm :
    data = OceanImageDataset(
        mat_file="../../data/rams_head/stjohn_hourly_5m_velocity_ramhead_v2.mat", # <--
        boundaries="../../data/rams_head/boundaries.yaml",
        num=100,
        transform=transform
    )
else:
    data = OceanImageDataset(

        mat_file="data/rams_head/stjohn_hourly_5m_velocity_ramhead_v2.mat", # <--
        boundaries="data/rams_head/boundaries.yaml",
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

def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0.0
    count = 0
    # KEEP THIS LINE BELOW (or not, idk) - Matt
    criterion = nn.MSELoss()

    with torch.no_grad():
        for batch in data_loader:
            x0 = batch[0].to(device).float()
            n = len(x0)

            epsilon = torch.randn_like(x0).to(device)
            t = torch.randint(0, model.n_steps, (n,)).to(device)

            noisy_imgs = model(x0, t, epsilon)
            epsilon_theta = model.backward(noisy_imgs, t.reshape(n, -1))
            loss = criterion(epsilon_theta, epsilon)
            total_loss += loss.item() * n
            count += n

    return total_loss / count

# ChatGPT, should probably check - Matt
def physical_loss(predicted: Tensor) -> Tensor:
    """
    Computes the mean squared divergence across a batch of predicted vector fields.
    `predicted` shape: (batch_size, 2, H, W) — where 2 corresponds to (u,v).
    """
    batch_divs = []
    for field in predicted:
        u, v = field[0], field[1]  # Get components
        div = compute_divergence(u, v)  # Shape (H, W)
        batch_divs.append(div.pow(2).mean())  # MSE of divergence for one field
    return torch.stack(batch_divs).mean()


def training_loop(ddpm, train_loader, test_loader, n_epochs, optim, device, display=False, store_path="ddpm_ocean_model.pt"):
    """Trains the xl model (1 more layer than the original). The xl model is the main one we use as of early August, 2024"""
    # building loss function
    w1 = 0.5
    w2 = 0.5
    loss_function = nn.MSELoss()

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
        # may be able to throw out "step" in "for step, batch in ... "
        for step, batch in enumerate(tqdm(train_loader, leave=False, desc=f"Epoch {epoch + 1}/{n_epochs}", colour="#005500")):
            x0 = batch[0].to(device).float()
            n = len(x0)

            t = torch.randint(0, n_steps, (n,)).to(device)  # Random time steps
            epsilon = divergence_free_noise(x0, t,device).to(device)  # Generate noise

            noisy_imgs = ddpm(x0, t, epsilon)
            predicted_value = ddpm.backward(noisy_imgs, t.reshape(n, -1))

            # Hacky, should refactor to put all definitions in same place I think - Matt
            loss = w1 * loss_function(predicted_value, epsilon) + w2 * physical_loss(predicted_value)

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

        """
        if display:
            show_images(generate_new_images(ddpm, device=device), f"Images generated at epoch {epoch + 1}")
        """

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Test Loss')
    plt.savefig('train_test_loss_xl.png')


optimizer = Adam(ddpm.parameters(), lr=lr)

if training_mode:
    training_loop(ddpm, train_loader, test_loader, n_epochs, optim=optimizer, device=device, store_path=store_path)
