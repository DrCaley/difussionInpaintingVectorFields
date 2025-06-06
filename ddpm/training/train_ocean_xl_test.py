import math
import os
import pickle
import sys
import random
import yaml
import csv
import numpy as np
import torch

from datetime import datetime
from matplotlib import pyplot as plt
from torch import nn, Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, Lambda
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from data_prep.ocean_image_dataset import OceanImageDataset
from ddpm.neural_networks.ddpm import MyDDPM
from ddpm.helper_functions.resize_tensor import resize_transform
from ddpm.helper_functions.standardize_data import standardize_data
from ddpm.neural_networks.unets.unet_xl import MyUNet
from ddpm.helper_functions.loss_functions import CustomLoss
from noising_process.incompressible_gp.adding_noise.divergence_free_noise import divergence_free_noise, gaussian_each_step_divergence_free_noise

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

timestamp = datetime.now().strftime("%h%d_%H%M")
output_dir = os.path.join(os.path.dirname(__file__), f"training_output")
os.makedirs(output_dir, exist_ok=True)
csv_file = os.path.join(output_dir, f"training_log_{timestamp}.csv")
plot_file = os.path.join(output_dir, f"train_test_loss_xl_{timestamp}.png")
model_file = os.path.join(output_dir, f"ddpm_ocean_model_{timestamp}.pt")

# CHANGE DESCRIPTION HERE, IT WILL ATTACH TO THE OUTPUT CSV:
description = 'Using 0 physical loss, 1 MSE along with non divergent noise that has the gaussian applied at each step'

using_dumb_pycharm = True
# Load the YAML file
if os.path.exists('../../data.yaml'):
    with open('../../data.yaml', 'r') as file: ## <- if you are running it on pycharm
        config = yaml.safe_load(file)
    print ("--> ALL HAIL PYCHARM!!!! PYCHARM IS THE BEST <--")
else :
    using_dumb_pycharm = False # <-- congrats on NOT using that dumb IDE!
    print("I see you are using the Terminal")
    with open('data.yaml', 'r') as file: ## <-- if you are running it on the terminal
        config = yaml.safe_load(file)

# Load the pickle
if using_dumb_pycharm :
    with open('../../data.pickle', 'rb') as f:
        training_data_np, validation_data_np, test_data_np = pickle.load(f)
else:
    with open('data.pickle', 'rb') as f:
        training_data_np, validation_data_np, test_data_np = pickle.load(f)

training_tensor = torch.from_numpy(training_data_np).float()
validation_tensor = torch.from_numpy(validation_data_np).float()
test_tensor = torch.from_numpy(test_data_np).float()

# Setting reproducibility
SEED = config['testSeed']
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Definitions
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("we are running on the:", device)
n_steps, min_beta, max_beta = config['n_steps'], config['min_beta'], config['max_beta']
ddpm = MyDDPM(MyUNet(n_steps), n_steps=n_steps, min_beta=min_beta, max_beta=max_beta, device=device)

training_mode = config['training_mode']
batch_size = config['batch_size']
n_epochs = config['n_epochs']
lr = config['lr']

transform = Compose([
    resize_transform((2, 64, 128)),        # Resized to (2, 64, 128)
    standardize_data(config['u_training_mean'], config['u_training_std'], config['v_training_mean'], config['v_training_std'])
])

# This is nasty
if using_dumb_pycharm :
    training_data = OceanImageDataset(
        data_tensor=training_tensor,
        boundaries="../../data/rams_head/boundaries.yaml",
        transform=transform,
        max_size=40
    )
    test_data = OceanImageDataset(
        data_tensor=test_tensor,
        boundaries="../../data/rams_head/boundaries.yaml",
        transform=transform,
        max_size=40
    )
    validation_data = OceanImageDataset(
        data_tensor=validation_tensor,
        boundaries="../../data/rams_head/boundaries.yaml",
        transform=transform
    )
else:
    training_data = OceanImageDataset(
        data_tensor=training_tensor,
        boundaries="data/rams_head/boundaries.yaml",
        transform=transform,
        max_size=40
    )
    test_data = OceanImageDataset(
        data_tensor=test_tensor,
        boundaries="data/rams_head/boundaries.yaml",
        transform=transform,
        max_size=40
    )
    validation_data = OceanImageDataset(
        data_tensor=validation_tensor,
        boundaries="data/rams_head/boundaries.yaml",
        transform=transform,
        max_size=40
    )

train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size)
val_loader = DataLoader(validation_data, batch_size=batch_size)

def evaluate(model, data_loader, device):
    """
        Evaluates a trained diffusion model using Mean Squared Error (MSE) loss.

        This function computes the average MSE loss over all batches in the given data loader.
        It performs a forward and backward diffusion process to compare the predicted noise with the true noise.

        Args:
           model (MyDDPM): The trained DDPM model.
           data_loader (DataLoader): PyTorch DataLoader containing the dataset to evaluate (train, test, or val).
           device (torch.device): The device to run the model on ('cuda' or 'cpu').

        Returns:
           float: The average loss across the dataset.
   """
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


def training_loop(ddpm, train_loader, test_loader, n_epochs, optim, device, display=False, loss_function = None):
    """
        Trains the xl model (1 more layer than the original). The xl model is the main one we use as of early August, 2024
        Trains a DDPM model over a specified number of epochs.

        This function handles loading from a checkpoint (if available), training the model using a custom
        or default loss function, logging losses to CSV, saving the best model by test loss, and optionally plotting loss curves.

        Args:
            ddpm (MyDDPM): The diffusion model to train.
            train_loader (DataLoader): PyTorch DataLoader for the training data.
            test_loader (DataLoader): PyTorch DataLoader for the test data.
            n_epochs (int): Number of epochs to train for.
            optim (torch.optim.Optimizer): The optimizer for training.
            device (torch.device): The device to run the model on ('cuda' or 'cpu').
            display (bool, optional): Whether to display/generated images per epoch (not currently implemented). Defaults to False.
            loss_function (Callable, optional): The loss function to use (e.g. nn.MSELoss). Required.

        Side Effects:
            - Writes a CSV file with training and test loss history.
            - Saves model checkpoints to disk.
            - Updates global model and optimizer states.
            - Plots and saves a loss curve PNG file after training completes.
    """
    best_train_loss = float("inf")
    best_test_loss = float("inf")
    n_steps = ddpm.n_steps

    train_losses = []
    test_losses = []

    start_epoch = 0
    if os.path.exists(model_file):
        checkpoint = torch.load(model_file)
        ddpm.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        train_losses = checkpoint['train_losses']
        test_losses = checkpoint['test_losses']
        best_train_loss = checkpoint['best_train_loss']
        best_test_loss = checkpoint['best_test_loss']
        print(f"Resuming training from epoch {start_epoch}")

    # CSV output setup
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([description])
        writer.writerow(['Epoch', 'Train Loss', 'Test Loss'])

    # Training arc (the ankle weights are coming off)
    for epoch in tqdm(range(start_epoch, start_epoch + n_epochs), desc="training progress", colour="#00ff00"):
        epoch_loss = 0.0
        ddpm.train()
        # may be able to throw out "step" in "for step, batch in ... "
        for step, batch in enumerate(tqdm(train_loader, leave=False, desc=f"Epoch {epoch + 1}/{n_epochs}", colour="#005500")):
            x0 = batch[0].to(device).float()
            n = len(x0)

            t = torch.randint(0, n_steps, (n,)).to(device)  # Random time steps
            noise = torch.randn_like(x0).to(device)  # Generate noise

            noisy_imgs = ddpm(x0, t, noise)
            predicted_noise = ddpm.backward(noisy_imgs, t.reshape(n, -1))

            loss = loss_function(predicted_noise, noise)

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

        log_string = f"\nepoch {epoch + 1}: \n" + f"Loss: {epoch_loss:.3f}"

        #TODO: Sanity check each losses above

        # Append current epoch results to CSV
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, avg_train_loss, avg_test_loss])

        if best_test_loss > avg_test_loss:
            best_test_loss = avg_test_loss
            torch.save(ddpm.state_dict(), model_file)
            log_string += " --> Best model ever (stored based on test loss)"

        log_string += (f"\nAverage test loss: {avg_test_loss:.3f} -> best: {best_test_loss:.3f}\n"
                       + f"Average train loss: {avg_train_loss:.3f}")

        tqdm.write(log_string)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': ddpm.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'train_losses': train_losses,
            'test_losses': test_losses,
            'best_train_loss': best_train_loss,
            'best_test_loss': best_test_loss
        }
        torch.save(checkpoint, model_file)

        """
        if display:
            show_images(generate_new_images(ddpm, device=device), f"Images generated at epoch {epoch + 1}")
        """

    plt.figure(figsize=(20, 10))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('training and Test Loss')
    plt.savefig(plot_file)

optimizer = Adam(ddpm.parameters(), lr=lr)

if training_mode:
    training_loop(ddpm, train_loader, test_loader, n_epochs, optim=optimizer, device=device, loss_function=nn.MSELoss())
