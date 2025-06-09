import os
import sys
import csv
import torch

from datetime import datetime
from matplotlib import pyplot as plt
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from data_prep.data_initializer import DDInitializer
from ddpm.neural_networks.ddpm_gaussian import MyDDPMGaussian
from ddpm.neural_networks.unets.unet_xl import MyUNet
from ddpm.helper_functions.model_evaluation import evaluate

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
best_model_file = os.path.join(output_dir, f"ddpm_ocean_model_{timestamp}_best.pt")

data_init = DDInitializer()

# CHANGE DESCRIPTION HERE, IT WILL ATTACH TO THE OUTPUT CSV:
description = 'Using 0 physical loss, 1 MSE along with non divergent noise that has the gaussian applied at each step'

n_steps = data_init.get_attribute('n_steps')
min_beta = data_init.get_attribute('min_beta')
max_beta = data_init.get_attribute('max_beta')

ddpm = MyDDPMGaussian(MyUNet(n_steps), n_steps=n_steps, min_beta=min_beta, max_beta=max_beta, device=data_init.get_device())
noise_strategy = data_init.noise_strategy
loss_strategy = data_init.loss_strategy

training_mode = data_init.get_attribute('training_mode')
batch_size = data_init.get_attribute('batch_size')
n_epochs = data_init.get_attribute('n_epochs')
lr = data_init.get_attribute('lr')

training_data = data_init.get_training_data()
test_data = data_init.get_test_data()
validation_data = data_init.get_validation_data()

train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size)
val_loader = DataLoader(validation_data, batch_size=batch_size)

def training_loop(ddpm, train_loader, test_loader, n_epochs, optim, device, display=False, loss_function = None, noise_function = None):
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
            noise_function (Callable, optional): The noise function to use (e.g gaussian). Defaults to Gaussian (torch.randn_like).

        Side Effects:
            - Writes a CSV file with training and test loss history.
            - Saves model checkpoints to disk.
            - Updates global model and optimizer states.
            - Plots and saves a loss curve PNG file after training completes.
    """

    best_test_loss = float("inf")
    n_steps = ddpm.n_steps

    epoch_losses = []
    train_losses = []
    test_losses = []

    start_epoch = 0

    """ 
    TODO: since each model has its own unique file name with timestamp, we are not able to train the same model
    more than once, we need to implement a way to train the same model again. I'm talking to you future me!
    """
    if os.path.exists(model_file):
        checkpoint = torch.load(model_file)
        ddpm.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        epoch_losses = checkpoint['epoch_losses']
        train_losses = checkpoint['train_losses']
        test_losses = checkpoint['test_losses']
        best_test_loss = checkpoint['best_test_loss']
        print(f"Resuming training from epoch {start_epoch}")

    # CSV output setup
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([description])
        writer.writerow(['Epoch', 'Epoch Loss', 'Train Loss', 'Test Loss'])

    # Training arc
    for epoch in tqdm(range(start_epoch, start_epoch + n_epochs), desc="training progress", colour="#00ff00"):
        epoch_loss = 0.0
        ddpm.train()
        for step, batch in enumerate(tqdm(train_loader, leave=False, desc=f"Epoch {epoch + 1}/{n_epochs}", colour="#005500")):
            x0 = batch[0].to(device).float()
            n = len(x0)

            t = torch.randint(0, n_steps, (n,)).to(device)  # Random time steps

            if noise_function is not None: # Generate noise
                noise = noise_function(x0, t).to(device)
            else:
                noise = torch.randn_like(x0).to(device)

            noisy_imgs = ddpm(x0, t, noise)
            predicted_noise = ddpm.backward(noisy_imgs, t.reshape(n, -1))

            loss = loss_function(predicted_noise, noise)

            optim.zero_grad()
            loss.backward()
            optim.step()

            epoch_loss += loss.item() * len(x0) / len(train_loader.dataset)

        # What is all of this doing? Do we want evaluate(...) ONLY, instead of epoch_loss?
        # I figure we may just want to toss epoch_loss.
        ddpm.eval()
        avg_train_loss = evaluate(ddpm, train_loader, device)
        avg_test_loss = evaluate(ddpm, test_loader, device)

        ddpm.train()

        epoch_losses.append(epoch_loss)
        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)

        log_string = f"\nepoch {epoch + 1}: \n" + f"EPOCH Loss: {epoch_loss:.3f}\n" + f"TRAIN Loss: {avg_train_loss:.3f}\n" + f"TEST Loss: {avg_test_loss:.3f}\n"

        #TODO: Sanity check each of the above losses

        # Append current epoch results to CSV
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, epoch_loss, avg_train_loss, avg_test_loss])

        if best_test_loss > avg_test_loss:
            best_test_loss = avg_test_loss
            torch.save(ddpm.state_dict(), best_model_file)
            log_string += " --> Best model ever (stored based on test loss)"

        log_string += (f"\nAverage test loss: {avg_test_loss:.3f} -> best: {best_test_loss:.3f}\n"
                       + f"Average train loss: {avg_train_loss:.3f}")

        tqdm.write(log_string)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': ddpm.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'epoch_losses': epoch_losses,
            'train_losses': train_losses,
            'test_losses': test_losses,
            'best_test_loss': best_test_loss
        }
        torch.save(checkpoint, model_file)

        """
        if display:
            show_images(generate_new_images(ddpm, device=device), f"Images generated at epoch {epoch + 1}")
        """

    plt.figure(figsize=(20, 10))
    plt.plot(epoch_losses, label='Epoch Loss')
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('training and Test Loss')
    plt.savefig(plot_file)

optimizer = Adam(ddpm.parameters(), lr=lr)

if training_mode:
    training_loop(ddpm, train_loader, test_loader, n_epochs,
                  optim=optimizer, device=data_init.get_device(),
                  loss_function=loss_strategy, noise_function=noise_strategy)