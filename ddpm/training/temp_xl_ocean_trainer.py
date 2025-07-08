import csv
import os
import sys
import logging
import pygame
from datetime import datetime

import torch
from halo import Halo
import matplotlib


matplotlib.use('Agg')  # Use non-interactive backend
from matplotlib import pyplot as plt
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from ddpm.helper_functions.model_evaluation import evaluate
from ddpm.neural_networks.temp_ddpm import MyDDPMGaussian
from ddpm.neural_networks.unets.new_unet_xl import MyUNet
from data_prep.data_initializer import DDInitializer
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class TrainOceanXL():
    """
    This file is being used to train the best model of all time baybee.
    There's never been a model better than this one, we got the best epsilons,
    our loss function becomes our win function, truly remarkable stuff.
    Absolute model
        | o |
        \ | /
          |
         / \
        |   |
    """

    def __init__(self):
        """
        Initializes model, datasets, loaders, and all training configs using DDInitializer.
        """
        self.dd = DDInitializer()
        dd = self.dd
        self._setup_paths_and_files(dd)
        self.device = dd.get_device()
        self.n_steps = dd.get_attribute('noise_steps')
        self.min_beta = dd.get_attribute('min_beta')
        self.max_beta = dd.get_attribute('max_beta')
        self.num_workers = dd.get_attribute('num_workers')
        self.ddpm = MyDDPMGaussian(MyUNet(self.n_steps),
                                   n_steps=self.n_steps,
                                   min_beta=self.min_beta,
                                   max_beta=self.max_beta,
                                   device=self.device)
        self.noise_strategy = dd.get_noise_strategy()
        self.loss_strategy = dd.get_loss_strategy()
        self.standardize_strategy = dd.get_standardizer()
        self.training_mode = dd.get_attribute('training_mode')
        self.batch_size = dd.get_attribute('batch_size')
        self.n_epochs = dd.get_attribute('epochs')
        self.lr = dd.get_attribute('lr')
        self._DEFAULT_BEST = "./training_output/ddpm_ocean_model_best_checkpoint.pt"

        self.train_loader = DataLoader(dd.get_training_data(),
                                       batch_size=self.batch_size,
                                       shuffle=True,
                                       )
        self.test_loader = DataLoader(dd.get_test_data(),
                                      batch_size=self.batch_size,
                                      )
        self.val_loader = DataLoader(dd.get_validation_data(),
                                     batch_size=self.batch_size,
                                     )

        self.continue_training = False
        self.model_to_retrain = dd.get_attribute('model_to_retrain')
        self.retrain_mode = dd.get_attribute('retrain_mode')

    def retrain_this(self, path: str = ""):
        """
        Sets the path of a model to resume training from.

        Args:
            path (str): Path to a saved checkpoint file.
        """
        if self.retrain_mode:
            self.continue_training = True

        if path == "":
            path = self._DEFAULT_BEST

        if not os.path.exists(path):
            raise FileNotFoundError(f"path {path} doesn't exist")
        else:
            self.model_to_retrain = path
            self.continue_training = True

    def set_music(self, music_path='music.mp3'):  # 'music.mp3'):
        self.music_path = os.path.join(os.path.dirname(__file__), music_path)

    def _setup_paths_and_files(self, dd):
        """
        Prepares all output paths for saving models, plots, and logs.
        """
        self.set_timestamp()
        self.set_output_directory()
        self.set_csv_file()
        self.save_config_used()
        self.set_model_file()
        self.set_plot_file()
        self.set_csv_description(dd)
        self.set_music()

    def load_checkpoint(self, optimizer: torch.optim.Optimizer):
        """
        Loads a model + optimizer state from a checkpoint file.

        Args:
            optimizer (torch.optim.Optimizer): Optimizer to restore.

        Returns:
            dict: Loaded checkpoint containing all training state.
        """
        checkpoint = torch.load(self.model_to_retrain, weights_only=False)
        self.ddpm.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint

    def set_timestamp(self, timestamp=datetime.now().strftime("%h%d_%H%M")):
        """
        Sets the training session timestamp for naming outputs.

        Args:
            timestamp (str, optional): Override timestamp value.
        """
        self.timestamp = timestamp

    def set_output_directory(self, training_output="training_output"):
        """
        Creates the output directory for this training run.

        Args:
            training_output (str, optional): Name of the output directory.
        """
        self.output_directory = os.path.join(os.path.dirname(__file__), f"{training_output}/")
        os.makedirs(self.output_directory, exist_ok=True)

    def save_config_used(self):
        import yaml
        with open(os.path.join(self.output_directory, "config_used.yaml"), 'w') as f:
            yaml.dump(self.dd.get_full_config(), f)
        logging.info("💾 Saved training config to config_used.yaml")

    def set_csv_file(self, csv_file="training_log"):
        """
        Creates the path to the CSV log file using the timestamp.

        Args:
            csv_file (str, optional): Base name for CSV file.
        """

        csv_file = f"{self.output_directory}{csv_file}_{self.timestamp}.csv"
        self.csv_file = os.path.join(os.path.dirname(__file__), csv_file)

    def set_plot_file(self, plot_file="training_test_loss_xl"):
        """
        Creates the path to the loss plot file.

        Args:
            plot_file (str, optional): Base name for plot image.
        """
        plot_file = f"{self.output_directory}{plot_file}_{self.timestamp}.png"
        self.plot_file = os.path.join(os.path.dirname(__file__), plot_file)

    def set_model_file(self, initial_model_file="ddpm_ocean_model"):
        """
        Creates paths for saving the current model and best checkpoints.

        Args:
            initial_model_file (str, optional): Base name for model files.
        """
        model_file = f"{self.output_directory}{initial_model_file}_{self.timestamp}"
        best_model_weights = f"{self.output_directory}{initial_model_file}_best_model_weights.pt"
        best_model_checkpoint = f"{self.output_directory}{initial_model_file}_best_checkpoint.pt"
        self.model_file = os.path.join(os.path.dirname(__file__), f"{model_file}.pt")
        self.best_model_weights = os.path.join(os.path.dirname(__file__), best_model_weights)
        self.best_model_checkpoint = os.path.join(os.path.dirname(__file__), best_model_checkpoint)

    def set_csv_description(self, dd: DDInitializer):
        self.description = f"Standardization Method: {dd.get_attribute('standardizer_type')} | Noise: {dd.get_attribute('noise_function')} | Loss: {dd.get_attribute('loss_function')}  "

    def set_ddpm(self, ddpm: torch.nn.Module):
        """
        Replaces the current DDPM model.

        Args:
            ddpm (torch.nn.Module): New DDPM model instance.
        """
        self.ddpm = ddpm

    def set_epochs(self, epochs: int):
        """
        Sets the number of training epochs.

        Args:
            epochs (int): Number of epochs.
        """
        self.epochs = epochs

    def training_loop(self, optim: torch.optim.Optimizer, loss_function: callable):
        """
        Main training logic. Trains DDPM over epochs, logs results, evaluates with multi-threading,
        and saves the best model based on test loss.

        Args:
            optim (torch.optim.Optimizer): Optimizer.
            loss_function (callable, optional): Loss function for training.
        """
        best_test_loss = float("inf")
        epoch_losses = []
        train_losses = []
        test_losses = []
        start_epoch = 0
        ddpm = self.ddpm
        device = self.device
        csv_file = self.csv_file
        n_epochs = self.n_epochs
        plot_file = self.plot_file
        model_file = self.model_file
        test_loader = self.test_loader
        train_loader = self.train_loader
        best_model_weights = self.best_model_weights
        best_model_checkpoint = self.best_model_checkpoint

        if self.continue_training:
            checkpoint = self.load_checkpoint(optim)
            start_epoch = checkpoint['epoch'] + 1
            epoch_losses = checkpoint['epoch_losses']
            train_losses = checkpoint['train_losses']
            test_losses = checkpoint['test_losses']
            best_test_loss = checkpoint['best_test_loss']
            print(f"Resuming training from epoch {start_epoch}. Training for {n_epochs} epochs!")
            logging.info(f"Resuming training from epoch {start_epoch}. Training for {n_epochs} epochs!")

        best_epoch = start_epoch

        # CSV output setup
        with open(self.csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([self.description])
            writer.writerow(['Epoch', 'Epoch Loss', 'Train Loss', 'Test Loss'])

        # Training arc
        for epoch in tqdm(range(start_epoch, start_epoch + n_epochs), desc="training progress", colour="#00ff00"):
            if (epoch % 100 == 87):
                pygame.mixer.music.play()

            epoch_loss = 0.0
            ddpm.train()

            for _, (x0, t, noise), in enumerate(
                    tqdm(train_loader, leave=False, desc=f"Epoch {epoch + 1}/{start_epoch + n_epochs}",
                         colour="#005500")):
                n = len(x0)
                x0 = x0.to(device)

                x0_reshaped = torch.permute(x0, (1, 2, 3, 0))
                mask_raw = (standardizer.unstandardize(x0_reshaped).abs() != 0.0).float()
                mask = torch.permute(mask_raw, (3, 0, 1, 2))

                t = t.to(device)
                noise = noise.to(device)

                noisy_imgs = ddpm(x0, t, noise)
                predicted_noise, _ = ddpm.backward(noisy_imgs, t.reshape(n, -1), mask)

                loss = loss_function(predicted_noise, noise)

                optim.zero_grad()
                loss.backward()
                optim.step()

                epoch_loss += loss.item() * len(x0) / len(train_loader.dataset)

            with ThreadPoolExecutor(max_workers=2) as executor:
                spinner = Halo("Evaluating DDPM...", spinner="dots")
                spinner.start()
                ddpm.eval()
                spinner.succeed()

                spinner = Halo("Evaluating average train loss...", spinner="dots")
                spinner.start()
                train_future = executor.submit(evaluate, ddpm, train_loader, device)

                spinner = Halo("Evaluating average test loss...", spinner="dots")
                spinner.start()
                test_future = executor.submit(evaluate, ddpm, test_loader, device)

                avg_train_loss = train_future.result()

                spinner.succeed()
                avg_test_loss = test_future.result()
                spinner.succeed()

            epoch_losses.append(epoch_loss)
            train_losses.append(avg_train_loss)
            test_losses.append(avg_test_loss)

            log_string = f"\nepoch {epoch + 1}: \n"
            log_string += f"EPOCH Loss: {epoch_loss:.7f}\n"
            log_string += f"TRAIN Loss: {avg_train_loss:.7f}\n"
            log_string += f"TEST Loss: {avg_test_loss:.7f}\n"

            # Append current epoch results to CSV
            with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([epoch + 1, epoch_loss, avg_train_loss, avg_test_loss])
            ddpm.train()

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': ddpm.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'epoch_losses': epoch_losses,
                'train_losses': train_losses,
                'test_losses': test_losses,
                'best_test_loss': best_test_loss,
                'n_steps': self.n_steps,
                'noise_strategy': self.noise_strategy,
                'standardizer_type': self.standardize_strategy,
            }

            if best_test_loss > avg_test_loss:
                best_test_loss = avg_test_loss
                torch.save(ddpm.state_dict(), best_model_weights)
                torch.save(checkpoint, best_model_checkpoint)
                log_string += '\033[32m' + " --> Best model ever (stored based on test loss)" + '\033[0m'
                best_epoch = epoch
            else:
                torch.save(checkpoint, model_file)

            log_string += (f"\nAverage test loss: {avg_test_loss:.7f} -> best: {best_test_loss:.7f}\n"
                           + f"Average train loss: {avg_train_loss:.7f}\n"
                           + f"Best Epoch: {best_epoch}")

            tqdm.write(log_string)
            # pygame.mixer.music.stop()

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
        plt.title('| || || |_')
        plt.savefig(plot_file)

    def train(self):
        """
        Sets up optimizer and kicks off training based on config mode.
        """
        logging.info("🔧 Starting DDPM training...")
        logging.info(f"👾 Device: {self.device}")
        logging.info(f"📦 Batch size: {self.batch_size}")
        logging.info(f"🧠 Epochs: {self.n_epochs}")
        logging.info(f"📁 Output Dir: {self.output_directory}")

        optimizer = Adam(self.ddpm.parameters(), lr=self.lr)

        if os.path.exists(self.csv_file):
            logging.warning("⚠️ CSV log already exists. This training run may overwrite it.")

        if self.retrain_mode:
            self.retrain_this()

        if self.training_mode:
            pygame.mixer.init()
            pygame.mixer.music.load(self.music_path)
            # pygame.mixer.music.play()
            self.training_loop(optimizer, self.loss_strategy)

        print("🎉 Training finished successfully!")
        print("last model saved in:", self.model_file)
        print("best model checkpoint saved in:", self.best_model_checkpoint)
        logging.info("🎉 Training finished successfully!")
        logging.info(f"📦 Final model: {self.model_file}")
        logging.info(f"🏆 Best model: {self.best_model_checkpoint}")


if __name__ == '__main__':
    try:
        trainer = TrainOceanXL()
        trainer.train()
    except Exception as e:
        logging.error("🚨 Oops! Something went wrong during training.")
        logging.error(f"💥 Error: {str(e)}")
        print("Training crashed. Check the logs or ask your local neighborhood AI expert 🧠.")
