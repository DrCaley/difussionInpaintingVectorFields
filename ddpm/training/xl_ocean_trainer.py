import argparse
import csv
import sys
import logging
import torch
import matplotlib

from datetime import datetime
from halo import Halo
from pathlib import Path


matplotlib.use('Agg')
from matplotlib import pyplot as plt
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

pkg_path = Path(__file__).resolve().parents[2]

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR.parent.parent))
from data_prep.data_initializer import DDInitializer
from ddpm.helper_functions.death_messages import get_death_message

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class TrainOceanXL:
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

    def __init__(self, config_path = None, pickle_path = None):
        """
        Initializes model, datasets, loaders, and all training configs using DDInitializer.
        """
        kwargs = {}
        if config_path is not None:
            kwargs['config_path'] = config_path
        if pickle_path is not None:
            kwargs['pickle_path'] = pickle_path

        self.dd = DDInitializer(**kwargs)

        dd = self.dd
        self._setup_paths_and_files(dd)
        self.device = dd.get_device()
        self.n_steps = dd.get_attribute('noise_steps')
        self.min_beta = dd.get_attribute('min_beta')
        self.max_beta = dd.get_attribute('max_beta')
        self.num_workers = dd.get_attribute('num_workers')
        self.partial_conv_mode = dd.get_attribute('partial_conv_mode')

        try:
            if self.partial_conv_mode:
                self.configure_partial_conv_mode()
            else:
                self.configure_gaussian_cov_mode()
        except Exception as e:
            logging.exception("🔥 Failed to initialize ddpm model.")
            raise e

        self.noise_strategy = dd.get_noise_strategy()
        self.loss_strategy = dd.get_loss_strategy()
        self.standardize_strategy = dd.get_standardizer()
        self.training_mode = dd.get_attribute('training_mode')
        self.batch_size = dd.get_attribute('batch_size')
        self.n_epochs = dd.get_attribute('epochs')
        self.lr = dd.get_attribute('lr')
        self._DEFAULT_BEST = "training_output/ddpm_ocean_model_best_checkpoint.pt"


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

    def configure_gaussian_cov_mode(self):
        from ddpm.neural_networks.ddpm import GaussianDDPM
        from ddpm.helper_functions.model_evaluation import evaluate
        from ddpm.neural_networks.unets.unet_xl import MyUNet

        self.ddpm = GaussianDDPM(MyUNet(self.n_steps).to(self.device),
                                 n_steps=self.n_steps,
                                 min_beta=self.min_beta,
                                 max_beta=self.max_beta,
                                 device=self.device)
        self.evaluate = evaluate

    def configure_partial_conv_mode(self):
        from ddpm.neural_networks.interpolation_ddpm import InterpolationDDPM
        from ddpm.helper_functions.temp_model_evaluation import evaluate
        from ddpm.neural_networks.unets.pconv_unet_xl import MyUNet

        self.ddpm = InterpolationDDPM(MyUNet(self.n_steps).to(self.device),
                                      n_steps=self.n_steps,
                                      min_beta=self.min_beta,
                                      max_beta=self.max_beta,
                                      device=self.device)
        self.evaluate = evaluate

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

        path = (Path(__file__).resolve().parent / path)

        print(path)

        if not Path(path).exists():
            raise FileNotFoundError(f"path {path} doesn't exist")
        else:
            self.model_to_retrain = path
            self.continue_training = True

    def _setup_paths_and_files(self, dd):
        """
        Prepares all output paths for saving models, plots, and logs.
        """
        self.set_timestamp()
        self.set_output_directory(self.dd.get_config_name())
        self.set_csv_file()
        self.save_config_used()
        # Build a descriptive model filename from noise type and T
        noise_fn = dd.get_attribute('noise_function') or 'unknown'
        n_steps = dd.get_attribute('noise_steps') or 0
        model_base_name = f"{noise_fn}_t{n_steps}"
        self.set_model_file(initial_model_file=model_base_name)
        self.set_plot_file()

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

    def set_output_directory(self, training_output = "model"):
        """
        Creates the output directory for this training run.

        Args:
            training_output (str, optional): Name of the output directory.
        """
        self.output_directory = (Path(__file__).parent / "training_output" / training_output).resolve()
        self.output_directory.mkdir(parents=True, exist_ok=True)

    def save_config_used(self):
        import yaml
        config_path = self.output_directory / "config_used.yaml"
        with config_path.open('w') as f:
            yaml.dump(self.dd.get_full_config(), f)
        logging.info("JARVIS: 💾 Saved training config to config_used.yaml")

    def set_csv_file(self, csv_file="training_log"):
        """
        Creates the path to the CSV log file using the timestamp.

        Args:
            csv_file (str, optional): Base name for CSV file.
        """

        self.csv_file = self.output_directory / f"{csv_file}_{self.timestamp}.csv"

    def set_plot_file(self, plot_file="training_test_loss_xl"):
        """
        Creates the path to the loss plot file.

        Args:
            plot_file (str, optional): Base name for plot image.
        """
        self.plot_file = self.output_directory / f"{plot_file}_{self.timestamp}.png"

    def set_model_file(self, initial_model_file="ddpm_ocean_model"):
        """
        Creates paths for saving the current model and best checkpoints.

        Args:
            initial_model_file (str, optional): Base name for model files.
        """
        model_base = f"{initial_model_file}_{self.timestamp}"
        self.model_file = self.output_directory / f"{model_base}.pt"
        self.best_model_weights = self.output_directory / f"{initial_model_file}_best_model_weights.pt"
        self.best_model_checkpoint = self.output_directory / f"{initial_model_file}_best_checkpoint.pt"

    def set_ddpm(self, ddpm: torch.nn.Module):
        """
        Replaces the current ddpm model.

        Args:
            ddpm (torch.nn.Module): New ddpm model instance.
        """
        self.ddpm = ddpm

    def set_epochs(self, epochs: int):
        """
        Sets the number of training epochs.

        Args:
            epochs (int): Number of epochs.
        """
        self.epochs = epochs

    def training_loop(self, optim: torch.optim.Optimizer, loss_function: callable, ):
        """
        Main training logic. Trains ddpm over epochs, logs results, evaluates with multi-threading,
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
            logging.info(f"Resuming training from epoch {start_epoch}. Training for {n_epochs} epochs!")

        best_epoch = start_epoch

        # CSV output setup
        with self.csv_file.open(mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Epoch', 'Epoch Loss', 'Train Loss', 'Test Loss'])

        # Training arc
        try:
            for epoch in tqdm(range(start_epoch, start_epoch + n_epochs), desc="training progress", colour="#00ff00"):
                epoch_loss = 0.0
                ddpm.train()

                for _, (x0, t, noise), in enumerate(
                        tqdm(train_loader, leave=False, desc=f"Epoch {epoch + 1}/{start_epoch + n_epochs}",
                             colour="#005500")):

                    n = len(x0)
                    x0 = x0.to(device)
                    t = t.to(device)
                    noise = noise.to(device)

                    noisy_imgs = ddpm(x0, t, noise)

                    if self.partial_conv_mode:
                        x0_reshaped = torch.permute(x0, (1, 2, 3, 0)).to(self.device)
                        mask_raw = (self.standardize_strategy.unstandardize(x0_reshaped).abs() > 1e-5).float().to(
                            self.device)
                        mask = torch.permute(mask_raw, (3, 0, 1, 2)).to(self.device)
                        predicted_noise, _ = ddpm.backward(noisy_imgs, t.reshape(n, -1), mask)
                    else:
                        predicted_noise = ddpm.backward(noisy_imgs, t.reshape(n, -1))

                    loss = loss_function(predicted_noise, noise, noisy_imgs)

                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                    epoch_loss += loss.item() * len(x0) / len(train_loader.dataset)

                spinner = Halo("Evaluating ddpm...", spinner="dots")
                spinner.start()
                ddpm.eval()
                spinner.succeed()

                spinner = Halo("Evaluating average train loss...", spinner="dots")
                spinner.start()
                avg_train_loss = self.evaluate(ddpm, train_loader, device)
                spinner.succeed()

                spinner = Halo("Evaluating average test loss...", spinner="dots")
                spinner.start()
                avg_test_loss = self.evaluate(ddpm, test_loader, device)
                spinner.succeed()

                epoch_losses.append(epoch_loss)
                train_losses.append(avg_train_loss)
                test_losses.append(avg_test_loss)

                log_string = f"\nepoch {epoch + 1}: \n"
                log_string += f"EPOCH Loss: {epoch_loss:.7f}\n"
                log_string += f"TRAIN Loss: {avg_train_loss:.7f}\n"
                log_string += f"TEST Loss: {avg_test_loss:.7f}\n"

                # Append current epoch results to CSV
                try:
                    with self.csv_file.open(mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([epoch + 1, epoch_loss, avg_train_loss, avg_test_loss])
                except Exception as e:
                    logging.exception("📉 Failed to write to training CSV file.")

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
                    'min_beta': self.min_beta,
                    'max_beta': self.max_beta,
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

                """
                if display:
                    show_images(generate_new_images(ddpm, device=device), f"Images generated at epoch {epoch + 1}")
                """
        except KeyboardInterrupt:
            logging.error(get_death_message())
        finally:
            self.plot_graphs(epoch_losses, train_losses, test_losses)

    def plot_graphs(self, epoch_losses, train_losses, test_losses):
        try:
            plt.figure(figsize=(20, 10))
            plt.plot(epoch_losses, label='Epoch Loss')
            plt.plot(train_losses, label='Train Loss')
            plt.plot(test_losses, label='Test Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.title('| || || |_')
            plt.savefig(self.plot_file)
            logging.info(f"🖼️ Saved training loss plot: {self.plot_file}")
        except Exception as e:
            logging.exception("🎨 Failed to generate/save training plot.")

    def train(self):
        """
        Sets up optimizer and kicks off training based on config mode.
        """
        logging.info("🔧 Starting ddpm training...")
        logging.info(f"👾 Device: {self.device}")
        logging.info(f"📦 Batch size: {self.batch_size}")
        logging.info(f"🧠 Epochs: {self.n_epochs}")

        if self.partial_conv_mode:
            logging.info("🗺️ Running with a partial Convolution ddpm")
        logging.info(f"📁 Output Dir: {self.output_directory}")

        optimizer = Adam(self.ddpm.parameters(), lr=self.lr)

        if self.csv_file.exists():
            logging.warning("⚠️ CSV log already exists. This training run may overwrite it.")

        if self.retrain_mode:
            path = self.dd.get_attribute('model_to_retrain')
            self.retrain_this(path)

        if self.training_mode:
            self.training_loop(optimizer, self.loss_strategy)

        logging.info("🎉 Training finished successfully!")
        logging.info(f"📦 Final model: {self.model_file}")
        logging.info(f"🏆 Best model: {self.best_model_checkpoint}")


def main():

    ##############################
    ### Parse Arguments
    ##############################

    parser = argparse.ArgumentParser()

    parser.add_argument("--training_cfg", 
                        default=pkg_path / "data.yaml")

    parser.add_argument("--model_name",
                        default='ddpm')

    args = parser.parse_args()   

    ##############################
    ### Parse Arguments
    ##############################
    try:
        trainer = TrainOceanXL(config_path=args.training_cfg)
        trainer.train()
    except Exception as e:
        logging.error("🚨 Oops! Something went wrong during training.")
        logging.error(f"💥 Error: {str(e)}")
        logging.error(get_death_message())
        logging.error("Training crashed. Check the logs or ask your local neighborhood AI expert 🧠.")



if __name__ == '__main__':
    main()