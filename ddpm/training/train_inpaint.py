"""Palette-style mask-aware inpainting trainer.

Trains a 5-channel UNet (MyUNet_Inpaint / MyUNet_FiLM) that receives:
    [x_t, mask, known_values] → predicts noise ε  OR  clean data x₀

Prediction target controlled by config key `prediction_target`:
    - "eps" (default): standard noise prediction
    - "x0": direct clean-image prediction (single-step inference)

Usage:
    PYTHONPATH=. python ddpm/training/train_inpaint.py
    PYTHONPATH=. python ddpm/training/train_inpaint.py --training_cfg path/to/config.yaml
"""

import argparse
import csv
import sys
import logging
import random
import torch
import matplotlib
import numpy as np

from datetime import datetime
from pathlib import Path

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

from data_prep.data_initializer import DDInitializer
from data_prep.ocean_inpaint_dataset import OceanInpaintDataset
from ddpm.neural_networks.ddpm import GaussianDDPM
from ddpm.neural_networks.unets.unet_inpaint import MyUNet_Inpaint
from ddpm.neural_networks.unets.unet_film import MyUNet_FiLM
from ddpm.neural_networks.unets.unet_xl import MyUNet
from ddpm.neural_networks.unets.unet_xl_attn import MyUNet_Attn
from ddpm.helper_functions.death_messages import get_death_message

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class TrainInpaint:
    """Trains a Palette-style mask-aware DDPM for inpainting."""

    def __init__(self, config_path=None):
        kwargs = {}
        if config_path is not None:
            kwargs["config_path"] = config_path
            # Reset singleton so DDInitializer reads OUR config, not data.yaml
            DDInitializer._instance = None

        self.dd = DDInitializer(**kwargs)
        dd = self.dd
        self.device = dd.get_device()
        self.n_steps = dd.get_attribute("noise_steps")
        self.min_beta = dd.get_attribute("min_beta")
        self.max_beta = dd.get_attribute("max_beta")
        self.batch_size = dd.get_attribute("batch_size")
        self.n_epochs = dd.get_attribute("epochs")
        self.lr = dd.get_attribute("lr")
        self.noise_strategy = dd.get_noise_strategy()
        self.loss_strategy = dd.get_loss_strategy()
        self.standardizer = dd.get_standardizer()

        # Classifier-free guidance: probability of dropping conditioning
        self.p_uncond = float(dd.get_attribute("p_uncond") or 0.0)

        # Mask x_t: replace known region with independent noise during training
        # Forces model to read conditioning channels instead of extracting
        # known-region info from x_t itself
        self.mask_xt = bool(dd.get_attribute("mask_xt") or False)

        # Prediction target: "eps" (noise) or "x0" (clean image)
        # x0-prediction enables single-step inference (no iterative denoising)
        self.prediction_target = dd.get_attribute("prediction_target") or "eps"
        assert self.prediction_target in ("eps", "x0"), (
            f"prediction_target must be 'eps' or 'x0', got '{self.prediction_target}'"
        )

        # Gradient clipping (max L2 norm); 0 = disabled
        self.max_grad_norm = float(dd.get_attribute("max_grad_norm") or 0)

        # UNet type: "concat" (Palette-style) or "film" (FiLM conditioning)
        self.unet_type = dd.get_attribute("unet_type") or "concat"

        # Build inpainting model
        if self.unet_type == "film":
            unet = MyUNet_FiLM(n_steps=self.n_steps).to(self.device)
            logging.info("Using FiLM-conditioned UNet")
        elif self.unet_type == "standard":
            unet = MyUNet(n_steps=self.n_steps).to(self.device)
            unet.in_channels = 2  # expose for logging
            logging.info("Using unconditional UNet (standard, 2-channel)")
        elif self.unet_type == "standard_attn":
            unet = MyUNet_Attn(n_steps=self.n_steps).to(self.device)
            logging.info("Using unconditional UNet with self-attention (standard_attn, 2-channel)")
        else:
            unet = MyUNet_Inpaint(n_steps=self.n_steps).to(self.device)
            logging.info("Using concat-conditioned UNet (Palette-style)")

        # Unconditional UNets: disable conditioning-related options
        if self.unet_type in ("standard", "standard_attn"):
            if self.mask_xt:
                logging.warning("mask_xt is ignored for unet_type='%s' (no conditioning)", self.unet_type)
                self.mask_xt = False
            if self.p_uncond > 0:
                logging.warning("p_uncond is ignored for unet_type='%s' (no conditioning)", self.unet_type)
                self.p_uncond = 0.0

        self.ddpm = GaussianDDPM(
            unet,
            n_steps=self.n_steps,
            min_beta=self.min_beta,
            max_beta=self.max_beta,
            device=self.device,
        )

        # Wrap datasets with inpainting conditioning
        self.train_loader = DataLoader(
            OceanInpaintDataset(dd.get_training_data(), standardizer=self.standardizer),
            batch_size=self.batch_size,
            shuffle=True,
        )
        self.test_loader = DataLoader(
            OceanInpaintDataset(dd.get_test_data(), standardizer=self.standardizer),
            batch_size=self.batch_size,
        )

        # Output paths
        self.timestamp = datetime.now().strftime("%h%d_%H%M")
        noise_fn = dd.get_attribute("noise_function") or "unknown"
        model_name = dd.get_attribute("model_name") or f"inpaint_{noise_fn}_t{self.n_steps}"

        # If output_dir is set (e.g. by experiment launcher), use it;
        # otherwise fall back to the legacy training_output/ location.
        custom_output = dd.get_attribute("output_dir")
        if custom_output:
            self.output_dir = Path(custom_output).resolve()
        else:
            self.output_dir = (
                Path(__file__).parent / "training_output" / model_name
            ).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.csv_file = self.output_dir / f"training_log_{self.timestamp}.csv"
        self.plot_file = self.output_dir / f"loss_plot_{self.timestamp}.png"

        model_base = f"inpaint_{noise_fn}_t{self.n_steps}"
        self.best_checkpoint = self.output_dir / f"{model_base}_best_checkpoint.pt"
        self.latest_checkpoint = self.output_dir / f"{model_base}_{self.timestamp}.pt"
        self.best_weights = self.output_dir / f"{model_base}_best_weights.pt"

        # Save config
        self._save_config()

        # Optional: resume from checkpoint
        self.retrain_mode = dd.get_attribute("retrain_mode")
        self.model_to_retrain = dd.get_attribute("model_to_retrain")

    def _save_config(self):
        import yaml
        config_path = self.output_dir / "config_used.yaml"
        with config_path.open("w") as f:
            yaml.dump(self.dd.get_full_config(), f)
        logging.info(f"Saved config to {config_path}")

    def evaluate(self, loader, fixed_seed=None):
        """Evaluate average MSE loss on a loader."""
        py_state = None
        np_state = None
        torch_state = None
        if fixed_seed is not None:
            py_state = random.getstate()
            np_state = np.random.get_state()
            torch_state = torch.random.get_rng_state()
            random.seed(fixed_seed)
            np.random.seed(fixed_seed)
            torch.manual_seed(fixed_seed)

        self.ddpm.eval()
        criterion = torch.nn.MSELoss()
        total_loss = 0.0
        count = 0

        try:
            with torch.no_grad():
                for i, (x0, t, noise, mask, known) in enumerate(loader):
                    if i > 20:
                        break
                    x0 = x0.to(self.device)
                    t = t.to(self.device)
                    noise = noise.to(self.device)
                    n = len(x0)

                    # Forward: noise the clean image
                    noisy = self.ddpm(x0, t, noise)

                    if self.unet_type in ("standard", "standard_attn"):
                        # Unconditional UNet: just (x_t, t) → ε or x̂₀
                        pred = self.ddpm.network(noisy, t.reshape(n, -1))
                    else:
                        mask = mask.to(self.device)
                        known = known.to(self.device)

                        # Mask x_t in known region (same as training)
                        if self.mask_xt:
                            known_mask = 1.0 - mask
                            indep_noise = torch.randn_like(noisy)
                            noisy = noisy * mask + indep_noise * known_mask

                        # Concatenate conditioning: [x_t, mask, known_values]
                        x_cond = torch.cat([noisy, mask, known], dim=1)  # (N, 5, H, W)
                        pred = self.ddpm.network(x_cond, t.reshape(n, -1))

                    # Choose target
                    target = x0 if self.prediction_target == "x0" else noise

                    if self.mask_xt:
                        # Only compute loss in missing region (same as training)
                        mask = mask.to(self.device)
                        mask_2ch = mask[:, :2]
                        diff = (pred - target) ** 2
                        loss = (diff * mask_2ch).sum() / mask_2ch.sum().clamp(min=1.0)
                    else:
                        loss = criterion(pred, target)
                    total_loss += loss.item() * n
                    count += n
        finally:
            if fixed_seed is not None:
                random.setstate(py_state)
                np.random.set_state(np_state)
                torch.random.set_rng_state(torch_state)

        return total_loss / count if count > 0 else float("inf")

    def train(self):
        logging.info(f"Device: {self.device}")
        logging.info(f"Batch size: {self.batch_size}, Epochs: {self.n_epochs}, LR: {self.lr}")
        logging.info(f"UNet type: {self.unet_type} ({type(self.ddpm.network).__name__})")
        logging.info(f"UNet input channels: {self.ddpm.network.in_channels}")
        logging.info(f"CFG p_uncond: {self.p_uncond}")
        logging.info(f"Mask x_t (known region): {self.mask_xt}")
        logging.info(f"Prediction target: {self.prediction_target}")
        logging.info(f"Output dir: {self.output_dir}")

        optimizer = Adam(self.ddpm.parameters(), lr=self.lr)

        start_epoch = 0
        best_test_loss = float("inf")
        epoch_losses, train_losses, test_losses = [], [], []

        # Resume if requested
        if self.retrain_mode and self.model_to_retrain:
            path = Path(self.model_to_retrain)
            if path.exists():
                checkpoint = torch.load(path, map_location=self.device, weights_only=False)
                self.ddpm.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                start_epoch = checkpoint.get("epoch", 0) + 1
                best_test_loss = checkpoint.get("best_test_loss", float("inf"))
                epoch_losses = checkpoint.get("epoch_losses", [])
                train_losses = checkpoint.get("train_losses", [])
                test_losses = checkpoint.get("test_losses", [])
                # If loss metric changed (e.g. switched to masked loss), reset best
                if self.dd.get_attribute("reset_best"):
                    best_test_loss = float("inf")
                    logging.info("Reset best_test_loss (loss metric changed)")
                logging.info(f"Resumed from epoch {start_epoch}")

        # CSV header
        with self.csv_file.open("w", newline="") as f:
            csv.writer(f).writerow(["Epoch", "Epoch Loss", "Train Loss", "Test Loss"])

        best_epoch = start_epoch

        try:
            for epoch in tqdm(
                range(start_epoch, start_epoch + self.n_epochs),
                desc="Training (inpaint)",
                colour="#00ff00",
            ):
                epoch_loss = 0.0
                self.ddpm.train()

                for _, (x0, t, noise, mask, known) in enumerate(
                    tqdm(
                        self.train_loader,
                        leave=False,
                        desc=f"Epoch {epoch + 1}/{start_epoch + self.n_epochs}",
                        colour="#005500",
                    )
                ):
                    n = len(x0)
                    x0 = x0.to(self.device)
                    t = t.to(self.device)
                    noise = noise.to(self.device)

                    # Forward diffusion: add noise to x0
                    noisy = self.ddpm(x0, t, noise)

                    if self.unet_type in ("standard", "standard_attn"):
                        # Unconditional UNet: just (x_t, t) → ε or x̂₀
                        pred = self.ddpm.network(noisy, t.reshape(n, -1))
                    else:
                        mask = mask.to(self.device)
                        known = known.to(self.device)

                        # Mask x_t: replace known region with independent noise
                        # so model can't extract known-region info from x_t
                        if self.mask_xt:
                            known_mask = 1.0 - mask  # 1 where known (mask=1 means missing)
                            indep_noise = torch.randn_like(noisy)
                            noisy = noisy * mask + indep_noise * known_mask

                        # Classifier-free guidance: randomly drop conditioning
                        if self.p_uncond > 0:
                            drop = (torch.rand(n, 1, 1, 1, device=self.device) < self.p_uncond).float()
                            mask = mask * (1.0 - drop)    # zero out mask
                            known = known * (1.0 - drop)  # zero out known values

                        # Concatenate conditioning channels
                        x_cond = torch.cat([noisy, mask, known], dim=1)  # (N, 5, H, W)

                        # UNet predicts ε or x₀
                        pred = self.ddpm.network(x_cond, t.reshape(n, -1))

                    # Choose target based on prediction_target
                    target = x0 if self.prediction_target == "x0" else noise

                    # Loss
                    if self.mask_xt:
                        # Only compute loss in missing region — in the known region
                        # x_t was replaced with independent noise, so the model
                        # literally cannot predict the target there.
                        mask_2ch = mask[:, :2]  # (N, 2, H, W), 1=missing
                        diff = (pred - target) ** 2
                        masked_diff = diff * mask_2ch
                        loss = masked_diff.sum() / mask_2ch.sum().clamp(min=1.0)
                    else:
                        loss = self.loss_strategy(pred, target, noisy)

                    optimizer.zero_grad()
                    loss.backward()
                    if self.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.ddpm.parameters(), self.max_grad_norm
                        )
                    optimizer.step()

                    epoch_loss += loss.item() * n / len(self.train_loader.dataset)

                # Evaluate
                self.ddpm.eval()
                avg_train_loss = self.evaluate(self.train_loader)
                avg_test_loss = self.evaluate(self.test_loader, fixed_seed=12345)

                epoch_losses.append(epoch_loss)
                train_losses.append(avg_train_loss)
                test_losses.append(avg_test_loss)

                # CSV logging
                try:
                    with self.csv_file.open("a", newline="") as f:
                        csv.writer(f).writerow([epoch + 1, epoch_loss, avg_train_loss, avg_test_loss])
                except Exception:
                    pass

                self.ddpm.train()

                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": self.ddpm.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch_losses": epoch_losses,
                    "train_losses": train_losses,
                    "test_losses": test_losses,
                    "best_test_loss": best_test_loss,
                    "n_steps": self.n_steps,
                    "min_beta": self.min_beta,
                    "max_beta": self.max_beta,
                    "noise_strategy": self.noise_strategy,
                    "standardizer_type": self.standardizer,
                    "model_type": "inpaint",
                    "unet_type": self.unet_type,
                    "prediction_target": self.prediction_target,
                }

                log_str = (
                    f"\nEpoch {epoch + 1}: epoch_loss={epoch_loss:.7f}, "
                    f"train={avg_train_loss:.7f}, test={avg_test_loss:.7f}"
                )

                if avg_test_loss < best_test_loss:
                    best_test_loss = avg_test_loss
                    best_epoch = epoch
                    torch.save(self.ddpm.state_dict(), self.best_weights)
                    torch.save(checkpoint, self.best_checkpoint)
                    log_str += "\033[32m  --> BEST\033[0m"
                else:
                    torch.save(checkpoint, self.latest_checkpoint)

                log_str += f"  (best={best_test_loss:.7f} @ epoch {best_epoch + 1})"
                tqdm.write(log_str)

        except KeyboardInterrupt:
            logging.error(get_death_message())
        finally:
            self._plot(epoch_losses, train_losses, test_losses)

    def _plot(self, epoch_losses, train_losses, test_losses):
        try:
            plt.figure(figsize=(14, 7))
            plt.plot(epoch_losses, label="Epoch Loss")
            plt.plot(train_losses, label="Train Loss")
            plt.plot(test_losses, label="Test Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.title("Inpainting Model Training")
            plt.savefig(self.plot_file)
            logging.info(f"Saved plot to {self.plot_file}")
        except Exception:
            logging.exception("Failed to save plot")


def main():
    parser = argparse.ArgumentParser(description="Train Palette-style inpainting DDPM")
    parser.add_argument(
        "--training_cfg",
        default=str(BASE_DIR / "data.yaml"),
        help="Path to training config YAML",
    )
    args = parser.parse_args()

    try:
        trainer = TrainInpaint(config_path=args.training_cfg)
        trainer.train()
    except Exception as e:
        logging.error(f"Training failed: {e}")
        logging.error(get_death_message())
        raise


if __name__ == "__main__":
    main()
