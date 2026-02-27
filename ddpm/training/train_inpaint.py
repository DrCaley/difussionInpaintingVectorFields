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
import shutil
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
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
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
from ddpm.neural_networks.unets.unet_attn_slim import MyUNet_Attn_Slim
from ddpm.neural_networks.unets.unet_attn_mid import MyUNet_Attn_Mid
from ddpm.neural_networks.unets.unet_film_attn import MyUNet_FiLM_Attn
from ddpm.neural_networks.unets.unet_st import MyUNet_ST
from data_prep.ocean_sequence_dataset import OceanSequenceDataset
from ddpm.helper_functions.death_messages import get_death_message
from ddpm.helper_functions.ema import EMA

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

        # Gradient accumulation: simulate larger batches without more memory
        self.gradient_accumulation_steps = int(dd.get_attribute("gradient_accumulation_steps") or 1)

        # Gradient clipping (max L2 norm); 0 = disabled
        self.max_grad_norm = float(dd.get_attribute("max_grad_norm") or 0)

        # Weight decay for AdamW (0 = plain Adam)
        self.weight_decay = float(dd.get_attribute("weight_decay") or 0)

        # Learning rate schedule
        self.lr_schedule = dd.get_attribute("lr_schedule") or "constant"  # constant | cosine
        self.warmup_epochs = int(dd.get_attribute("warmup_epochs") or 0)

        # EMA (Exponential Moving Average of model weights)
        self.use_ema = bool(dd.get_attribute("use_ema") or False)
        self.ema_decay = float(dd.get_attribute("ema_decay") or 0.9999)
        self.ema_warmup_steps = int(dd.get_attribute("ema_warmup_steps") or 0)

        # Data augmentation (velocity-field-aware flips)
        self.augment = bool(dd.get_attribute("augment") or False)

        # UNet type: "concat" (Palette-style) or "film" (FiLM conditioning)
        self.unet_type = dd.get_attribute("unet_type") or "concat"

        # Spatiotemporal config
        self.T = int(dd.get_attribute("T") or 1)
        self.pretrained_spatial = dd.get_attribute("pretrained_spatial_checkpoint") or None
        self.freeze_spatial_epochs = int(dd.get_attribute("freeze_spatial_epochs") or 0)

        # Build inpainting model
        if self.unet_type == "film":
            unet = MyUNet_FiLM(n_steps=self.n_steps).to(self.device)
            logging.info("Using FiLM-conditioned UNet")
        elif self.unet_type == "film_attn":
            unet = MyUNet_FiLM_Attn(n_steps=self.n_steps).to(self.device)
            logging.info("Using FiLM-conditioned UNet with self-attention (film_attn)")
        elif self.unet_type == "standard":
            unet = MyUNet(n_steps=self.n_steps).to(self.device)
            unet.in_channels = 2  # expose for logging
            logging.info("Using unconditional UNet (standard, 2-channel)")
        elif self.unet_type == "standard_attn":
            unet = MyUNet_Attn(n_steps=self.n_steps).to(self.device)
            logging.info("Using unconditional UNet with self-attention (standard_attn, 2-channel)")
        elif self.unet_type == "standard_attn_slim":
            unet = MyUNet_Attn_Slim(n_steps=self.n_steps).to(self.device)
            logging.info("Using slim unconditional UNet with bottleneck attention + dropout (standard_attn_slim, 2-channel)")
        elif self.unet_type == "standard_attn_mid":
            unet = MyUNet_Attn_Mid(n_steps=self.n_steps).to(self.device)
            logging.info("Using mid-size unconditional UNet with level4+bottleneck attention + dropout (standard_attn_mid, 2-channel)")
        elif self.unet_type == "spatiotemporal":
            if self.pretrained_spatial:
                ckpt_path = Path(self.pretrained_spatial)
                if not ckpt_path.is_absolute():
                    ckpt_path = BASE_DIR / ckpt_path
                unet = MyUNet_ST.from_pretrained_spatial(
                    str(ckpt_path), T=self.T, n_steps=self.n_steps,
                ).to(self.device)
                logging.info(f"Loaded pretrained spatial weights from {ckpt_path}")
            else:
                unet = MyUNet_ST(n_steps=self.n_steps, T=self.T).to(self.device)
            logging.info(
                f"Using spatiotemporal UNet (T={self.T}, "
                f"spatial={unet.num_spatial_params:,}, "
                f"temporal={unet.num_temporal_params:,}, "
                f"total={unet.num_total_params:,})"
            )
        else:
            unet = MyUNet_Inpaint(n_steps=self.n_steps).to(self.device)
            logging.info("Using concat-conditioned UNet (Palette-style)")

        # Unconditional UNets: disable conditioning-related options
        self._unconditional_types = (
            "standard", "standard_attn", "standard_attn_slim",
            "standard_attn_mid", "spatiotemporal",
        )
        if self.unet_type in self._unconditional_types:
            if self.mask_xt:
                logging.warning("mask_xt is ignored for unet_type='%s' (no conditioning)", self.unet_type)
                self.mask_xt = False
            if self.p_uncond > 0:
                logging.warning("p_uncond is ignored for unet_type='%s' (no conditioning)", self.unet_type)
                self.p_uncond = 0.0

        image_chw = (self.T * 2, 64, 128) if self.unet_type == "spatiotemporal" else (2, 64, 128)
        self.ddpm = GaussianDDPM(
            unet,
            n_steps=self.n_steps,
            min_beta=self.min_beta,
            max_beta=self.max_beta,
            device=self.device,
            image_chw=image_chw,
        )

        # Wrap datasets: use OceanSequenceDataset for spatiotemporal,
        # OceanInpaintDataset for everything else
        if self.unet_type == "spatiotemporal":
            self.train_loader = DataLoader(
                OceanSequenceDataset(
                    data_tensor=dd.training_tensor,
                    n_steps=self.n_steps,
                    noise_strategy=self.noise_strategy,
                    transform=dd.get_transform(),
                    T=self.T,
                ),
                batch_size=self.batch_size,
                shuffle=True,
            )
            self.test_loader = DataLoader(
                OceanSequenceDataset(
                    data_tensor=dd.test_tensor,
                    n_steps=self.n_steps,
                    noise_strategy=self.noise_strategy,
                    transform=dd.get_transform(),
                    T=self.T,
                ),
                batch_size=self.batch_size,
            )
        else:
            self.train_loader = DataLoader(
                OceanInpaintDataset(
                    dd.get_training_data(),
                    standardizer=self.standardizer,
                    augment=self.augment,
                ),
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
        self.best_ema_weights = self.output_dir / f"{model_base}_best_ema_weights.pt"

        # Drive backup (auto-detected on Colab, or set drive_backup_dir in config)
        self.drive_backup_dir = self._setup_drive_backup(dd)

        # Save config
        self._save_config()

        # Optional: resume from checkpoint
        self.retrain_mode = dd.get_attribute("retrain_mode")
        self.model_to_retrain = dd.get_attribute("model_to_retrain")

    def _setup_drive_backup(self, dd):
        """Set up Google Drive backup directory for Colab resilience.

        If running on Colab with Drive mounted, automatically backs up
        checkpoints to Drive after every save so they survive runtime
        disconnects.  Can also be set explicitly via config key
        ``drive_backup_dir``.
        """
        explicit = dd.get_attribute("drive_backup_dir")
        if explicit:
            backup = Path(explicit)
        else:
            # Auto-detect Colab with mounted Drive
            drive_root = Path("/content/drive/MyDrive")
            if drive_root.exists():
                model_name = dd.get_attribute("model_name") or "unknown"
                backup = drive_root / "Ocean Inpainting" / "training_results" / model_name
            else:
                return None  # not on Colab / no Drive mounted

        backup.mkdir(parents=True, exist_ok=True)
        logging.info(f"Drive backup enabled → {backup}")
        return backup

    def _backup_to_drive(self, *paths):
        """Copy files to Drive backup dir (no-op if not on Colab)."""
        if self.drive_backup_dir is None:
            return
        for p in paths:
            p = Path(p)
            if p.exists():
                dst = self.drive_backup_dir / p.name
                try:
                    shutil.copy2(p, dst)
                except Exception:
                    logging.warning(f"Drive backup failed for {p.name}")

    def _save_config(self):
        import yaml
        config_path = self.output_dir / "config_used.yaml"
        with config_path.open("w") as f:
            yaml.dump(self.dd.get_full_config(), f)
        self._backup_to_drive(config_path)
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
                for batch in loader:
                    # Handle both 3-tuple (sequence) and 5-tuple (inpaint) loaders
                    if len(batch) == 3:
                        x0, t, noise = batch
                        mask = known = None
                    else:
                        x0, t, noise, mask, known = batch

                    x0 = x0.to(self.device)
                    t = t.to(self.device)
                    noise = noise.to(self.device)
                    n = len(x0)

                    # Forward: noise the clean image
                    noisy = self.ddpm(x0, t, noise)

                    if self.unet_type in self._unconditional_types:
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
        logging.info(f"LR schedule: {self.lr_schedule}, warmup: {self.warmup_epochs} epochs")
        logging.info(f"Weight decay: {self.weight_decay}")
        logging.info(f"EMA: {self.use_ema} (decay={self.ema_decay}, warmup_steps={self.ema_warmup_steps})")
        logging.info(f"Augmentation: {self.augment}")
        logging.info(f"Output dir: {self.output_dir}")
        if self.unet_type == "spatiotemporal":
            logging.info(f"Spatiotemporal: T={self.T}, freeze_spatial_epochs={self.freeze_spatial_epochs}")
            if self.pretrained_spatial:
                logging.info(f"Pretrained spatial checkpoint: {self.pretrained_spatial}")

        # Optimizer: AdamW if weight_decay > 0, else plain Adam
        if self.weight_decay > 0:
            optimizer = AdamW(self.ddpm.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            logging.info(f"Using AdamW (weight_decay={self.weight_decay})")
        else:
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

        # LR scheduler (built after optimizer restore so state is correct)
        scheduler = None
        if self.lr_schedule == "cosine":
            total_epochs = self.n_epochs
            if self.warmup_epochs > 0:
                warmup_sched = LinearLR(
                    optimizer,
                    start_factor=1e-3,
                    end_factor=1.0,
                    total_iters=self.warmup_epochs,
                )
                cosine_sched = CosineAnnealingLR(
                    optimizer,
                    T_max=total_epochs - self.warmup_epochs,
                    eta_min=self.lr * 0.01,  # min LR = 1% of peak
                )
                scheduler = SequentialLR(
                    optimizer,
                    schedulers=[warmup_sched, cosine_sched],
                    milestones=[self.warmup_epochs],
                )
            else:
                scheduler = CosineAnnealingLR(
                    optimizer,
                    T_max=total_epochs,
                    eta_min=self.lr * 0.01,
                )
            # Fast-forward scheduler if resuming (suppress benign warning)
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "Detected call of `lr_scheduler.step\\(\\)` before")
                for _ in range(start_epoch):
                    scheduler.step()
            logging.info(f"Cosine LR schedule: peak={self.lr}, warmup={self.warmup_epochs}, "
                         f"min={self.lr * 0.01:.6f}")

        if self.gradient_accumulation_steps > 1:
            effective_batch = self.batch_size * self.gradient_accumulation_steps
            logging.info(f"Gradient accumulation: {self.gradient_accumulation_steps} steps "
                         f"(effective batch size: {effective_batch})")

        # EMA
        ema = None
        if self.use_ema:
            ema = EMA(self.ddpm, decay=self.ema_decay, warmup_steps=self.ema_warmup_steps)
            # Restore EMA state if resuming
            if self.retrain_mode and self.model_to_retrain:
                path = Path(self.model_to_retrain)
                if path.exists():
                    ema_ckpt = torch.load(path, map_location=self.device, weights_only=False)
                    if "ema_state" in ema_ckpt:
                        ema.load_state_dict(ema_ckpt["ema_state"])
                        logging.info("Restored EMA state from checkpoint")
            logging.info(f"EMA enabled (decay={self.ema_decay}, warmup_steps={self.ema_warmup_steps})")

        # CSV header
        with self.csv_file.open("w", newline="") as f:
            if ema is not None:
                csv.writer(f).writerow(["Epoch", "Epoch Loss", "Train Loss", "Test Loss",
                                        "EMA Train Loss", "EMA Test Loss"])
            else:
                csv.writer(f).writerow(["Epoch", "Epoch Loss", "Train Loss", "Test Loss"])

        best_epoch = start_epoch
        accum_steps = self.gradient_accumulation_steps

        # ── Two-phase training for spatiotemporal UNet ───────────────
        # Phase 1: freeze spatial weights, train only temporal layers
        # Phase 2: unfreeze everything for end-to-end fine-tuning
        spatial_frozen = False
        if (self.unet_type == "spatiotemporal" and self.freeze_spatial_epochs > 0):
            self.ddpm.network.freeze_spatial()
            spatial_frozen = True
            logging.info(
                f"Phase 1: spatial weights frozen for first "
                f"{self.freeze_spatial_epochs} epochs"
            )

        try:
            for epoch in tqdm(
                range(start_epoch, start_epoch + self.n_epochs),
                desc="Training (inpaint)",
                colour="#00ff00",
            ):
                # Phase transition: unfreeze spatial weights after N epochs
                if spatial_frozen and epoch >= start_epoch + self.freeze_spatial_epochs:
                    self.ddpm.network.unfreeze_spatial()
                    spatial_frozen = False
                    logging.info(
                        f"Phase 2: unfroze spatial weights at epoch {epoch + 1}"
                    )

                epoch_loss = 0.0
                self.ddpm.train()
                optimizer.zero_grad()  # zero once at start of epoch

                for batch_idx, batch in enumerate(
                    tqdm(
                        self.train_loader,
                        leave=False,
                        desc=f"Epoch {epoch + 1}/{start_epoch + self.n_epochs}",
                        colour="#005500",
                    )
                ):
                    # Handle both 3-tuple (sequence) and 5-tuple (inpaint) loaders
                    if len(batch) == 3:
                        x0, t, noise = batch
                        mask = known = None
                    else:
                        x0, t, noise, mask, known = batch

                    n = len(x0)
                    x0 = x0.to(self.device)
                    t = t.to(self.device)
                    noise = noise.to(self.device)

                    # Forward diffusion: add noise to x0
                    noisy = self.ddpm(x0, t, noise)

                    if self.unet_type in self._unconditional_types:
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

                    # Scale loss for gradient accumulation
                    scaled_loss = loss / accum_steps
                    scaled_loss.backward()

                    # Step optimizer every accum_steps microbatches (or at end of epoch)
                    if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                        if self.max_grad_norm > 0:
                            torch.nn.utils.clip_grad_norm_(
                                self.ddpm.parameters(), self.max_grad_norm
                            )
                        optimizer.step()
                        optimizer.zero_grad()

                        # Update EMA after each optimizer step
                        if ema is not None:
                            ema.update()

                    epoch_loss += loss.item() * n / len(self.train_loader.dataset)

                # Step LR scheduler (once per epoch)
                if scheduler is not None:
                    scheduler.step()

                # Evaluate with RAW weights (used for checkpoint selection)
                self.ddpm.eval()
                avg_train_loss = self.evaluate(self.train_loader)
                avg_test_loss = self.evaluate(self.test_loader, fixed_seed=12345)

                # Also evaluate with EMA weights (logged separately)
                ema_train_loss = None
                ema_test_loss = None
                if ema is not None:
                    ema.apply()
                    ema_train_loss = self.evaluate(self.train_loader)
                    ema_test_loss = self.evaluate(self.test_loader, fixed_seed=12345)
                    ema.restore()

                epoch_losses.append(epoch_loss)
                train_losses.append(avg_train_loss)
                test_losses.append(avg_test_loss)

                # CSV logging
                try:
                    with self.csv_file.open("a", newline="") as f:
                        row = [epoch + 1, epoch_loss, avg_train_loss, avg_test_loss]
                        if ema is not None:
                            row.extend([ema_train_loss, ema_test_loss])
                        csv.writer(f).writerow(row)
                except Exception:
                    pass

                self.ddpm.train()

                # Get current LR for logging
                current_lr = optimizer.param_groups[0]['lr']

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
                if ema is not None:
                    checkpoint["ema_state"] = ema.state_dict()

                ema_str = ""
                if ema is not None:
                    ema_str = f", ema_test={ema_test_loss:.7f}"
                log_str = (
                    f"\nEpoch {epoch + 1}: epoch_loss={epoch_loss:.7f}, "
                    f"train={avg_train_loss:.7f}, test={avg_test_loss:.7f}"
                    f"{ema_str}, lr={current_lr:.6f}"
                )

                if avg_test_loss < best_test_loss:
                    best_test_loss = avg_test_loss
                    best_epoch = epoch
                    torch.save(self.ddpm.state_dict(), self.best_weights)
                    torch.save(checkpoint, self.best_checkpoint)
                    # Save EMA weights separately for inference
                    if ema is not None:
                        ema.apply()
                        torch.save(self.ddpm.state_dict(), self.best_ema_weights)
                        ema.restore()
                    self._backup_to_drive(
                        self.best_weights, self.best_checkpoint, self.csv_file
                    )
                    log_str += "\033[32m  --> BEST\033[0m"
                else:
                    torch.save(checkpoint, self.latest_checkpoint)
                    self._backup_to_drive(self.latest_checkpoint, self.csv_file)

                log_str += f"  (best={best_test_loss:.7f} @ epoch {best_epoch + 1})"
                tqdm.write(log_str)

        except KeyboardInterrupt:
            logging.error(get_death_message())
        finally:
            self._plot(epoch_losses, train_losses, test_losses)
            self._backup_to_drive(self.plot_file, self.csv_file)

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
