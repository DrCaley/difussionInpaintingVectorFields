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
from data_prep.ocean_gp_forward_dataset import OceanGPForwardDataset
from data_prep.ocean_gp_context_dataset import OceanGPContextDataset
from data_prep.ocean_gp_multimask_dataset import OceanGPMultiMaskDataset
from data_prep.ocean_gp_onthefly_dataset import OceanGPOnTheFlyDataset
from data_prep.ocean_voronoi_forward_dataset import OceanVoronoiForwardDataset
from ddpm.neural_networks.ddpm import GaussianDDPM
from ddpm.helper_functions.interpolation_tool import gp_fill
from ddpm.neural_networks.unets.unet_inpaint import MyUNet_Inpaint
from ddpm.neural_networks.unets.unet_film import MyUNet_FiLM
from ddpm.neural_networks.unets.unet_xl import MyUNet
from ddpm.neural_networks.unets.unet_xl_attn import MyUNet_Attn
from ddpm.neural_networks.unets.unet_attn_slim import MyUNet_Attn_Slim
from ddpm.neural_networks.unets.unet_attn_mid import MyUNet_Attn_Mid
from ddpm.neural_networks.unets.unet_film_attn import MyUNet_FiLM_Attn
from ddpm.neural_networks.unets.unet_helmholtz import MyUNet_Helmholtz
from ddpm.neural_networks.unets.unet_helmholtz_split import MyUNet_Helmholtz_Split
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
        self.loss_strategy = dd.get_loss_strategy().to(self.device)
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

        # UNet type: "concat" (Palette-style) or "film" (FiLM conditioning)
        # Must be set early — GP-context branch below depends on it.
        self.unet_type = dd.get_attribute("unet_type") or "concat"

        # GP-forward training: noise GP posterior instead of GT so the
        # model learns to refine GP reconstructions at inference time.
        # Requires prediction_target='x0' (target is still ground truth).
        self.gp_forward = bool(dd.get_attribute("gp_forward") or False)

        # GP-conditioned: noise GT normally (standard DDPM), but condition
        # on the full GP field via the known_values channels.  This perfectly
        # aligns training and inference distributions.
        self.gp_conditioned = bool(dd.get_attribute("gp_conditioned") or False)

        # Voronoi-forward training: noise Voronoi (nearest-neighbour) fill
        # instead of GT, predict x0.  Analogous to gp_forward but much
        # cheaper — Voronoi is computed on-the-fly with random masks.
        self.voronoi_forward = bool(dd.get_attribute("voronoi_forward") or False)
        self.voronoi_known_fracs = dd.get_attribute("voronoi_known_fracs") or [0.001, 0.005, 0.01, 0.02, 0.05]

        # Helmholtz-split noise schedule: independent noise schedules for
        # solenoidal and irrotational subspaces (Proposal 6)
        self.helmholtz_split_noise = bool(dd.get_attribute("helmholtz_split_noise") or False)
        self.irr_speed = float(dd.get_attribute("irr_speed") or 2.0)
        self.bootstrap_rollout_training = bool(dd.get_attribute("bootstrap_rollout_training") or False)
        rollout_probs_raw = dd.get_attribute("bootstrap_rollout_depth_probs") or {0: 1.0}
        self.bootstrap_rollout_depth_probs = {}
        for depth, prob in rollout_probs_raw.items():
            depth_i = int(depth)
            prob_f = float(prob)
            if depth_i < 0:
                raise ValueError(f"bootstrap rollout depth must be >= 0, got {depth_i}")
            if prob_f < 0:
                raise ValueError(f"bootstrap rollout probability must be >= 0, got {prob_f}")
            self.bootstrap_rollout_depth_probs[depth_i] = prob_f
        total_rollout_prob = sum(self.bootstrap_rollout_depth_probs.values())
        if total_rollout_prob <= 0:
            raise ValueError("bootstrap_rollout_depth_probs must sum to > 0")
        self.bootstrap_rollout_depth_probs = {
            depth: prob / total_rollout_prob
            for depth, prob in sorted(self.bootstrap_rollout_depth_probs.items())
        }

        # Detached stage rollout training: repeated one-step denoising on
        # self-generated sources, with graph cuts between outer stages.
        self.detached_rollout_training = bool(dd.get_attribute("detached_rollout_training") or False)
        self.rollout_num_stages = int(dd.get_attribute("rollout_num_stages") or 1)
        self.rollout_stage_aware = bool(dd.get_attribute("rollout_stage_aware") or False)
        repaste_known_raw = dd.get_attribute("rollout_repaste_known")
        self.rollout_repaste_known = True if repaste_known_raw is None else bool(repaste_known_raw)

        rollout_weights_raw = dd.get_attribute("rollout_stage_loss_weights")
        if rollout_weights_raw is None:
            rollout_weights_raw = [1.0] * self.rollout_num_stages
        self.rollout_stage_loss_weights = [float(w) for w in rollout_weights_raw]

        rollout_t_fracs_raw = dd.get_attribute("rollout_stage_t_max_fractions")
        if rollout_t_fracs_raw is None:
            rollout_t_fracs_raw = [1.0] * self.rollout_num_stages
        self.rollout_stage_t_max_fractions = [float(v) for v in rollout_t_fracs_raw]

        self.init_from_weights = dd.get_attribute("init_from_weights") or None

        # Self-conditioning: feed previous x0 prediction back as input
        self.self_conditioning = bool(dd.get_attribute("self_conditioning") or False)
        self.p_self_cond = float(dd.get_attribute("p_self_cond") or 0.5)

        if self.rollout_num_stages < 1:
            raise ValueError(f"rollout_num_stages must be >= 1, got {self.rollout_num_stages}")
        if len(self.rollout_stage_loss_weights) != self.rollout_num_stages:
            raise ValueError(
                "rollout_stage_loss_weights length must match rollout_num_stages "
                f"({len(self.rollout_stage_loss_weights)} != {self.rollout_num_stages})"
            )
        if len(self.rollout_stage_t_max_fractions) != self.rollout_num_stages:
            raise ValueError(
                "rollout_stage_t_max_fractions length must match rollout_num_stages "
                f"({len(self.rollout_stage_t_max_fractions)} != {self.rollout_num_stages})"
            )
        if any(w < 0 for w in self.rollout_stage_loss_weights):
            raise ValueError("rollout_stage_loss_weights must be non-negative")
        if sum(self.rollout_stage_loss_weights) <= 0:
            raise ValueError("rollout_stage_loss_weights must sum to > 0")
        self.rollout_stage_loss_weights = [
            w / sum(self.rollout_stage_loss_weights)
            for w in self.rollout_stage_loss_weights
        ]
        if any((v <= 0 or v > 1.0) for v in self.rollout_stage_t_max_fractions):
            raise ValueError("rollout_stage_t_max_fractions entries must lie in (0, 1]")

        # GP-context UNet auto-enables GP-conditioned (needs GP cache)
        # but supports both eps and x0 prediction targets (concat-style).
        if self.unet_type == "gp_context":
            self.gp_conditioned = True

        if self.voronoi_forward:
            assert self.prediction_target == "x0", (
                "voronoi_forward training requires prediction_target='x0' — "
                "eps-prediction would recover the Voronoi field, not ground truth"
            )

        if self.helmholtz_split_noise:
            assert self.prediction_target == "x0", (
                "helmholtz_split_noise requires prediction_target='x0' — "
                "eps-prediction is undefined with dual noise schedules"
            )

        if self.bootstrap_rollout_training:
            assert self.voronoi_forward, (
                "bootstrap_rollout_training is currently supported only with voronoi_forward"
            )
            assert self.prediction_target == "x0", (
                "bootstrap_rollout_training requires prediction_target='x0'"
            )

        if self.detached_rollout_training:
            assert self.voronoi_forward, (
                "detached_rollout_training currently requires voronoi_forward"
            )
            assert self.prediction_target == "x0", (
                "detached_rollout_training requires prediction_target='x0'"
            )
            assert self.unet_type in ("standard_attn", "standard_attn_slim", "standard_attn_mid"), (
                "detached_rollout_training currently supports unconditional attention UNets only"
            )
            assert not self.bootstrap_rollout_training, (
                "detached_rollout_training and bootstrap_rollout_training are mutually exclusive"
            )

        if self.gp_forward or self.gp_conditioned:
            # Path to pre-computed GP fields (.pt file from scripts/precompute_gp.py)
            # Set to "on_the_fly" to skip precompute and generate random masks each epoch
            gp_cache_default = str(BASE_DIR / "data" / "rams_head" / "gp_precomputed.pt")
            self.gp_cache_path = dd.get_attribute("gp_cache_path") or gp_cache_default
            self.gp_random_masks = (self.gp_cache_path == "on_the_fly")

            # GP params for on-the-fly computation
            self.gp_lengthscale = float(dd.get_attribute("gp_lengthscale") or 14.1)
            self.gp_variance = float(dd.get_attribute("gp_variance") or 0.0103420345)
            self.gp_noise = float(dd.get_attribute("gp_noise") or 1e-8)
            self.gp_kernel_type = dd.get_attribute("gp_kernel_type") or "rbf_legacy"
            self.gp_coord_system = dd.get_attribute("gp_coord_system") or "pixels"

            # Known fractions to sample from during on-the-fly mask generation
            self.gp_known_fracs = dd.get_attribute("gp_known_fracs") or [0.001, 0.01, 0.05, 0.10]

            # DataLoader workers for on-the-fly GP computation
            self.gp_num_workers = int(dd.get_attribute("gp_num_workers") or 16)

        if (self.gp_forward or self.gp_conditioned) and self.unet_type != "gp_context" and not getattr(self, 'gp_random_masks', False):
            assert self.prediction_target == "x0", (
                "gp_forward/gp_conditioned training requires prediction_target='x0' — "
                "eps-prediction would recover the GP field, not ground truth"
            )

        # Gradient accumulation: simulate larger batches without more memory
        self.gradient_accumulation_steps = int(dd.get_attribute("gradient_accumulation_steps") or 1)

        # Gradient clipping (max L2 norm); 0 = disabled
        self.max_grad_norm = float(dd.get_attribute("max_grad_norm") or 0)

        # Weight decay for AdamW (0 = plain Adam)
        self.weight_decay = float(dd.get_attribute("weight_decay") or 0)

        # Learning rate schedule
        self.lr_schedule = dd.get_attribute("lr_schedule") or "constant"  # constant | cosine

        # LR warmup (used with cosine schedule, independent of EMA)
        self.warmup_epochs = int(dd.get_attribute("warmup_epochs") or 0)

        # EMA (Exponential Moving Average of model weights)
        self.use_ema = bool(dd.get_attribute("use_ema") or False)
        self.ema_decay = float(dd.get_attribute("ema_decay") or 0.9999)
        self.ema_warmup_steps = int(dd.get_attribute("ema_warmup_steps") or 0)

        # Data augmentation (velocity-field-aware flips)
        self.augment = bool(dd.get_attribute("augment") or False)

        # Spatiotemporal config
        self.T = int(dd.get_attribute("T") or 1)
        self.pretrained_spatial = dd.get_attribute("pretrained_spatial_checkpoint") or None
        self.freeze_spatial_epochs = int(dd.get_attribute("freeze_spatial_epochs") or 0)
        # Differential LR: spatial_lr_factor scales lr for pretrained spatial params after unfreeze
        self.spatial_lr_factor = float(dd.get_attribute("spatial_lr_factor") or 0.01)
        # Temporal dropout: regularization for temporal layers in ST UNet
        self.temporal_dropout = float(dd.get_attribute("temporal_dropout") or 0.0)
        # Temporal mode: 'full' (conv+attn, ~2.4M) or 'lite' (bottleneck conv, ~190K)
        self.temporal_mode = dd.get_attribute("temporal_mode") or "full"
        # Bottleneck reduction factor for lite mode (default 4)
        self.temporal_reduction = int(dd.get_attribute("temporal_reduction") or 4)
        # Max evaluation batches per loader (0 = evaluate all)
        self.eval_max_batches = int(dd.get_attribute("eval_max_batches") or 0)

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
            unet = MyUNet_Attn(
                n_steps=self.n_steps,
                n_stage_tokens=self.rollout_num_stages if self.rollout_stage_aware else 0,
                self_cond_channels=2 if self.self_conditioning else 0,
            ).to(self.device)
            logging.info("Using unconditional UNet with self-attention (standard_attn, 2-channel%s)",
                         ", self-cond" if self.self_conditioning else "")
        elif self.unet_type == "standard_attn_slim":
            unet = MyUNet_Attn_Slim(n_steps=self.n_steps).to(self.device)
            logging.info("Using slim unconditional UNet with bottleneck attention + dropout (standard_attn_slim, 2-channel)")
        elif self.unet_type == "standard_attn_mid":
            unet = MyUNet_Attn_Mid(n_steps=self.n_steps).to(self.device)
            logging.info("Using mid-size unconditional UNet with level4+bottleneck attention + dropout (standard_attn_mid, 2-channel)")
        elif self.unet_type == "gp_context":
            unet = MyUNet_Attn(n_steps=self.n_steps, in_channels=8).to(self.device)
            logging.info(
                "Using GP-context conditioned UNet with self-attention "
                "(8ch: x_t[2]+mask[1]+gp_mean[2]+gp_var[1]+dist[1]+ocean[1])"
            )
        elif self.unet_type == "helmholtz":
            unet = MyUNet_Helmholtz(
                n_steps=self.n_steps,
                n_stage_tokens=self.rollout_num_stages if self.rollout_stage_aware else 0,
                self_cond_channels=2 if self.self_conditioning else 0,
            ).to(self.device)
            logging.info("Using Helmholtz dual-head UNet (ψ + φ → curl + grad, 2-channel%s)",
                         ", self-cond" if self.self_conditioning else "")
        elif self.unet_type == "helmholtz_split":
            unet = MyUNet_Helmholtz_Split(
                n_steps=self.n_steps,
                n_stage_tokens=self.rollout_num_stages if self.rollout_stage_aware else 0,
                self_cond_channels=2 if self.self_conditioning else 0,
            ).to(self.device)
            logging.info("Using Helmholtz split-decoder UNet (independent high-res decoders per head, 2-channel%s)",
                         ", self-cond" if self.self_conditioning else "")
        elif self.unet_type == "spatiotemporal":
            if self.pretrained_spatial:
                ckpt_path = Path(self.pretrained_spatial)
                if not ckpt_path.is_absolute():
                    ckpt_path = BASE_DIR / ckpt_path
                unet = MyUNet_ST.from_pretrained_spatial(
                    str(ckpt_path), T=self.T, n_steps=self.n_steps,
                    temporal_dropout=self.temporal_dropout,
                    temporal_mode=self.temporal_mode,
                    temporal_reduction=self.temporal_reduction,
                ).to(self.device)
                logging.info(f"Loaded pretrained spatial weights from {ckpt_path}")
            else:
                unet = MyUNet_ST(
                    n_steps=self.n_steps, T=self.T,
                    temporal_dropout=self.temporal_dropout,
                    temporal_mode=self.temporal_mode,
                    temporal_reduction=self.temporal_reduction,
                ).to(self.device)
            logging.info(
                f"Using spatiotemporal UNet (T={self.T}, mode={self.temporal_mode}, "
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
            "standard_attn_mid", "helmholtz", "helmholtz_split", "spatiotemporal",
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

        # Helmholtz-split schedule (after DDPM so device is set)
        if self.helmholtz_split_noise:
            from ddpm.utils.helmholtz_split import HelmholtzSplitSchedule
            self.split_schedule = HelmholtzSplitSchedule(
                n_steps=self.n_steps,
                min_beta=self.min_beta,
                max_beta=self.max_beta,
                irr_speed=self.irr_speed,
                device=self.device,
            )
            logging.info(
                f"Helmholtz-split noise: irr_speed={self.irr_speed}, "
                f"α_bar_sol[T-1]={self.split_schedule.alpha_bars_sol[-1]:.4f}, "
                f"α_bar_irr[T-1]={self.split_schedule.alpha_bars_irr[-1]:.4f}"
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
            if self.gp_forward or self.gp_conditioned:
                # ── On-the-fly random mask mode (no precompute needed) ────
                if self.gp_random_masks and self.unet_type == "gp_context":
                    gp_params = {
                        "lengthscale": self.gp_lengthscale,
                        "variance": self.gp_variance,
                        "noise": self.gp_noise,
                        "kernel_type": self.gp_kernel_type,
                        "coord_system": self.gp_coord_system,
                    }
                    known_fracs = self.gp_known_fracs
                    num_workers = self.gp_num_workers
                    logging.info(
                        f"  ON-THE-FLY GP mode: random masks every sample access\n"
                        f"    Known fracs: {known_fracs}\n"
                        f"    GP params: ls={gp_params['lengthscale']}, "
                        f"var={gp_params['variance']}, noise={gp_params['noise']}, "
                        f"kernel={gp_params['kernel_type']}\n"
                        f"    DataLoader workers: {num_workers}"
                    )
                    # Raw (unstandardized) data for GP regression
                    # dd.training_tensor is stored as (W=94, H=44, C=2, N)
                    # Reshape to (N, 2, 64, 128) with NaN→0 and zero-padding
                    def _reshape_raw(tensor_whcn):
                        t = tensor_whcn.permute(3, 2, 1, 0)   # (N, 2, 44, 94)
                        t = torch.nan_to_num(t, nan=0.0)
                        padded = torch.zeros(t.shape[0], 2, 64, 128, dtype=t.dtype)
                        padded[:, :, :44, :94] = t
                        return padded

                    raw_train = _reshape_raw(dd.training_tensor)
                    raw_test = _reshape_raw(dd.test_tensor)
                    logging.info(
                        f"    Raw data: train={raw_train.shape}, test={raw_test.shape}"
                    )
                    self.train_loader = DataLoader(
                        OceanGPOnTheFlyDataset(
                            dd.get_training_data(),
                            raw_train,
                            self.standardizer,
                            gp_params,
                            known_fracs=known_fracs,
                            augment=self.augment,
                        ),
                        batch_size=self.batch_size,
                        shuffle=True,
                        num_workers=num_workers,
                        persistent_workers=True if num_workers > 0 else False,
                        prefetch_factor=2 if num_workers > 0 else None,
                    )
                    self.test_loader = DataLoader(
                        OceanGPOnTheFlyDataset(
                            dd.get_test_data(),
                            raw_test,
                            self.standardizer,
                            gp_params,
                            known_fracs=known_fracs,
                            augment=False,
                        ),
                        batch_size=self.batch_size,
                        num_workers=num_workers,
                        persistent_workers=True if num_workers > 0 else False,
                        prefetch_factor=2 if num_workers > 0 else None,
                    )
                else:
                    # ── Precomputed GP cache mode ─────────────────────────
                    gp_cache = Path(self.gp_cache_path)
                    if not gp_cache.exists():
                        raise FileNotFoundError(
                            f"GP cache not found at {gp_cache}. "
                            "Run: PYTHONPATH=. python3 scripts/precompute_gp.py\n"
                            "Or set gp_cache_path: on_the_fly for random mask training"
                        )
                    logging.info(f"Loading pre-computed GP fields from {gp_cache}")
                    gp_data = torch.load(gp_cache, map_location="cpu", weights_only=False)

                    # Detect multi-mask vs single-mask cache format early
                    is_multimask = "mask_configs" in gp_data

                    if is_multimask:
                        # Multi-mask: gp_train/gp_test are LISTS of tensors
                        gp_train_raw = gp_data["gp_train"][0]  # first mask for shape log
                        gp_test_raw = gp_data["gp_test"][0]
                        n_masks = gp_data["n_masks"]
                        logging.info(
                            f"  Multi-mask GP cache: {n_masks} masks, "
                            f"fracs={gp_data.get('known_fracs', 'N/A')}, "
                            f"train shape per mask: {gp_train_raw.shape}"
                        )
                    else:
                        gp_train_raw = gp_data["gp_train"]  # (N_train, 2, H, W) raw
                        gp_test_raw = gp_data["gp_test"]     # (N_test, 2, H, W) raw
                        gp_mask = gp_data["mask_1ch"]         # (1, 1, H, W)
                        logging.info(
                            f"  GP train: {gp_train_raw.shape}, "
                            f"GP test: {gp_test_raw.shape}, "
                            f"mask coverage: {(gp_mask == 0).sum().item()} known pixels"
                        )
                    logging.info(f"  GP mode: {'gp_context' if self.unet_type == 'gp_context' else 'gp_conditioned' if self.gp_conditioned else 'gp_forward'}")

                    if self.unet_type == "gp_context":
                        if is_multimask:
                            # Multi-mask cache: lists of GP fields per mask
                            n_masks = gp_data["n_masks"]
                            mask_configs = gp_data["mask_configs"]
                            logging.info(
                                f"  Multi-mask GP cache: {n_masks} masks, "
                                f"fracs={gp_data.get('known_fracs', 'N/A')}"
                            )
                            for i, cfg in enumerate(mask_configs):
                                logging.info(
                                    f"    Mask {i}: {cfg['known_frac']*100:.1f}% known, "
                                    f"{cfg['n_known']} pixels"
                                )

                            gp_train_list = gp_data["gp_train"]    # list of (N, 2, H, W)
                            gp_test_list = gp_data["gp_test"]
                            var_train_list = gp_data["var_train"]
                            var_test_list = gp_data["var_test"]
                            masks_1ch = [cfg["mask_1ch"] for cfg in mask_configs]
                            dist_maps = [cfg["dist_map"] for cfg in mask_configs]

                            self.train_loader = DataLoader(
                                OceanGPMultiMaskDataset(
                                    dd.get_training_data(),
                                    gp_train_list,
                                    var_train_list,
                                    self.standardizer,
                                    masks_1ch,
                                    dist_maps,
                                    augment=self.augment,
                                ),
                                batch_size=self.batch_size,
                                shuffle=True,
                            )
                            self.test_loader = DataLoader(
                                OceanGPMultiMaskDataset(
                                    dd.get_test_data(),
                                    gp_test_list,
                                    var_test_list,
                                    self.standardizer,
                                    masks_1ch,
                                    dist_maps,
                                ),
                                batch_size=self.batch_size,
                            )
                        else:
                            # Single-mask cache (legacy format)
                            var_train_raw = gp_data["var_train"]
                            var_test_raw = gp_data["var_test"]
                            dist_map = gp_data["dist_map"]
                            logging.info(
                                f"  GP-context extras: var_train={var_train_raw.shape}, "
                                f"dist_map={dist_map.shape}"
                            )
                            self.train_loader = DataLoader(
                                OceanGPContextDataset(
                                    dd.get_training_data(),
                                    gp_train_raw,
                                    var_train_raw,
                                    self.standardizer,
                                    gp_mask,
                                    dist_map,
                                    augment=self.augment,
                                ),
                                batch_size=self.batch_size,
                                shuffle=True,
                            )
                            self.test_loader = DataLoader(
                                OceanGPContextDataset(
                                    dd.get_test_data(),
                                    gp_test_raw,
                                    var_test_raw,
                                    self.standardizer,
                                    gp_mask,
                                    dist_map,
                                ),
                                batch_size=self.batch_size,
                            )
                    else:
                        self.train_loader = DataLoader(
                            OceanGPForwardDataset(
                                dd.get_training_data(),
                                gp_train_raw,
                                self.standardizer,
                                gp_mask,
                                gp_conditioned=self.gp_conditioned,
                                augment=self.augment,
                            ),
                            batch_size=self.batch_size,
                            shuffle=True,
                        )
                        self.test_loader = DataLoader(
                            OceanGPForwardDataset(
                                dd.get_test_data(),
                                gp_test_raw,
                                self.standardizer,
                                gp_mask,
                                gp_conditioned=self.gp_conditioned,
                            ),
                            batch_size=self.batch_size,
                        )
            elif self.voronoi_forward:
                # ── Voronoi-forward: on-the-fly Voronoi fill with random masks ──
                def _reshape_raw_vor(tensor_whcn):
                    t = tensor_whcn.permute(3, 2, 1, 0)  # (N, 2, 44, 94)
                    t = torch.nan_to_num(t, nan=0.0)
                    padded = torch.zeros(t.shape[0], 2, 64, 128, dtype=t.dtype)
                    padded[:, :, :44, :94] = t
                    return padded

                raw_train = _reshape_raw_vor(dd.training_tensor)
                raw_test = _reshape_raw_vor(dd.test_tensor)
                logging.info(
                    f"  VORONOI-FORWARD mode: on-the-fly Voronoi fill\n"
                    f"    Known fracs: {self.voronoi_known_fracs}\n"
                    f"    Raw data: train={raw_train.shape}, test={raw_test.shape}"
                )
                self.train_loader = DataLoader(
                    OceanVoronoiForwardDataset(
                        dd.get_training_data(),
                        raw_train,
                        self.standardizer,
                        known_fracs=self.voronoi_known_fracs,
                        augment=self.augment,
                    ),
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=8,
                    persistent_workers=True,
                    prefetch_factor=4,
                )
                self.test_loader = DataLoader(
                    OceanVoronoiForwardDataset(
                        dd.get_test_data(),
                        raw_test,
                        self.standardizer,
                        known_fracs=self.voronoi_known_fracs,
                    ),
                    batch_size=self.batch_size,
                    num_workers=4,
                    persistent_workers=True,
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

    def _compute_gp_batch(self, x0_batch, mask_batch):
        """Compute GP posterior for each sample in a batch.

        For each sample: unstandardize x0 → GP fill in raw space →
        re-standardize.  The GP uses the known observations and mask
        to produce a smooth posterior mean, which is then used as the
        "clean" image to noise (instead of the ground truth).

        Args:
            x0_batch: (N, 2, H, W) standardized ground truth
            mask_batch: (N, 1, H, W) mask, 1=missing 0=known

        Returns:
            gp_std_batch: (N, 2, H, W) standardized GP posterior mean
        """
        N = x0_batch.shape[0]
        gp_batch = torch.zeros_like(x0_batch)

        for i in range(N):
            # Unstandardize to raw physical space
            x0_raw = self.standardizer.unstandardize(
                x0_batch[i].cpu()
            )  # (2, H, W)

            # Build 2-channel mask: (1, 2, H, W)
            mask_1ch = mask_batch[i].cpu()  # (1, H, W)
            mask_2ch = mask_1ch.expand(2, -1, -1).unsqueeze(0)  # (1, 2, H, W)

            # GP fill in raw space
            gp_raw = gp_fill(
                x0_raw.unsqueeze(0),  # (1, 2, H, W)
                mask_2ch,
                lengthscale=self.gp_lengthscale,
                variance=self.gp_variance,
                noise=self.gp_noise,
                use_double=True,
                kernel_type=self.gp_kernel_type,
                coord_system=self.gp_coord_system,
                return_variance=False,
            )  # (1, 2, H, W)

            # Re-standardize and store
            gp_std = self.standardizer(gp_raw.squeeze(0))  # (2, H, W)
            gp_batch[i] = gp_std

        return gp_batch.to(x0_batch.device)

    def _sample_bootstrap_rollout_depth(self):
        depths = list(self.bootstrap_rollout_depth_probs.keys())
        weights = list(self.bootstrap_rollout_depth_probs.values())
        return random.choices(depths, weights=weights, k=1)[0]

    def _build_stage_tensor(self, batch_size, stage_idx=None):
        if not self.rollout_stage_aware:
            return None
        if stage_idx is None:
            stage_idx = 0
        stage_idx = int(stage_idx)
        stage_idx = max(0, min(stage_idx, self.rollout_num_stages - 1))
        return torch.full((batch_size,), stage_idx, device=self.device, dtype=torch.long)

    def _sample_rollout_stage_t(self, batch_size, stage_idx):
        max_frac = self.rollout_stage_t_max_fractions[min(stage_idx, len(self.rollout_stage_t_max_fractions) - 1)]
        max_t = int(round(max_frac * (self.n_steps - 1)))
        max_t = max(0, min(max_t, self.n_steps - 1))
        return torch.randint(0, max_t + 1, (batch_size,), device=self.device)

    def _prediction_loss(self, pred, x0, noise, noisy, t, mask=None, epoch=0):
        target = x0 if self.prediction_target == "x0" else noise

        if self.mask_xt:
            mask_2ch = mask[:, :2]
            diff = (pred - target) ** 2
            masked_diff = diff * mask_2ch
            return masked_diff.sum() / mask_2ch.sum().clamp(min=1.0)

        return self.loss_strategy(
            pred,
            target,
            noisy,
            x0=x0,
            t=t,
            ddpm=self.ddpm,
            prediction_target=self.prediction_target,
            epoch=epoch,
        )

    def _predict_denoised(self, noisy, t, mask=None, known=None, stage_idx=None, self_cond=None):
        n = len(noisy)
        stage_tensor = self._build_stage_tensor(n, stage_idx=stage_idx)

        if self.unet_type in self._unconditional_types:
            return self.ddpm.network(noisy, t.reshape(n, -1), stage=stage_tensor, self_cond=self_cond)

        if mask is None or known is None:
            raise ValueError("Conditional prediction requires mask and known tensors")

        noisy_in = noisy
        if self.mask_xt:
            known_mask = 1.0 - mask
            indep_noise = torch.randn_like(noisy_in)
            noisy_in = noisy_in * mask + indep_noise * known_mask

        x_cond = torch.cat([noisy_in, mask, known], dim=1)
        return self.ddpm.network(x_cond, t.reshape(n, -1), stage=stage_tensor, self_cond=self_cond)

    def _compute_detached_rollout_loss(self, x0, source, mask, known, epoch=0):
        stage_source = source
        total_loss = torch.zeros((), device=self.device)
        n = len(x0)

        for stage_idx in range(self.rollout_num_stages):
            stage_t = self._sample_rollout_stage_t(n, stage_idx)
            if self.helmholtz_split_noise:
                noisy, _, _ = self.split_schedule.q_sample(stage_source, stage_t)
                stage_noise = None  # unused for x0 prediction
            else:
                stage_noise = self.noise_strategy(stage_source, stage_t).to(self.device)
                noisy = self.ddpm(stage_source, stage_t, stage_noise)

            # Self-conditioning: two-pass with probability p_self_cond
            sc = None
            if self.self_conditioning and random.random() < self.p_self_cond:
                with torch.no_grad():
                    first_pass = self._predict_denoised(
                        noisy, stage_t, mask=mask, known=known,
                        stage_idx=stage_idx, self_cond=None,
                    )
                    sc = first_pass.detach()

            pred = self._predict_denoised(
                noisy,
                stage_t,
                mask=mask,
                known=known,
                stage_idx=stage_idx,
                self_cond=sc,
            )

            stage_loss = self._prediction_loss(
                pred,
                x0,
                stage_noise,
                noisy,
                stage_t,
                mask=mask,
                epoch=epoch,
            )
            total_loss = total_loss + self.rollout_stage_loss_weights[stage_idx] * stage_loss

            if stage_idx + 1 < self.rollout_num_stages:
                stage_source = pred.detach()
                if self.rollout_repaste_known and mask is not None and known is not None:
                    stage_source = stage_source * mask + known

        return total_loss

    def _unpack_batch(self, batch):
        if len(batch) == 3:
            x0, t, noise = batch
            mask = known = gp_source = None
        elif len(batch) == 6:
            x0, gp_source, t, noise, mask, known = batch
        else:
            x0, t, noise, mask, known = batch
            gp_source = None

        x0 = x0.to(self.device)
        t = t.to(self.device)
        noise = noise.to(self.device)
        if mask is not None:
            mask = mask.to(self.device)
        if known is not None:
            known = known.to(self.device)
        if gp_source is not None:
            gp_source = gp_source.to(self.device)

        return x0, t, noise, mask, known, gp_source

    def _compute_batch_loss(self, batch, epoch=0):
        x0, t, noise, mask, known, gp_source = self._unpack_batch(batch)
        n = len(x0)

        source_for_diffusion = None
        if gp_source is not None:
            source_for_diffusion = gp_source
            if self.bootstrap_rollout_training:
                rollout_depth = self._sample_bootstrap_rollout_depth()
                source_for_diffusion = self._bootstrap_rollout_source(
                    source_for_diffusion,
                    t,
                    mask=mask,
                    known=known,
                    depth=rollout_depth,
                )

        if self.detached_rollout_training and source_for_diffusion is not None:
            loss = self._compute_detached_rollout_loss(
                x0,
                source_for_diffusion,
                mask,
                known,
                epoch=epoch,
            )
            return loss, n

        if self.helmholtz_split_noise:
            source = source_for_diffusion if (source_for_diffusion is not None and not self.gp_conditioned) else x0
            noisy, _, _ = self.split_schedule.q_sample(source, t)
        elif source_for_diffusion is not None and not self.gp_conditioned:
            noisy = self.ddpm(source_for_diffusion, t, noise)
        elif self.gp_forward and not self.gp_conditioned and mask is not None:
            gp_batch = self._compute_gp_batch(x0, mask)
            noisy = self.ddpm(gp_batch.to(self.device), t, noise)
        else:
            noisy = self.ddpm(x0, t, noise)

        # Self-conditioning: two-pass with probability p_self_cond
        sc = None
        if self.self_conditioning and random.random() < self.p_self_cond:
            with torch.no_grad():
                first_pass = self._predict_denoised(noisy, t, mask=mask, known=known, self_cond=None)
                sc = first_pass.detach()

        pred = self._predict_denoised(noisy, t, mask=mask, known=known, self_cond=sc)
        loss = self._prediction_loss(pred, x0, noise, noisy, t, mask=mask, epoch=epoch)
        return loss, n

    def _load_initial_weights(self):
        if not self.init_from_weights:
            return

        path = Path(self.init_from_weights)
        if not path.is_absolute():
            path = BASE_DIR / path
        if not path.exists():
            raise FileNotFoundError(f"init_from_weights not found: {path}")

        state = torch.load(path, map_location=self.device, weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            state_dict = state["model_state_dict"]
        else:
            state_dict = state

        strict = not (self.rollout_stage_aware or self.self_conditioning)
        incompatible = self.ddpm.load_state_dict(state_dict, strict=strict)
        if strict:
            logging.info(f"Loaded initial weights from {path}")
            return

        missing = list(getattr(incompatible, "missing_keys", []))
        unexpected = list(getattr(incompatible, "unexpected_keys", []))
        logging.info(f"Loaded initial weights from {path} with strict=False")
        if missing:
            logging.info(f"  Missing keys (expected for new stage params): {missing}")
        if unexpected:
            logging.info(f"  Unexpected keys ignored: {unexpected}")

    def _bootstrap_rollout_source(self, source, t, mask=None, known=None, depth=0):
        if depth <= 0:
            return source

        was_training = self.ddpm.training
        rollout = source.detach()

        try:
            self.ddpm.eval()
            with torch.no_grad():
                for _ in range(depth):
                    rollout_noise = torch.randn_like(rollout)
                    rollout_noisy = self.ddpm(rollout, t, rollout_noise)
                    rollout_pred = self._predict_denoised(rollout_noisy, t, mask=mask, known=known)

                    if mask is not None and known is not None:
                        rollout = rollout_pred * mask + known
                    else:
                        rollout = rollout_pred
        finally:
            if was_training:
                self.ddpm.train()

        return rollout.detach()

    def evaluate(self, loader, fixed_seed=None):
        """Evaluate average MSE loss on a loader.

        If ``self.eval_max_batches > 0``, only evaluate the first N batches
        for speed (useful with large datasets).
        """
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
        max_batches = self.eval_max_batches if self.eval_max_batches > 0 else float("inf")

        try:
            with torch.no_grad():
                for batch_idx, batch in enumerate(loader):
                    if batch_idx >= max_batches:
                        break
                    loss, n = self._compute_batch_loss(batch, epoch=0)
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
        logging.info(f"GP-forward training: {self.gp_forward}")
        logging.info(f"GP-conditioned training: {self.gp_conditioned}")
        logging.info(f"Voronoi-forward training: {self.voronoi_forward}")
        logging.info(f"Helmholtz-split noise: {self.helmholtz_split_noise}")
        if self.helmholtz_split_noise:
            logging.info(f"  Irr speed: {self.irr_speed}")
            logging.info(f"  α_bar_sol[T-1]: {self.split_schedule.alpha_bars_sol[-1]:.4f}")
            logging.info(f"  α_bar_irr[T-1]: {self.split_schedule.alpha_bars_irr[-1]:.4f}")
        logging.info(f"Bootstrap rollout training: {self.bootstrap_rollout_training}")
        if self.bootstrap_rollout_training:
            logging.info(f"  Rollout depth probs: {self.bootstrap_rollout_depth_probs}")
        logging.info(f"Detached rollout training: {self.detached_rollout_training}")
        if self.detached_rollout_training:
            logging.info(f"  Rollout stages: {self.rollout_num_stages}")
            logging.info(f"  Stage-aware denoiser: {self.rollout_stage_aware}")
            logging.info(f"  Stage loss weights: {self.rollout_stage_loss_weights}")
            logging.info(f"  Stage t max fracs: {self.rollout_stage_t_max_fractions}")
            logging.info(f"  Repaste known each stage: {self.rollout_repaste_known}")
        if self.init_from_weights:
            logging.info(f"Init from weights: {self.init_from_weights}")
        if self.gp_forward or self.gp_conditioned:
            logging.info(f"  GP cache: {self.gp_cache_path}")
            if hasattr(self, 'gp_lengthscale'):
                logging.info(f"  GP params: ls={self.gp_lengthscale}, var={self.gp_variance}, "
                             f"noise={self.gp_noise}, kernel={self.gp_kernel_type}")
            if self.unet_type == "gp_context":
                logging.info(f"  GP-context mode (8ch concat conditioning)")
        logging.info(f"Prediction target: {self.prediction_target}")
        logging.info(f"LR schedule: {self.lr_schedule}, warmup: {self.warmup_epochs} epochs")
        logging.info(f"Weight decay: {self.weight_decay}")
        logging.info(f"EMA: {self.use_ema} (decay={self.ema_decay}, warmup_steps={self.ema_warmup_steps})")
        logging.info(f"Augmentation: {self.augment}")
        logging.info(f"Output dir: {self.output_dir}")
        if self.unet_type == "spatiotemporal":
            logging.info(f"Spatiotemporal: T={self.T}, freeze_spatial_epochs={self.freeze_spatial_epochs}")
            logging.info(f"Spatial LR factor: {self.spatial_lr_factor} (spatial_lr={self.lr * self.spatial_lr_factor:.6f} at unfreeze)")
            logging.info(f"Temporal dropout: {self.temporal_dropout}")
            if self.freeze_spatial_epochs >= self.n_epochs:
                logging.info("Spatial weights frozen for ALL epochs (freeze-forever mode)")
            if self.pretrained_spatial:
                logging.info(f"Pretrained spatial checkpoint: {self.pretrained_spatial}")

        # Optimizer: AdamW if weight_decay > 0, else plain Adam
        if self.init_from_weights and not (self.retrain_mode and self.model_to_retrain):
            self._load_initial_weights()

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
                # (skip if freeze_spatial_epochs >= total epochs → freeze-forever mode)
                if (spatial_frozen
                        and self.freeze_spatial_epochs < self.n_epochs
                        and epoch >= start_epoch + self.freeze_spatial_epochs):
                    self.ddpm.network.unfreeze_spatial()
                    spatial_frozen = False

                    # Rebuild optimizer with differential learning rates:
                    # spatial params get spatial_lr_factor * lr, temporal keep lr
                    spatial_lr = self.lr * self.spatial_lr_factor
                    temporal_lr = self.lr
                    param_groups = self.ddpm.network.param_groups(spatial_lr, temporal_lr)
                    if self.weight_decay > 0:
                        optimizer = AdamW(param_groups, weight_decay=self.weight_decay)
                    else:
                        optimizer = Adam(param_groups)

                    # Reset cosine LR schedule for the remaining epochs
                    remaining_epochs = self.n_epochs - self.freeze_spatial_epochs
                    phase2_warmup = min(5, remaining_epochs // 10)  # brief warmup
                    if self.lr_schedule == "cosine" and remaining_epochs > phase2_warmup:
                        warmup_sched = LinearLR(
                            optimizer,
                            start_factor=0.1,
                            end_factor=1.0,
                            total_iters=phase2_warmup,
                        )
                        cosine_sched = CosineAnnealingLR(
                            optimizer,
                            T_max=remaining_epochs - phase2_warmup,
                            eta_min=spatial_lr * 0.01,  # min LR based on smaller spatial LR
                        )
                        scheduler = SequentialLR(
                            optimizer,
                            schedulers=[warmup_sched, cosine_sched],
                            milestones=[phase2_warmup],
                        )
                    elif self.lr_schedule == "cosine":
                        scheduler = CosineAnnealingLR(
                            optimizer, T_max=max(1, remaining_epochs),
                            eta_min=spatial_lr * 0.01,
                        )

                    logging.info(
                        f"Phase 2: unfroze spatial weights at epoch {epoch + 1} | "
                        f"spatial_lr={spatial_lr:.6f}, temporal_lr={temporal_lr:.6f} | "
                        f"new cosine schedule for {remaining_epochs} remaining epochs"
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
                    loss, n = self._compute_batch_loss(batch, epoch=epoch)

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
                if len(optimizer.param_groups) > 1:
                    lr_parts = [f"{g.get('name', f'g{i}')}={g['lr']:.6f}"
                                for i, g in enumerate(optimizer.param_groups)]
                    current_lr_str = ", ".join(lr_parts)
                else:
                    current_lr_str = f"{optimizer.param_groups[0]['lr']:.6f}"

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
                    f"{ema_str}, lr={current_lr_str}"
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
