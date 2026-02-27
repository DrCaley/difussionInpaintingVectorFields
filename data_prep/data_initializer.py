import inspect
import os
import sys
import pickle
import torch
import random
import numpy as np
import yaml
from torchvision.transforms import Compose
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from data_prep.ocean_image_dataset import OceanImageDataset
from ddpm.utils.noise_utils import NoiseStrategy, get_noise_strategy
from ddpm.helper_functions.loss_functions import LossStrategy, get_loss_strategy
from ddpm.helper_functions.resize_tensor import resize_transform
from ddpm.helper_functions.standardize_data import STANDARDIZER_REGISTRY, Standardizer
from ddpm.protocols import validate_noise_standardizer, ComponentIncompatibilityError

class PickleNotFoundException(Exception):
    """Raise for my specific kind of exception"""

class DDInitializer:
    _instance = None

    def __new__(cls,
                config_path='data.yaml',
                pickle_path='data.pickle',
                boundaries_path='data/rams_head/boundaries.yaml'):
        if cls._instance is None:
            cls._instance = super(DDInitializer, cls).__new__(cls)
            cls._instance._init(
                Path(config_path),
                Path(pickle_path),
                Path(boundaries_path)
                )
        return cls._instance

    def _init(self, config_path, pickle_path, boundaries_path):
        root = Path(__file__).resolve().parent.parent
        self.config_name = config_path.stem
        self.full_boundaries_path = root / boundaries_path
        self._instance._setup_yaml_file(root / config_path)
        self._instance._setup_tensors(root / pickle_path)

        self.gpu = self._config.get('gpu_to_use')
        # Check for GPU: CUDA (NVIDIA) > MPS (Apple Silicon) > CPU
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.gpu}")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print("we are running on the:", self.device)

        self._setup_transforms()
        self._set_random_seed()
        self._setup_noise_strategy()
        self._setup_vector_combination()
        self._setup_loss_strategy()
        self._resolve_noise_dependent_settings()
        self._setup_alphas()
        self._setup_datasets(self.full_boundaries_path)
        self._validate_component_compatibility()

    def reinitialize(self, min_beta, max_beta, n_steps, standardizer : Standardizer):
        self._setup_alphas(min_beta, max_beta, n_steps)
        self._setup_transforms(standardizer)
        self._resolve_noise_dependent_settings()
        self._setup_datasets(self.full_boundaries_path)
        self._validate_component_compatibility()

    def _setup_yaml_file(self, config_path : Path) -> None:
        if not config_path.exists():
            raise FileNotFoundError(f"{config_path} does not exist")

        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)

    def _setup_alphas(self, min_beta=None, max_beta=None, n_steps=None):
        self.min_beta = min_beta if min_beta is not None else self._config["min_beta"]
        self.max_beta = max_beta if max_beta is not None else self._config["max_beta"]
        self.n_steps = n_steps if n_steps is not None else self._config["noise_steps"]

        self.betas = torch.linspace(self.min_beta, self.max_beta, self.n_steps)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.tensor([torch.prod(self.alphas[:i + 1]) for i in range(len(self.alphas))])

    def _setup_tensors(self, pickle_path : Path) -> None:
        if not pickle_path.exists():
            raise PickleNotFoundException("Pickle file that contains the data was not found, "
                                          "make sure you created it with the slitting datasets python script")

        with open(pickle_path, 'rb') as f:
            training_data_np, validation_data_np, test_data_np = pickle.load(f)

        self.training_tensor = torch.from_numpy(training_data_np).float()
        self.validation_tensor = torch.from_numpy(validation_data_np).float()
        self.test_tensor = torch.from_numpy(test_data_np).float()

    def _setup_datasets(self, boundaries_file):
        self.training_data = OceanImageDataset(
            n_steps=self.n_steps,
            noise_strategy=self.noise_strategy,
            data_tensor=self.training_tensor,
            boundaries=boundaries_file,
            transform=self.transform,
        )
        self.test_data = OceanImageDataset(
            data_tensor=self.test_tensor,
            n_steps=self.n_steps,
            noise_strategy=self.noise_strategy,
            boundaries=boundaries_file,
            transform=self.transform,
        )
        self.validation_data = OceanImageDataset(
            data_tensor=self.validation_tensor,
            n_steps=self.n_steps,
            noise_strategy=self.noise_strategy,
            boundaries=boundaries_file,
            transform=self.transform,
        )

    def _resolve_noise_dependent_settings(self):
        """Resolve all 'auto' settings that depend on the active noise type.

        Called during init and reinitialize so that consumers (e.g.
        inpaint_generate_new_images) get the correct resolved values
        from get_attribute() without needing any noise-type awareness.
        """
        noise_type = self.get_noise_type()

        # --- divergence projection ---
        proj_raw = self._config.get("enable_divergence_projection", "auto")
        if isinstance(proj_raw, str) and proj_raw.lower() == "auto":
            mapping = self._config.get("projection_by_noise", {})
            self._resolved_projection = bool(mapping.get(noise_type, False))
        else:
            self._resolved_projection = bool(proj_raw)

    def get_noise_type(self) -> str:
        """Return the active noise function name."""
        return self._config.get("noise_function", "gaussian")

    def _setup_noise_strategy(self):
        noise_type = self._config.get("noise_function", "gaussian")
        try:
            self.noise_strategy: NoiseStrategy = get_noise_strategy(noise_type)
            print(f"Loaded noise strategy: {noise_type}")
        except KeyError:
            raise ValueError(f"Unknown noise strategy: {noise_type}")

    def get_noise_strategy(self) -> NoiseStrategy:
        return self.noise_strategy

    def _setup_vector_combination(self):
        comb_net_instruction = self._config.get("use_comb_net", "auto")
        match comb_net_instruction:
            case "auto":
                self.use_comb_net = self._config.get("noise_function") == "div_free"
            case "yes" | True:
                self.use_comb_net = True
            case "no" | False:
                self.use_comb_net = False
            case _:
                raise ValueError(f"Unknown combination net instruction: {comb_net_instruction}")

    def get_use_comb_net(self) -> bool:
        return self.use_comb_net

    def _setup_loss_strategy(self):
        loss_type = self._config.get("loss_function", "mse")
        w1 = self._config.get("w1", 1.0)
        w2 = self._config.get("w2", 0.0)
        try:
            self.loss_strategy: LossStrategy = get_loss_strategy(loss_type)
            print(f"Loaded loss strategy: {loss_type} (w1={w1}, w2={w2})")
        except KeyError:
            raise ValueError(f"Unknown loss strategy: {loss_type}")

    def get_loss_strategy(self) -> LossStrategy:
        return self.loss_strategy

    def get_tensors(self):
        return self.training_tensor, self.test_tensor, self.validation_tensor

    def _set_random_seed(self):
        seed = self._config.get('testSeed')
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        self.seed = seed

    def _setup_transforms(self, standardizer: Standardizer = None):
        if standardizer is not None:
            self.standardizer = standardizer
        else:
            std_type = self._config.get('standardizer_type')

            if std_type == "auto":
                noise_type = self._config.get("noise_function", "gaussian")
                mapping = self._config.get("standardizer_by_noise", {})
                std_type = mapping.get(noise_type, "zscore")

            std_class = STANDARDIZER_REGISTRY.get(std_type)

            if std_class is None:
                raise ValueError(f"Unknown standardizer_type: {std_type}")

            # Get required constructor parameters (excluding 'self')
            sig = inspect.signature(std_class.__init__)
            init_params = list(sig.parameters.keys())[1:]

            # Collect arguments from config
            args = [self._config[param] for param in init_params]

            # Instantiate standardizer from registry
            self.standardizer = std_class(*args)

        self.transform = Compose([
            resize_transform((2, 64, 128)),
            self.standardizer
        ])

    def get_attribute(self, attr):
        # Return noise-resolved values for auto-configured settings
        if attr == "enable_divergence_projection":
            return getattr(self, "_resolved_projection",
                           self._config.get("enable_divergence_projection"))
        try:
            return self._config.get(attr)
        except:
            raise Exception(f"Unknown attribute in data.yaml: {attr}")

    def get_device(self):
        return self.device

    def get_standardizer(self):
        return self.standardizer

    def get_transform(self):
        return self.transform

    def get_training_data(self):
        return self.training_data

    def get_test_data(self):
        return self.test_data

    def get_validation_data(self):
        return self.validation_data

    def get_alphas(self):
        return self.alphas

    def get_betas(self):
        return self.betas

    def get_alpha_bars(self):
        return self.alpha_bars

    def get_full_config(self):
        return self._config

    def get_config_name(self):
        return self.config_name

    @classmethod
    def reset_instance(cls):
        cls._instance = None

    # ------------------------------------------------------------------
    # Component compatibility validation
    # ------------------------------------------------------------------

    def _validate_component_compatibility(self):
        """Validate that all building-block pairings are consistent.

        Called automatically at the end of ``_init()`` and ``reinitialize()``.
        Raises ``ComponentIncompatibilityError`` on the first violation,
        or prints a success message if all checks pass.

        Validated rules
        ---------------
        1. Div-free noise strategies require a unified standardizer
           (same std for u,v) so that standardization preserves the
           zero-divergence property.  See ``protocols.py``.
        """
        try:
            validate_noise_standardizer(self.noise_strategy, self.standardizer)
            print("[DDInitializer] Component compatibility check: PASSED")
        except ComponentIncompatibilityError as e:
            print(f"[DDInitializer] WARNING — component incompatibility: {e}")
            raise