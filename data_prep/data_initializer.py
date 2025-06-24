import inspect
import os
import sys
import pickle
import torch
import random
import numpy as np
import yaml
from torchvision.transforms import Compose

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './../')))
from data_prep.ocean_image_dataset import OceanImageDataset
from ddpm.utils.noise_utils import NoiseStrategy, get_noise_strategy
from ddpm.helper_functions.loss_functions import LossStrategy, get_loss_strategy
from ddpm.helper_functions.resize_tensor import resize_transform
from ddpm.helper_functions.standardize_data import STANDARDIZER_REGISTRY, Standardizer


class DDInitializer:
    _instance = None

    def __new__(cls,
                config_path='data.yaml',
                pickle_path='data.pickle',
                boundaries_path='data/rams_head/boundaries.yaml'):
        if cls._instance is None:
            cls._instance = super(DDInitializer, cls).__new__(cls)
            cls._instance._init(config_path, pickle_path, boundaries_path)
        return cls._instance

    def _init(self, config_path, pickle_path, boundaries_path):
        self.using_pycharm = os.path.exists('../../data.yaml')
        prefix = "../../" if self.using_pycharm else "./"
        self.full_boundaries_path = os.path.join(prefix, boundaries_path)

        self._instance._setup_yaml_file(os.path.join(prefix, config_path))
        self._instance._setup_tensors(os.path.join(prefix, pickle_path))
        self.gpu = self._config.get('gpu_to_use')
        self.device = torch.device(f"cuda:{self.gpu}" if torch.cuda.is_available() else "cpu")
        print("we are running on the:", self.device)
        self._setup_transforms()
        self._set_random_seed()
        self._setup_noise_strategy()
        self._setup_loss_strategy()
        self._setup_datasets(self.full_boundaries_path)
        self._setup_alphas()

    def reinitialize(self, min_beta, max_beta, n_steps, standardizer : Standardizer):
        self._setup_alphas(min_beta, max_beta, n_steps)
        self._setup_transforms(standardizer)
        self._setup_datasets(self.full_boundaries_path)

    def _setup_yaml_file(self, config_path) -> None:
        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)

    def _setup_alphas(self, min_beta=None, max_beta=None, n_steps=None):
        min_beta = min_beta if min_beta is not None else self._config["min_beta"]
        max_beta = max_beta if max_beta is not None else self._config["max_beta"]
        n_steps = n_steps if n_steps is not None else self._config["noise_steps"]

        self.betas = torch.linspace(min_beta, max_beta, n_steps)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.tensor([torch.prod(self.alphas[:i + 1]) for i in range(len(self.alphas))])

    def _setup_tensors(self, pickle_path) -> None:
        with open(pickle_path, 'rb') as f:
            training_data_np, validation_data_np, test_data_np = pickle.load(f)

        self.training_tensor = torch.from_numpy(training_data_np).float()
        self.validation_tensor = torch.from_numpy(validation_data_np).float()
        self.test_tensor = torch.from_numpy(test_data_np).float()

    def _setup_datasets(self, boundaries_file):
        self.training_data = OceanImageDataset(
            n_steps=self._config["n_steps"],
            noise_strategy=self.noise_strategy,
            data_tensor=self.training_tensor,
            boundaries=boundaries_file,
            transform=self.transform,
        )
        self.test_data = OceanImageDataset(
            data_tensor=self.test_tensor,
            n_steps=self._config["n_steps"],
            noise_strategy=self.noise_strategy,
            boundaries=boundaries_file,
            transform=self.transform,
        )
        self.validation_data = OceanImageDataset(
            data_tensor=self.validation_tensor,
            n_steps=self._config["n_steps"],
            noise_strategy=self.noise_strategy,
            boundaries=boundaries_file,
            transform=self.transform,
        )

    def _setup_noise_strategy(self):
        noise_type = self._config.get("noise_function", "gaussian")
        try:
            self.noise_strategy: NoiseStrategy = get_noise_strategy(noise_type)
            print(f"Loaded noise strategy: {noise_type}")
        except KeyError:
            raise ValueError(f"Unknown noise strategy: {noise_type}")

    def get_noise_strategy(self) -> NoiseStrategy:
        return self.noise_strategy

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
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _setup_transforms(self, standardizer: Standardizer = None):
        if standardizer is not None:
            self.standardizer = standardizer
        else:
            std_type = self._config.get('standardizer_type')
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

    def get_using_pycharm(self):
        return self.using_pycharm

    def get_alphas(self):
        return self.alphas

    def get_betas(self):
        return self.betas

    def get_alpha_bars(self):
        return self.alpha_bars
