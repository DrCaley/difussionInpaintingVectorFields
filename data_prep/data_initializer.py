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
from ddpm.helper_functions.standardize_data import STANDARDIZER_REGISTRY  # Updated import


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

        self._instance._setup_yaml_file(os.path.join(prefix, config_path))
        self._instance._setup_tensors(os.path.join(prefix, pickle_path))
        self.gpu = self._config.get('gpu_to_use')
        self.device = torch.device(f"cuda:{self.gpu}" if torch.cuda.is_available() else "cpu")
        print("we are running on the:", self.device)
        self._setup_transforms()
        self._set_random_seed()
        self._setup_noise_strategy()
        self._setup_loss_strategy()
        self._setup_datasets(os.path.join(prefix, boundaries_path))
        self._setup_alphas()

    def _setup_yaml_file(self, config_path) -> None:
        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)

    def _setup_alphas(self):
        self.betas = torch.linspace(self._config["min_beta"], self._config["max_beta"], self._config["n_steps"])
        self.alphas = (1 - self.betas)
        self.alpha_bars = torch.tensor([torch.prod(self.alphas[:i + 1]) for i in range(len(self.alphas))])

    def _setup_tensors(self, pickle_path) -> None:
        with open(pickle_path, 'rb') as f:
            training_data_np, validation_data_np, test_data_np = pickle.load(f)

        self.training_tensor = torch.from_numpy(training_data_np).float()
        self.validation_tensor = torch.from_numpy(validation_data_np).float()
        self.test_tensor = torch.from_numpy(test_data_np).float()

    def _setup_datasets(self, boundaries_file):
        size = self._config.get('max_size')
        self.training_data = OceanImageDataset(
            n_steps=self._config["n_steps"],
            noise_strategy=self.noise_strategy,
            data_tensor=self.training_tensor,
            boundaries=boundaries_file,
            transform=self.transform,
            max_size=size
        )
        self.test_data = OceanImageDataset(
            data_tensor=self.test_tensor,
            n_steps=self._config["n_steps"],
            noise_strategy=self.noise_strategy,
            boundaries=boundaries_file,
            transform=self.transform,
            max_size=size
        )
        self.validation_data = OceanImageDataset(
            data_tensor=self.validation_tensor,
            n_steps=self._config["n_steps"],
            noise_strategy=self.noise_strategy,
            boundaries=boundaries_file,
            transform=self.transform,
            max_size=size
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

    def _setup_transforms(self):
        std_type = self._config.get('standardizer_type')

        if std_type == 'zscore':
            self.standardizer = STANDARDIZER_REGISTRY[std_type](
                self._config['u_training_mean'], self._config['u_training_std'],
                self._config['v_training_mean'], self._config['v_training_std']
            )
        elif std_type in STANDARDIZER_REGISTRY:
            self.standardizer = STANDARDIZER_REGISTRY[std_type]()
        else:
            raise ValueError(f"Unknown standardizer_type: {std_type}")

        self.transform = Compose([
            resize_transform((2, 64, 128)),
            self.standardizer
        ])

    def get_attribute(self, attr):
        try:
            return self._config.get(attr)
        except:
            print("no attribute", attr)
            return None

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
