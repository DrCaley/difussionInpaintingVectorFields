import os
import math
import pickle

import torch
import random
import numpy as np
import yaml
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose

from data_prep.ocean_image_dataset import OceanImageDataset
from ddpm.helper_functions.resize_tensor import resize_transform
from ddpm.helper_functions.standardize_data import standardize_data


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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("we are running on the:", self.device)
        self._setup_transforms()
        self._set_random_seed()
        self._setup_datasets(os.path.join(prefix, boundaries_path))

    def _setup_yaml_file(self, config_path) -> None:
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def _setup_tensors(self, pickle_path) -> None:
        with open(pickle_path, 'rb') as f:
            training_data_np, validation_data_np, test_data_np = pickle.load(f)

        self.training_tensor = torch.from_numpy(training_data_np).float()
        self.validation_tensor = torch.from_numpy(validation_data_np).float()
        self.test_tensor = torch.from_numpy(test_data_np).float()

    def _setup_datasets(self, boundaries_file):
        self.training_data = OceanImageDataset(
            data_tensor=self.training_tensor,
            boundaries=boundaries_file,
            transform=self.transform
            )
        self.test_data = OceanImageDataset(
            data_tensor=self.test_tensor,
            boundaries=boundaries_file,
            transform=self.transform
            )
        self.validation_data = OceanImageDataset(
            data_tensor=self.validation_tensor,
            boundaries=boundaries_file,
            transform=self.transform
            )

    def get_tensors(self):
        return training_tensor, test_tensor, validation_tensor

    def _set_random_seed(self):
        seed = self.config.get('testSeed')
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _setup_transforms(self):
        self.standardizer = standardize_data(
            self.config['u_training_mean'], self.config['u_training_std'],
            self.config['v_training_mean'], self.config['v_training_std']
        )

        self.transform = Compose([
            resize_transform((2, 64, 128)),
            self.standardizer
        ])

    def get_attribute(self, attr):
        return self.config.get(attr)

    def get_device(self):
        return self.device

    def get_standardizer(self):
        return self.standardizer

    def get_transform(self):
        return self.transform

    def get_standarizer(self):
        return self.standardizer

    def get_training_data(self):
        return self.training_data

    def get_test_data(self):
        return self.test_data

    def get_validation_data(self):
        return self.validation_data

    def get_using_pycharm(self):
        return self.using_pycharm