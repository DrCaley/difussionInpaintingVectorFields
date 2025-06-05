import unittest

from torchvision.transforms import Compose

from ddpm.Helper_Functions.resize_tensor import resize_transform
from ddpm.Helper_Functions.standardize_data import standardize_data
from ddpm.Testing.inpainting_model_test import config
from data_prep.ocean_image_dataset import OceanImageDataset


class TestDataPrep(unittest.TestCase):

    def setUp(self):
        transform = Compose([
            resize_transform((2, 64, 128)),
            standardize_data(config['u_training_mean'], config['u_training_std'], config['v_training_mean'],
                             config['v_training_std'])  # Resized to (2, 64, 128)
        ])
        self.data = OceanImageDataset(
            mat_file="../../data/rams_head/stjohn_hourly_5m_velocity_ramhead_v2.mat",
            boundaries="../../data/rams_head/boundaries.yaml",
            num=10,
            transform = transform
        )
        return

    def test_get_item(self):
        tensor = self.data.tensor_arr[0]


    def test_dataloader_load_array(self):
        self.assertTrue(False)

    def test_normalize(self):
        self.assertTrue(False)

    def test_split_data(self):
        self.assertTrue(False)

