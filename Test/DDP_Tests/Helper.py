import unittest
import torch

from DDPM.Helper_Functions.standardize_data import standardize_data


class TestHelperFunctions(unittest.TestCase):

    def test_inpaint_generate_new_images(self):
        self.assertTrue(False)

    def test_inpaint_generate_new_images_denoise(self):
        self.assertTrue(False)

    def test_inpaint_generate_new_images_noise(self):
        self.assertTrue(False)

    def test_calculate_mse(self):
        self.assertTrue(False)

    def test_avg_pixel_value(self):
        self.assertTrue(False)

    def test_generate_random_mask(self):
        self.assertTrue(False)

    def test_generate_straight_line_mask(self):
        self.assertTrue(False)

    def test_generate_random_path_mask(self):
        self.assertTrue(False)

    def test_generate_squiggly_line_mask(self):
        self.assertTrue(False)

    def test_generate_robot_path_mask(self):
        self.assertTrue(False)

    def test_create_border_mask(self):
        self.assertTrue(False)

    def test_resize(self):
        self.assertTrue(False)

    def test_resize_transform(self):
        self.assertTrue(False)

    def test_standardize(self):
        shape = (2, 2, 2)
        self.assertTrue(False)

    def test_unstandardize(self):
        shape = (2, 64, 128)
        tensor = torch.randn(shape)
        standardized_tensor = standardize_data(tensor)
        self.assertTrue(False)
