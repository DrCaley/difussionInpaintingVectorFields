import unittest
import torch
import numpy as np
from mpmath.calculus.extrapolation import standardize

from ddpm.helper_functions.standardize_data import standardize_data


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
        tensor = torch.tensor([
            [[1.0, 2.0],
             [3.0, 4.0]],

            [[5.0, 6.0],
             [7.0, 8.0]]
        ])

        tensorU = tensor[0:1]
        tensorV = tensor[1:2]

        u_mean = np.nanmean(tensorU)
        u_std = np.nanstd(tensorU)
        v_mean = np.nanmean(tensorV)
        v_std = np.nanstd(tensorV)

        data_standardizer = standardize_data(u_mean, u_std, v_mean, v_std)
        standardized_tensor = data_standardizer(tensor)

        real_standardization = torch.tensor([
            [[-1.341640786, -0.4472135955],
             [0.4472135955, 1.341640786]],

            [[-1.341640786, -0.4472135955],
             [0.4472135955, 1.341640786]]
        ])

        self.assertTrue(torch.allclose(real_standardization, standardized_tensor, atol=1e-6))

    def test_unstandardize(self):
        tensor = torch.tensor([
            [[1.0, 2.0],
             [3.0, 4.0]],

            [[5.0, 6.0],
             [7.0, 8.0]]
        ])

        tensorU = tensor[0:1]
        tensorV = tensor[1:2]

        u_mean = np.nanmean(tensorU)
        u_std = np.nanstd(tensorU)
        v_mean = np.nanmean(tensorV)
        v_std = np.nanstd(tensorV)

        data_standardizer = standardize_data(u_mean, u_std, v_mean, v_std)
        standardized_tensor = data_standardizer(tensor)
        unstandardized_tensor = data_standardizer.unstandardize(standardized_tensor)

        self.assertTrue(torch.allclose(unstandardized_tensor, tensor, atol=1e-6))

