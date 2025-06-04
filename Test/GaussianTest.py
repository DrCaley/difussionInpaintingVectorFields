import unittest
import torch
from gaussian_process.incompressible_gp.adding_noise.divergence_free_noise import divergence_free_noise
from gaussian_process.incompressible_gp.adding_noise.compute_divergence import compute_divergence

from sympy.vector import divergence


class GaussianTest(unittest.TestCase):
    def test_test(self):
        self.assertTrue(False)

    def test_compute_divergence_vertical_direction(self):

        tensor = torch.tensor(
            [[
            [[0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0.],],
            [[1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1.],
             ],]])


        divergence_tensor = compute_divergence(tensor[0][0], tensor[0][1])
        divergence = divergence_tensor.abs().mean().item()

        self.assertAlmostEqual(0, divergence, delta=0.01)

    def test_compute_divergence_uniform_direction(self):

        tensor = torch.tensor(
            [[
            [[1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1.],],
            [[1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1.],
             ],]])


        divergence_tensor = compute_divergence(tensor[0][0], tensor[0][1])
        divergence = divergence_tensor.abs().mean().item()

        self.assertAlmostEqual(0, divergence, delta=0.01)
