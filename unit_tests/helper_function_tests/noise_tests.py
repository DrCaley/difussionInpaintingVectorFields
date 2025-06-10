import unittest
import torch
from noising_process.incompressible_gp.adding_noise.divergence_free_noise import gaussian_each_step_divergence_free_noise
from ddpm.helper_functions.compute_divergence import compute_divergence
from plots.plot_data_tool import plot_vector_field

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

    def test_compute_divergence_vortex(self):

        vx = torch.tensor([[0.0, -1.0, 0.0],
                           [1.0, 0.0, -1.0],
                           [0.0, 1.0, 0.0]])

        vy = torch.tensor([[0.0, 1.0, 0.0],
                           [-1.0, 0.0, 1.0],
                           [0.0, -1.0, 0.0]])

        divergence_tensor = compute_divergence(vx, vy)
        divergence = divergence_tensor.abs().mean().item()

        self.assertAlmostEqual(0, divergence, delta=0.3)

    def test_compute_divergence_alternating_signs(self):

        vx = torch.tensor([[1, -1, 1],
                           [-1, 1, -1],
                           [1, -1, 1]], dtype=torch.float32)

        vy = torch.tensor([[1, 1, 1],
                           [-1, -1, -1],
                           [1, 1, 1]], dtype=torch.float32)

        divergence_tensor = compute_divergence(vx, vy)
        divergence = divergence_tensor.abs().mean().item()

        self.assertAlmostEqual(0, divergence, delta=0.0)

    def test_gaussian_each_step(self):
        tensor = torch.zeros((1, 2, 128, 128))
        t = torch.randint(0, 1000, (1000,))
        noise = gaussian_each_step_divergence_free_noise(tensor.shape, t)
        plot_vector_field(noise[0][0], noise[0][1],2,30, title=f"{t[0].item()}", file=f"{t[0].item()}.png")

        divergence_tensor = compute_divergence(noise[1][0], noise[1][1])
        divergence = divergence_tensor.abs().mean().item()

        self.assertAlmostEqual(5, divergence, delta=0.3)




