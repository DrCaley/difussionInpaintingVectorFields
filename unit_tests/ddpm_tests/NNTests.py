import unittest

import torch


class NNTests(unittest.TestCase):
    def test_ddpm(self):
        nn = torch.load('../../ddpm/trained_models/weekend_ddpm_ocean_model.pt', map_location=torch.device('cpu'))

        x = 5 + 1
        print('hello')
