import os
import sys
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from ddpm.helper_functions.HH_decomp import decompose_vector_field
from ddpm.helper_functions.compute_divergence import compute_divergence



H, W = 64, 64
field = torch.ones(H, W, 2)  # Constant (1, 1) everywhere
