import torch

"""To easily view results as numpy arrays in debugger. Set checkpoint on "if True" and inspect array"""

path = "results/predicted/img1_random_path_thick_resample5.pt"
vectors = torch.load(path).numpy()
if True:
    print("Checkpoint")