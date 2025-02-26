import torch

path = "results/predicted/img1_random_path_thick_resample5.pt"
vectors = torch.load(path).numpy()
if True:
    print("Checkpoint")