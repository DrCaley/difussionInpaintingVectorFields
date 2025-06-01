import torch
import numpy as np
# Load the .pt file
my_mask = "maskNormed.pt"
my_input = "myInputNormed.pt"
my_prediction = "img1_random_path_thin_resample5.pt"
my_mask = torch.load(my_mask)
my_input = torch.load(my_input)
my_prediction = torch.load(my_prediction)
my_mask = my_mask.numpy()
my_input = my_input.numpy()
my_prediction = my_prediction.numpy()
print(my_input)