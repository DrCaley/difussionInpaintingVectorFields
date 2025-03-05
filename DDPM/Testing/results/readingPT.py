import torch
import numpy as np
# Load the .pt file
myMask = "maskNormed.pt"
myInput = "myInputNormed.pt"
myPrediction = "img1_random_path_thin_resample5.pt"
myMask = torch.load(myMask)
myInput = torch.load(myInput)
myPrediction = torch.load(myPrediction)
myMask = myMask.numpy()
myInput = myInput.numpy()
myPrediction = myPrediction.numpy()
print(myInput)