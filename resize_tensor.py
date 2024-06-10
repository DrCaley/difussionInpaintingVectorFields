import torch
from tensors_to_png import generate_png
#input should be a 3x44x94 tensor and a tuple of the desired demensions for the last two
#outputs are normalized, so comparing the strength of currents between maps won't work
def resize(tensors, end_shape):
    expected_shape = (3, 44, 94)
    actual_shape = tensors.shape
    if actual_shape != expected_shape:
        raise ValueError(f"Expected tensor with shape {expected_shape}, but got {actual_shape}.")

    #1st dimension
    zeros_tensor = torch.zeros((actual_shape[0], end_shape[1] - actual_shape[1], actual_shape[2]) )
    resized_tensor = torch.cat((tensors, zeros_tensor), dim=1)
    #2nd dimension
    zeros_tensor = torch.zeros(actual_shape[0], end_shape[1], end_shape[2] - actual_shape[2])
    resized_tensor = torch.cat((resized_tensor, zeros_tensor), dim=2)

    return resized_tensor


generate_png(resize(torch.load("./data/tensors/0.pt"), (3, 512, 512)), )
