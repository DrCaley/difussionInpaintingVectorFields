import torch
from tensors_to_png import generate_png


#input should be a 3xAnyXAny tensor and a triple of the desired dimensions for the last two
#outputs are normalized, so comparing the strength of currents between maps won't work
def resize(tensors, end_shape):
    actual_shape = tensors.shape
    if actual_shape[0] != 3:
        raise ValueError(f"Expected tensor with shape (3, Any, Any), but got {actual_shape}.")

    #1st dimension
    zeros_tensor = torch.zeros((actual_shape[0], end_shape[1] - actual_shape[1], actual_shape[2]) )
    resized_tensor = torch.cat((tensors, zeros_tensor), dim=1)
    #2nd dimension
    zeros_tensor = torch.zeros(actual_shape[0], end_shape[1], end_shape[2] - actual_shape[2])
    resized_tensor = torch.cat((resized_tensor, zeros_tensor), dim=2)

    return resized_tensor


#generate_png(resize(torch.load("./data/tensors/0.pt"), (3, 512, 512)), )
