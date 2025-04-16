import torch
import numpy as np


#input should be a tensor with the same number of dimensions as the desired end shape
#or a greater number of dimensions where the first (input_dim - end_dim) dimensions will be ignored
def resize(tensors, end_shape):
    start_shape = list(tensors.shape)
    dimensions = len(start_shape)

    if dimensions > len(end_shape):
        resized_tensors = [resize(tensors[i], end_shape) for i in range(start_shape[0])]
        return torch.from_numpy(np.array(resized_tensors))

    if dimensions < len(end_shape):
        raise ValueError(f"Expected tensor with {len(end_shape)} dimensions,"
                         f" but got {len(tensors.shape)} dimensions for Tensor with shape {start_shape}.")
    for i in range(dimensions):
        if start_shape[i] == end_shape[i]:
            pass

        elif start_shape[i] > end_shape[i]:
            diff = start_shape[i] - end_shape[i]
            slices = [slice(0, start_shape[j] - diff) if j == i else slice(None) for j in range(dimensions)]
            tensors = tensors[tuple(slices)]

        elif end_shape[i] > start_shape[i]:
            diff = end_shape[i] - start_shape[i]
            zeros_shape = list(tensors.shape)
            zeros_shape[i] = diff
            zero_tensor = torch.zeros(tuple(zeros_shape))
            tensors = torch.concat((tensors, zero_tensor), dim=i)

    return tensors

#generate_png(resize(torch.load("../data/tensors/0.pt"), (2, 64, 128)), )

class ResizeTransform:
    def __init__(self, end_shape):
        self.end_shape = end_shape

    def __call__(self, tensor):
        return resize(tensor, self.end_shape).float()


