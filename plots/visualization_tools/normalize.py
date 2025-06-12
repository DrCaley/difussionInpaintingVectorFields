import torch

# Should be used to depict directions of vectors in vector field,
# never to be used for manipulating data to work with

class normalize:
    def __init__(self, eps=1e-8):
        self.eps = eps

    def __call__(self, tensor):
        mag = torch.sqrt(tensor[0:1]**2 + tensor[1:2]**2 + self.eps)
        u = tensor[0:1] / mag
        v = tensor[1:2] / mag
        return torch.cat((u, v), dim=0)