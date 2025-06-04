import torch
class standardize_data:
    def __init__(self, u_mean, u_std, v_mean, v_std):
        self.v_mean = v_mean
        self.v_std = v_std
        self.u_mean = u_mean
        self.u_std = u_std

    def __call__(self, tensor):
        u = (tensor[0:1]-self.u_mean) / self.u_std
        v = (tensor[1:2]-self.v_mean) / self.v_std
        return torch.cat((u, v), dim=0)

    def unstandardize(self, tensor):
        u = tensor[0:1] * self.u_std + self.u_mean
        v = tensor[1:2] * self.v_std + self.v_mean
        return torch.cat((u, v), dim=0)

