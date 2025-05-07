import torch
class StandardizeData:
    def __init__(self,u_mean,u_std,v_mean,v_std):
        self.v_mean = v_mean
        self.v_std = v_std
        self.u_mean = u_mean
        self.u_std = u_std

    def __call__(self, tensor):#check this math
        return torch.cat((((tensor[0:1]-self.u_mean)/self.u_std),((tensor[1:2]-self.v_mean)/self.v_std)),0)

def reverseStandardization(data,mean,std):
    return (data*std)+mean



