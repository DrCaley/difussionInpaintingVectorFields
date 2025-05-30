import torch
class StandardizeData:
    def __init__(self,u_mean,u_std,v_mean,v_std):
        self.v_mean = v_mean
        self.v_std = v_std
        self.u_mean = u_mean
        self.u_std = u_std

    def __call__(self, tensor): #check this math
        # could modularize this, or leave as is :)
        return torch.cat((((tensor[0:1]-self.u_mean)/self.u_std),
                          ((tensor[1:2]-self.v_mean)/self.v_std)),0)

    def unstandardize(self, tensor):
        u = tensor[0:1] * self.u_std + self.u_mean
        v = tensor[1:2] * self.v_std + self.v_mean
        return torch.cat((u, v), dim=0)
    
"""
Matt: Here's what was originally under lines 9-12:

def reverseStandardization(data,mean,std):
    return (data*std)+mean

"""

