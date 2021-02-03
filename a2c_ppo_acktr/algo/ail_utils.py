import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data 
import lmdb
import os.path as osp
import pyarrow as pa
import six
import numpy as np 
import PIL 
from matplotlib import pyplot as plt
import pickle 

## Concatnenate interface
def sa_cat( state, action ):
    if action is not None:
        return torch.cat([state, action], dim=1)
    else:
        return state 
        
class Unhinged_Loss(nn.Module):   
    def __init__(self, scale=1):
        super().__init__()
        self.scale = scale if scale != 0 else 1

    def forward(self, z, reduction=True):
        loss = self.scale - z
        return loss.mean() if reduction else loss   

    def reward(self, z, reduction=False):
        return self.forward(z, reduction)
    
class Logistic_Loss(nn.Module):   
    def __init__(self):
        super().__init__()

    def forward(self, z, reduction=True):
        loss = -F.logsigmoid(z) # logistic loss is log(1+exp(-z)). F.logsigmoid is -log(1+exp(-z))
        return loss.mean() if reduction else loss   

    def reward(self, z, reduction=False):
        return self.forward(z, reduction)
    
class Sigmoid_Loss(nn.Module):   
    def __init__(self):
        super().__init__()

    def forward(self, z, reduction=True):
        loss = torch.sigmoid(-z) 
        return loss.mean() if reduction else loss   

    def reward(self, z, reduction=False):
        return self.forward(z, reduction)
    
class Normalized_Logistic_Loss(nn.Module):   
    def __init__(self):
        super().__init__()

    def forward(self, z, reduction=True):
        loss = F.logsigmoid(z) / ( F.logsigmoid(z) + F.logsigmoid(-z) )
        return loss.mean() if reduction else loss   

    def reward(self, z, reduction=False):
        return self.forward(z, reduction)
        
## Active-Passive Loss. Linear combination of normalized logistic and sigmoid losses (i.e., NCE and RCE)
class APL_Loss(nn.Module):   
    def __init__(self, alpha=0.5, beta=0.5):
        super().__init__()
        self.alpha = alpha 
        self.beta = beta 

    def forward(self, z, reduction=True):
        normalized_logistic = F.logsigmoid(z) / ( F.logsigmoid(z) + F.logsigmoid(-z) )
        sigmoid = torch.sigmoid(-z) 
        loss = self.alpha * normalized_logistic + self.beta * sigmoid
        return loss.mean() if reduction else loss   

    def reward(self, z, reduction=False):
        return self.forward(z, reduction)
    
### There losses are in referenced papers but not used.
class Hinge_Loss(nn.Module):   
    def __init__(self):
        super().__init__()

    def forward(self, z, reduction=True):
        loss = torch.clamp( 1 - z, min=0) 
        return loss.mean() if reduction else loss   

    def reward(self, z, reduction=False):
        return self.forward(z, reduction)
    
class Normalized_Hinge_Loss(nn.Module):   
    def __init__(self):
        super().__init__()

    def forward(self, z, reduction=True):
        loss = torch.clamp( 1 - z, min=0) / (torch.clamp( 1 - z, min=0) + torch.clamp( 1 + z, min=0))
        return loss.mean() if reduction else loss   

    def reward(self, z, reduction=False):
        return self.forward(z, reduction)

class Ramp_Loss(nn.Module):   
    def __init__(self):
        super().__init__()

    def forward(self, z, reduction=True):
        loss = torch.clamp( 0.5 - 0.5 * z, min=0, max=1) 
        return loss.mean() if reduction else loss   

    def reward(self, z, reduction=False):
        return self.forward(z, reduction)
    
class Barrier_Hinge_Loss(nn.Module):   
    def __init__(self, b=200, r=50):
        super().__init__()
        self.b = b 
        self.r = r

    def forward(self, z, reduction=True):
        loss = torch.max(-self.b * (self.r+z) + self.r, torch.max(self.b * (z - self.r), self.r - z))
        return loss.mean() if reduction else loss   

    def reward(self, z, reduction=False):
        return self.forward(z, reduction)
    