import torch.nn as nn
import torch.nn.functional as F


class BicubicBaseline(nn.Module):
    '''
    This is used to check if our learned model will be better 
    than just a simple bicubic interpolation from low-res to high-res using PyTorch's function
    Used as a reference/baseline later on
    '''
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return F.interpolate(
            x,
            scale_factor=self.scale,
            mode="bicubic",
            align_corners=False,
        )
