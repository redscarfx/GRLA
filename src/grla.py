import torch
import torch.nn as nn
from twsa import TWSABlock
from edsr import ResidualBlock
from ffn import ConvFFN
from tla import GRBFLA

class GRLABlock(nn.Module):
    '''
    This block contains the TWSA (Transformed Window Self-Attention) module as well as the TLA (Transformed Layer Attention) module
    These two modules are defined in their own files
    Before each of those GRLA block there is a local convolutional layer (EDSR style residual block)
    After those two attention modules there is a feed-forward network (FFN) implemented with 1x1 convolutions
    '''
    def __init__(
        self,
        dim,
        window_size=8,
        num_heads=4,
    ):
        super().__init__()

        # Local conv before the attention modules
        self.conv = ResidualBlock(dim)

        # MHA applied to the windows
        self.twsa = TWSABlock(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
        )

        # Global Linear attention
        self.tla = GRBFLA(dim, num_heads)

        # Feed-Forward Network
        self.ffn = ConvFFN(dim) 

    def forward(self, x):
        shortcut = x
        x = self.conv(x)
        x = self.twsa(x)
        x = self.tla(x)
        x = self.ffn(x)
        return x + shortcut