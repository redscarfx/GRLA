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
        include_layer_norm=False,
        include_conv=True,
    ):
        super().__init__()

        # Local conv before the attention modules
        # to complex
        #self.res = ResidualBlock(dim)

        # MHA applied to the windows
        self.twsa = TWSABlock(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            include_layer_norm=include_layer_norm,
        )

        # Global Linear attention
        self.tla = GRBFLA(
            dim, 
            num_heads=1, # hard coded to 1 head for now (i think paper does this?)
            include_layer_norm=include_layer_norm
        ) 

        self.conv = nn.Conv2d(dim, dim, 1, padding=0) # the one by one conv before the TWSA and TLA blocks

        # Feed-Forward Network
        # self.ffn = ConvFFN(dim) # not 100% about the architecture but should be fine for now

    def forward(self, x):
        shortcut = x
        x = self.conv(x) # the 1x1 conv before the 2 attention blocks
        x = self.twsa(x)
        x = self.tla(x)
        x = x + shortcut # add residual connection
        return x