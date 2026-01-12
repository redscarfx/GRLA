import torch
import torch.nn as nn
from twsa import TWSABlock
from edsr import ResidualBlock

class GRLABlock(nn.Module):
    def __init__(
        self,
        dim,
        window_size=8,
        num_heads=4,
    ):
        super().__init__()

        # 1. Local convolution (EDSR-style)
        self.conv = ResidualBlock(dim)

        # 2. Local window attention
        self.twsa = TWSABlock(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
        )

        # 3. Global attention: TODO
        self.tla = nn.Identity() 

        self.ffn = ConvFFN(dim) 

    def forward(self, x):
        shortcut = x
        x = self.conv(x)
        x = self.twsa(x)
        x = self.tla(x)
        x = self.ffn(x)
        return x + shortcut
    
class ConvFFN(nn.Module):
    def __init__(self, dim, expansion=4):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, dim * expansion, 1),
            nn.GELU(),
            nn.Conv2d(dim * expansion, dim, 1),
        )

    def forward(self, x):
        return x + self.ffn(x)

