import torch.nn as nn

class ConvFFN(nn.Module):
    '''
    Basic feed forward netowrk using 1x1 convolutions and GELU for non linearity
    '''
    def __init__(self, dim, expansion=2):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, dim * expansion, 1),
            nn.GELU(),
            nn.Conv2d(dim * expansion, dim, 1),
        )

    def forward(self, x):
        return x + self.ffn(x)