import torch
import torch.nn as nn
import torch.nn.functional as F

class TWSABlock(nn.Module):
    '''
    Implementation of the Transformed Window Self-Attention module
    Uses MHA on windows of extracted patches from the feature map
    '''
    def __init__(self, dim, window_size=8, num_heads=4):
        super().__init__()
        self.window_size = window_size

        # the extra convolution before the MHSA + batch norms
        self.conv_path = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),  # depthwise
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, dim, 1),
        )

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        ws = self.window_size

        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        Hp, Wp = x.shape[2], x.shape[3] 

        # convolutional path before the MHSA
        conv_res = x
        x = self.conv_path(x)
        x = x + conv_res

        # window attention
        attn_res = x
        x = window_partition(x, ws)    
        x = self.norm1(x)
        x = self.attn(x)
        x = window_reverse(x, ws, Hp, Wp)  

        x = x + attn_res

        # FFN
        ffn_res = x
        x_flat = x.permute(0, 2, 3, 1)
        x_flat = self.norm2(x_flat)
        x_flat = self.mlp(x_flat)
        x = ffn_res + x_flat.permute(0, 3, 1, 2)

        # remove padding 
        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :H, :W]

        return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        """
        x: (num windows  *B, window size * window size, C)
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(
            B_, N, 3, self.num_heads, self.head_dim
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x

def window_partition(x, window_size):
    """
    Args:
        x: (B, C, H, W)
    Returns:
        windows: (num windows  *B, window size * window size, C)
    """
    B, C, H, W = x.shape
    x = x.view(
        B,
        C,
        H // window_size,
        window_size,
        W // window_size,
        window_size,
    )
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
    windows = x.view(-1, window_size * window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num windows * B, window size * window size, C)
    Returns:
        x: (B, C, H, W)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(
        B,
        H // window_size,
        W // window_size,
        window_size,
        window_size,
        -1,
    )
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
    x = x.view(B, -1, H, W)
    return x
