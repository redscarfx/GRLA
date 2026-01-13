import torch
import torch.nn as nn
import torch.nn.functional as F

class GRBFLA(nn.Module):
    """
    Gaussian RBF-based Linear Attention (TLA)
    Global attention with linear complexity

    Idea of this block: we want to replace the costly softmax(Q K^T) operation
    with a feature map phi(.) such that:
    softmax(Q K^T) V  ~=  phi(Q) (phi(K)^T V)

    this gives us a linear complexity (we still need to compute phi(K)^T V, but this is done only once))
    """

    def __init__(self, dim, num_heads=4, gamma=0.5, eps=1e-6):
        super().__init__()

        assert dim % num_heads == 0

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.gamma = gamma
        self.eps = eps

        self.conv_path = nn.Sequential(
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, dim, 1),
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),  # depthwise
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, dim, 1),
        )

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def _phi(self, x):
        """
        GRBF feature map
        x: (B, heads, N, head_dim)
        """
        return torch.exp(-self.gamma * x.pow(2)) * torch.exp(2 * self.gamma * x)

    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        B, C, H, W = x.shape
        N = H * W

        # conv path
        conv_res = x
        x = self.conv_path(x)
        x = x + conv_res

        # GRBF Linear Attention
        att_res = x

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        # reshape to (B, heads, N, head_dim)
        q = q.view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)
        k = k.view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)
        v = v.view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)

        # GRBF feature mapping
        q_phi = self._phi(q)
        k_phi = self._phi(k)

        # Compute KV summary (linear attention core)
        kv = torch.einsum("bhnd,bhne->bhde", k_phi, v)

        # Normalization term
        z = 1.0 / (
            torch.einsum("bhnd,bhd->bhn", q_phi, k_phi.sum(dim=2))
            + self.eps
        )

        # Attention output
        out = torch.einsum("bhnd,bhde,bhn->bhne", q_phi, kv, z)

        # reshape back
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)

        out = self.proj(out)
        x = att_res + out

        # FFN
        ffn_res = x
        x_flat = x.permute(0, 2, 3, 1)
        x_flat = self.norm2(x_flat)
        x_flat = self.mlp(x_flat)
        x = ffn_res + x_flat.permute(0, 3, 1, 2)

        return x