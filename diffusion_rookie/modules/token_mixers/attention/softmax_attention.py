import torch.nn as nn
import xformers.ops as xops
from einops import rearrange


class SimpleAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b (x y) h c", h=self.heads), qkv
        )
        out = xops.memory_efficient_attention(q, k, v)
        out = rearrange(out, "b (x y) h d -> b (h d) x y", x=h, y=w)

        return self.to_out(out)
