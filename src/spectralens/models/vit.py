import torch
from torch import nn


class Block(nn.Module):
    """Pre-norm transformer block: x + attn(norm(x)), then x + mlp(norm(x))."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class TinyViT(nn.Module):
    """A small vision transformer (~1.8M params at the defaults) trained from scratch.

    Mostly here as a contrast: its weight and Hessian spectra are known to look different from a
    CNN's, which is one of the things the analysis phases look at.
    """

    def __init__(
        self,
        num_classes: int,
        image_size: int,
        patch: int,
        in_channels: int = 3,
        dim: int = 192,
        depth: int = 6,
        heads: int = 3,
    ):
        super().__init__()
        if image_size % patch:
            raise ValueError("image size must be a multiple of the patch size")
        n_tokens = (image_size // patch) ** 2
        # a strided conv is the same as cutting patches and applying one linear layer to each
        self.patch_embed = nn.Conv2d(in_channels, dim, kernel_size=patch, stride=patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, n_tokens + 1, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        self.blocks = nn.Sequential(*[Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)  # (B, tokens, dim)
        x = torch.cat([self.cls.expand(x.shape[0], -1, -1), x], dim=1) + self.pos
        x = self.blocks(x)
        return self.head(self.norm(x[:, 0]))
