"""Small Vision Transformers for 32x32 inputs (Dosovitskiy et al. 2021 layout).

Patch size 4 -> 64 patch tokens plus a class token, pre-norm blocks, learned
position embeddings. Attention uses explicit ``qkv`` and ``proj`` Linear layers
(not nn.MultiheadAttention's fused parameters) so every projection can be
quantized like any other Linear. Written to be torch.fx-traceable (no Python
unpacking of tensor shapes).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.heads, self.head_dim = heads, dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv(x).reshape(x.shape[0], x.shape[1], 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, N, head_dim]
        out = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
        return self.proj(out.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1))


class MLP(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_dim: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class ViT(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int = 10,
        image_size: int = 32,
        patch: int = 4,
        dim: int = 192,
        depth: int = 6,
        heads: int = 3,
        mlp_dim: int = 384,
    ) -> None:
        super().__init__()
        n_patches = (image_size // patch) ** 2
        self.patch_embed = nn.Conv2d(3, dim, patch, stride=patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.blocks = nn.Sequential(*[Block(dim, heads, mlp_dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x).flatten(2).transpose(1, 2)  # [B, N, dim]
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        x = self.norm(self.blocks(x))
        return self.head(x[:, 0])


# Three sizes for the ViT track (parameter counts in tests/test_vit.py).
VIT_SIZES = {
    "vit_t_cifar": {"dim": 128, "depth": 6, "heads": 4, "mlp_dim": 256},
    "vit_s_cifar": {"dim": 256, "depth": 6, "heads": 8, "mlp_dim": 512},
    "vit_m_cifar": {"dim": 384, "depth": 8, "heads": 8, "mlp_dim": 768},
}


def vit_cifar(size: str, num_classes: int = 10) -> ViT:
    return ViT(num_classes=num_classes, **VIT_SIZES[size])
