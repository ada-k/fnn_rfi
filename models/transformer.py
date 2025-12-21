"""Lightweight transformer baseline for image classification.

We convert the image to non-overlapping patches, linearly embed them, add a
learnable CLS token and run a small TransformerEncoder stack.
"""
from typing import Tuple

import torch
import torch.nn as nn


class TinyVisionTransformer(nn.Module):
    def __init__(self, in_ch: int = 3, img_size: int = 32, patch_size: int = 4, emb_dim: int = 64, depth: int = 2, num_heads: int = 4, num_classes: int = 10):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.patch_size = patch_size
        num_patches = (img_size // patch_size) ** 2
        self.patch_dim = in_ch * patch_size * patch_size
        self.emb = nn.Linear(self.patch_dim, emb_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_emb = nn.Parameter(torch.zeros(1, 1 + num_patches, emb_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads, dim_feedforward=emb_dim * 2)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.mlp_head = nn.Linear(emb_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        b, c, h, w = x.shape
        p = self.patch_size
        # reshape into patches
        patches = x.unfold(2, p, p).unfold(3, p, p)  # B, C, n_h, n_w, p, p
        n_h = patches.size(2)
        n_w = patches.size(3)
        patches = patches.contiguous().view(b, c, n_h * n_w, p, p)
        patches = patches.permute(0, 2, 1, 3, 4).reshape(b, n_h * n_w, -1)
        tokens = self.emb(patches)  # B, N, E
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, tokens], dim=1)
        x = x + self.pos_emb
        # transformer expects (S, B, E) by default, so transpose
        x = x.permute(1, 0, 2)
        x = self.encoder(x)
        # take CLS token output
        cls_out = x[0]  # (B, E)
        logits = self.mlp_head(cls_out)
        return logits


if __name__ == "__main__":
    m = TinyVisionTransformer()
    x = torch.randn(2, 3, 32, 32)
    y = m(x)
    print(y.shape)
