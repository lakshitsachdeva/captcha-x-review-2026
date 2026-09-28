#!/usr/bin/env python3
"""Vision Transformer CAPTCHA recognizer with a CTC head."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """Patchify CAPTCHA images while preserving left-to-right token order."""

    def __init__(self, input_channels: int, embed_dim: int, patch_size: int = 8):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv2d(input_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.projection(x)
        tokens = tokens.mean(dim=2).permute(0, 2, 1)
        return tokens


class ViTCaptchaCTC(nn.Module):
    """
    ViT-small-style CAPTCHA model.

    This implementation avoids a hard timm dependency for smoke tests while
    matching the roadmap intent: patch size 8, global self-attention, and CTC
    sequence logits over the CAPTCHA vocabulary.
    """

    decoder_type = "ctc"

    def __init__(
        self,
        num_classes: int,
        input_channels: int = 1,
        embed_dim: int = 384,
        depth: int = 8,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        patch_size: int = 8,
        max_width_tokens: int = 32,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.dropout = dropout
        self.patch_size = patch_size
        self.max_width_tokens = max_width_tokens

        self.patch_embedding = PatchEmbedding(input_channels, embed_dim, patch_size=patch_size)
        self.position_embedding = nn.Parameter(torch.zeros(1, max_width_tokens, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embedding(images)
        if tokens.shape[1] > self.position_embedding.shape[1]:
            raise ValueError(
                f"Input produced {tokens.shape[1]} width tokens, but max_width_tokens={self.max_width_tokens}"
            )
        tokens = tokens + self.position_embedding[:, : tokens.shape[1]]
        encoded = self.encoder(tokens)
        return self.classifier(self.norm(encoded))

    def get_config(self) -> Dict[str, float | int | str]:
        return {
            "architecture": "ViTCaptchaCTC",
            "num_classes": self.num_classes,
            "input_channels": self.input_channels,
            "embed_dim": self.embed_dim,
            "depth": self.depth,
            "num_heads": self.num_heads,
            "dropout": self.dropout,
            "patch_size": self.patch_size,
            "max_width_tokens": self.max_width_tokens,
            "decoder_type": self.decoder_type,
        }


if __name__ == "__main__":
    model = ViTCaptchaCTC(num_classes=34, embed_dim=128, depth=2, num_heads=4)
    print(tuple(model(torch.randn(2, 1, 80, 200)).shape))

