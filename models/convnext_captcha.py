#!/usr/bin/env python3
"""ConvNeXt-style CAPTCHA recognizer with a CTC head."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight[:, None, None] + self.bias[:, None, None]


class ConvNeXtBlock(nn.Module):
    def __init__(self, channels: int, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels)
        self.norm = LayerNorm2d(channels)
        self.pointwise1 = nn.Conv2d(channels, channels * expansion, kernel_size=1)
        self.pointwise2 = nn.Conv2d(channels * expansion, channels, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout)
        self.gamma = nn.Parameter(torch.full((channels,), 1e-6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.depthwise(x)
        x = self.norm(x)
        x = F.gelu(self.pointwise1(x))
        x = self.dropout(self.pointwise2(x))
        return residual + self.gamma[:, None, None] * x


class ConvNeXtCaptchaCTC(nn.Module):
    """Modern CNN backbone that preserves width for CTC decoding."""

    decoder_type = "ctc"

    def __init__(
        self,
        num_classes: int,
        input_channels: int = 1,
        channels: tuple[int, int, int] = (96, 192, 384),
        depths: tuple[int, int, int] = (2, 2, 6),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.channels = channels
        self.depths = depths
        self.dropout = dropout

        stages = [
            nn.Sequential(
                nn.Conv2d(input_channels, channels[0], kernel_size=4, stride=(4, 4)),
                LayerNorm2d(channels[0]),
                *[ConvNeXtBlock(channels[0], dropout=dropout) for _ in range(depths[0])],
            )
        ]
        for index in range(1, len(channels)):
            stages.append(
                nn.Sequential(
                    LayerNorm2d(channels[index - 1]),
                    nn.Conv2d(channels[index - 1], channels[index], kernel_size=2, stride=(2, 1)),
                    *[ConvNeXtBlock(channels[index], dropout=dropout) for _ in range(depths[index])],
                )
            )
        self.backbone = nn.Sequential(*stages)
        self.norm = nn.LayerNorm(channels[-1])
        self.classifier = nn.Linear(channels[-1], num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        sequence = features.mean(dim=2).permute(0, 2, 1)
        return self.classifier(self.norm(sequence))

    def get_config(self) -> Dict[str, object]:
        return {
            "architecture": "ConvNeXtCaptchaCTC",
            "num_classes": self.num_classes,
            "input_channels": self.input_channels,
            "channels": list(self.channels),
            "depths": list(self.depths),
            "dropout": self.dropout,
            "decoder_type": self.decoder_type,
        }


if __name__ == "__main__":
    model = ConvNeXtCaptchaCTC(num_classes=34, channels=(32, 64, 128), depths=(1, 1, 1))
    print(tuple(model(torch.randn(2, 1, 80, 200)).shape))

