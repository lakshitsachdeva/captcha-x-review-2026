#!/usr/bin/env python3
"""Clean CRNN baseline with CTC decoding for variable-length CAPTCHAs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Small ResNet-style block that preserves feature-map width when requested."""

    def __init__(self, in_channels: int, out_channels: int, stride: tuple[int, int] = (1, 1)):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != (1, 1) or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = F.gelu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.gelu(x + residual)


class ResNetStride4Backbone(nn.Module):
    """
    ResNet-like CAPTCHA backbone.

    For a 200px-wide input, the output sequence has width about 50, satisfying
    the roadmap requirement that feature-map width remain far above max text
    length for stable CTC alignment.
    """

    def __init__(self, input_channels: int = 1, base_channels: int = 64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, kernel_size=5, stride=(2, 2), padding=2, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),
            ResidualBlock(base_channels, base_channels),
            ResidualBlock(base_channels, base_channels * 2, stride=(2, 2)),
            ResidualBlock(base_channels * 2, base_channels * 4, stride=(2, 1)),
            ResidualBlock(base_channels * 4, base_channels * 4, stride=(2, 1)),
        )
        self.out_channels = base_channels * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x)


class CRNNClean(nn.Module):
    """End-to-end CRNN baseline: ResNet-style CNN, BiLSTM, CTC logits."""

    decoder_type = "ctc"

    def __init__(
        self,
        num_classes: int,
        input_channels: int = 1,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        base_channels: int = 64,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.base_channels = base_channels

        self.backbone = ResNetStride4Backbone(input_channels=input_channels, base_channels=base_channels)
        self.sequence_projection = nn.Linear(self.backbone.out_channels, hidden_size)
        self.rnn = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        pooled = features.mean(dim=2).permute(0, 2, 1)
        sequence = F.gelu(self.sequence_projection(pooled))
        encoded, _ = self.rnn(sequence)
        return self.classifier(encoded)

    def ctc_logits(self, images: torch.Tensor) -> torch.Tensor:
        """Return time-major logits for PyTorch CTCLoss."""
        return self(images).permute(1, 0, 2)

    def input_lengths(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            width = self.backbone(images[:1]).shape[-1]
        return torch.full((images.shape[0],), width, dtype=torch.long, device=images.device)

    def get_config(self) -> Dict[str, float | int | str]:
        return {
            "architecture": "CRNNClean",
            "num_classes": self.num_classes,
            "input_channels": self.input_channels,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "base_channels": self.base_channels,
            "decoder_type": self.decoder_type,
        }


def ctc_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_lengths: torch.Tensor,
    blank_idx: int = 0,
) -> torch.Tensor:
    """CTC loss for batch-major logits and padded targets."""
    log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)
    input_lengths = torch.full((logits.shape[0],), logits.shape[1], dtype=torch.long, device=logits.device)
    flat_targets = torch.cat([target[: int(length.item())] for target, length in zip(targets, target_lengths)])
    return F.ctc_loss(log_probs, flat_targets, input_lengths, target_lengths, blank=blank_idx, zero_infinity=True)


def greedy_decode(logits: torch.Tensor, idx_to_char: Dict[int, str], blank_idx: int = 0) -> List[str]:
    token_ids = logits.argmax(dim=-1).detach().cpu().tolist()
    decoded = []
    for row in token_ids:
        chars: List[str] = []
        previous = blank_idx
        for token in row:
            if token != blank_idx and token != previous:
                chars.append(idx_to_char.get(int(token), ""))
            previous = token
        decoded.append("".join(chars))
    return decoded


@dataclass
class BeamHypothesis:
    text: str
    score: float


def beam_search_decode(logits: torch.Tensor, idx_to_char: Dict[int, str], blank_idx: int = 0, beam_width: int = 10) -> List[str]:
    """Compact CTC prefix beam search suitable for evaluation scripts."""
    log_probs = F.log_softmax(logits, dim=-1).detach().cpu()
    outputs: List[str] = []
    for sequence in log_probs:
        beams = {"": 0.0}
        for step in sequence:
            next_beams: Dict[str, float] = {}
            top_scores, top_indices = torch.topk(step, k=min(beam_width, step.numel()))
            for prefix, prefix_score in beams.items():
                for token_score, token_id in zip(top_scores.tolist(), top_indices.tolist()):
                    char = "" if token_id == blank_idx else idx_to_char.get(int(token_id), "")
                    candidate = prefix if token_id == blank_idx or prefix.endswith(char) else prefix + char
                    score = prefix_score + float(token_score)
                    if candidate not in next_beams or score > next_beams[candidate]:
                        next_beams[candidate] = score
            beams = dict(sorted(next_beams.items(), key=lambda item: item[1], reverse=True)[:beam_width])
        outputs.append(max(beams.items(), key=lambda item: item[1])[0])
    return outputs


CaptchaCRNN = CRNNClean


if __name__ == "__main__":
    model = CRNNClean(num_classes=34)
    x = torch.randn(2, 1, 80, 200)
    y = model(x)
    print(tuple(y.shape))
