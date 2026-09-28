#!/usr/bin/env python3
"""
High-accuracy fixed-position CAPTCHA recognizer and metric helpers.

For the synthetic benchmark in this repository, characters are arranged in a
roughly monotonic left-to-right layout. This model exploits that structure by
splitting the image into overlapping position windows and classifying each slot
with a shared character encoder plus position-specific heads.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Reusable convolutional block."""

    def __init__(self, in_channels: int, out_channels: int, pool: Optional[tuple[int, int]] = None):
        super().__init__()
        layers: List[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        ]
        if pool is not None:
            layers.append(nn.MaxPool2d(kernel_size=pool, stride=pool))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CaptchaCNN(nn.Module):
    """
    Fixed-position character recognizer.

    The name is preserved for compatibility with the rest of the repo.
    """

    decoder_type = "position"

    def __init__(
        self,
        num_classes: int,
        input_channels: int = 1,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_length: int = 6,
        num_heads: int = 4,
        decoder_type: str = "position",
        segment_width: int = 48,
        overlap_ratio: float = 0.25,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.input_channels = input_channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.max_length = max_length
        self.num_heads = num_heads
        self.decoder_type = decoder_type
        self.segment_width = segment_width
        self.overlap_ratio = overlap_ratio

        self.character_encoder = nn.Sequential(
            ConvBlock(input_channels, 32, pool=(2, 2)),
            ConvBlock(32, 64, pool=(2, 2)),
            ConvBlock(64, 128, pool=(2, 2)),
            ConvBlock(128, hidden_size, pool=None),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.feature_projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.position_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_size),
                    nn.Linear(hidden_size, hidden_size),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, num_classes),
                )
                for _ in range(max_length)
            ]
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def _slice_positions(self, x: torch.Tensor) -> torch.Tensor:
        """Split the CAPTCHA into overlapping windows."""
        _, _, height, width = x.shape
        base_width = width / self.max_length
        margin = max(2, int(round(base_width * self.overlap_ratio)))
        segments: List[torch.Tensor] = []

        for position in range(self.max_length):
            left = int(round(position * base_width))
            right = int(round((position + 1) * base_width))
            start = max(0, left - margin)
            end = min(width, right + margin)
            crop = x[:, :, :, start:end]
            resized = F.interpolate(
                crop,
                size=(height, self.segment_width),
                mode="bilinear",
                align_corners=False,
            )
            segments.append(resized)

        return torch.stack(segments, dim=1)

    def get_config(self) -> Dict[str, float | int | str]:
        return {
            "num_classes": self.num_classes,
            "input_channels": self.input_channels,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout_rate,
            "max_length": self.max_length,
            "num_heads": self.num_heads,
            "decoder_type": self.decoder_type,
            "segment_width": self.segment_width,
            "overlap_ratio": self.overlap_ratio,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        segments = self._slice_positions(x)
        batch_size, num_positions, channels, height, width = segments.shape
        flat_segments = segments.reshape(batch_size * num_positions, channels, height, width)

        encoded = self.character_encoder(flat_segments)
        features = self.feature_projection(encoded).reshape(batch_size, num_positions, self.hidden_size)

        logits = [
            head(features[:, position, :])
            for position, head in enumerate(self.position_heads)
        ]
        return torch.stack(logits, dim=1)


class PositionSequenceLoss(nn.Module):
    """Cross-entropy across all sequence positions."""

    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, max_length, num_classes = logits.shape
        return self.loss(logits.reshape(batch_size * max_length, num_classes), targets.reshape(batch_size * max_length))


class CTCLoss(PositionSequenceLoss):
    """Backward-compatible alias from the earlier CTC version."""

    def __init__(self, blank: int = 0, reduction: str = "mean"):
        super().__init__(label_smoothing=0.0)
        self.blank = blank
        self.reduction = reduction


def decode_predictions(
    predictions: torch.Tensor,
    idx_to_char: Dict[int, str],
    input_lengths: Optional[torch.Tensor] = None,
    blank_idx: int = 0,
) -> List[str]:
    """Decode position-wise logits or token ids into strings."""
    if predictions.dim() == 3:
        token_ids = predictions.argmax(dim=-1)
    elif predictions.dim() == 2:
        token_ids = predictions
    else:
        raise ValueError(f"Invalid prediction tensor shape: {tuple(predictions.shape)}")

    decoded_strings: List[str] = []
    for row in token_ids:
        chars: List[str] = []
        for idx in row.tolist():
            if idx == blank_idx:
                break
            char = idx_to_char.get(int(idx), "")
            if char in {"<PAD>", "<UNK>"}:
                break
            chars.append(char)
        decoded_strings.append("".join(chars))
    return decoded_strings


def decode_targets(
    targets: torch.Tensor,
    lengths: torch.Tensor,
    idx_to_char: Dict[int, str],
) -> List[str]:
    """Convert padded targets back to strings."""
    decoded: List[str] = []
    for target, length in zip(targets, lengths):
        chars = [
            idx_to_char.get(int(idx), "")
            for idx in target[: int(length.item())].tolist()
            if int(idx) > 0
        ]
        decoded.append("".join(chars).replace("<PAD>", "").replace("<UNK>", ""))
    return decoded


def calculate_edit_distance(pred: str, target: str) -> int:
    """Levenshtein distance."""
    if not pred:
        return len(target)
    if not target:
        return len(pred)

    rows = len(pred) + 1
    cols = len(target) + 1
    dp = [[0] * cols for _ in range(rows)]

    for i in range(rows):
        dp[i][0] = i
    for j in range(cols):
        dp[0][j] = j

    for i in range(1, rows):
        for j in range(1, cols):
            substitution_cost = 0 if pred[i - 1] == target[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + substitution_cost,
            )

    return dp[-1][-1]


def calculate_accuracy(predictions: Sequence[str], targets: Sequence[str]) -> float:
    """Exact sequence accuracy."""
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have the same length")
    if not predictions:
        return 0.0
    correct = sum(int(pred == target) for pred, target in zip(predictions, targets))
    return correct / len(predictions)


def calculate_char_accuracy(predictions: Sequence[str], targets: Sequence[str]) -> float:
    """Character accuracy derived from edit distance."""
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have the same length")
    if not predictions:
        return 0.0

    total_chars = 0
    total_edits = 0
    for pred, target in zip(predictions, targets):
        total_chars += max(len(target), 1)
        total_edits += calculate_edit_distance(pred, target)

    return max(0.0, 1.0 - (total_edits / total_chars))


def calculate_normalized_edit_distance(predictions: Sequence[str], targets: Sequence[str]) -> float:
    """Average normalized edit distance."""
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have the same length")
    if not predictions:
        return 0.0

    normalized_distances = [
        calculate_edit_distance(pred, target) / max(len(target), 1)
        for pred, target in zip(predictions, targets)
    ]
    return sum(normalized_distances) / len(normalized_distances)


def summarize_sequence_metrics(predictions: Sequence[str], targets: Sequence[str]) -> Dict[str, float]:
    """Aggregate sequence metrics."""
    edit_distances = [calculate_edit_distance(pred, target) for pred, target in zip(predictions, targets)]
    return {
        "accuracy": calculate_accuracy(predictions, targets),
        "char_accuracy": calculate_char_accuracy(predictions, targets),
        "avg_edit_distance": (sum(edit_distances) / len(edit_distances)) if edit_distances else 0.0,
        "normalized_edit_distance": calculate_normalized_edit_distance(predictions, targets),
    }


def calculate_prediction_confidences(
    logits: torch.Tensor,
    input_lengths: Optional[torch.Tensor] = None,
    blank_idx: int = 0,
) -> List[float]:
    """Average max-softmax confidence over predicted non-pad positions."""
    probabilities = torch.softmax(logits, dim=-1)
    max_probs = probabilities.max(dim=-1).values
    predicted_ids = probabilities.argmax(dim=-1)

    confidences: List[float] = []
    for row_ids, row_probs in zip(predicted_ids.tolist(), max_probs.tolist()):
        retained: List[float] = []
        for idx, prob in zip(row_ids, row_probs):
            if idx == blank_idx:
                break
            retained.append(float(prob))
        confidences.append(sum(retained) / len(retained) if retained else float(sum(row_probs) / len(row_probs)))
    return confidences


def _dummy_idx_to_char(num_classes: int) -> Dict[int, str]:
    mapping = {0: "<PAD>", 1: "<UNK>"}
    charset = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"
    for index, char in enumerate(charset[: max(num_classes - 2, 0)], start=2):
        mapping[index] = char
    return mapping


def main() -> None:
    """Smoke-test the model module."""
    batch_size = 4
    height, width = 80, 200
    max_length = 5
    num_classes = 34

    model = CaptchaCNN(num_classes=num_classes, max_length=max_length)
    inputs = torch.randn(batch_size, 1, height, width)
    outputs = model(inputs)

    print(f"Input shape: {tuple(inputs.shape)}")
    print(f"Output shape: {tuple(outputs.shape)}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    targets = torch.tensor(
        [
            [2, 3, 4, 5, 6],
            [7, 8, 9, 10, 11],
            [12, 13, 14, 15, 16],
            [17, 18, 19, 20, 21],
        ],
        dtype=torch.long,
    )
    lengths = torch.tensor([5, 5, 5, 5], dtype=torch.long)
    loss = PositionSequenceLoss()(outputs, targets, lengths)
    print(f"Sequence loss: {loss.item():.4f}")

    idx_to_char = _dummy_idx_to_char(num_classes)
    decoded = decode_predictions(outputs, idx_to_char)
    print(f"Decoded predictions: {decoded}")


if __name__ == "__main__":
    main()
