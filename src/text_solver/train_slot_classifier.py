#!/usr/bin/env python3
"""
Train a high-accuracy fixed-slot character classifier and report CAPTCHA-level accuracy.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from PIL import Image


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_metadata(metadata_file: str) -> List[Dict[str, str]]:
    return pd.read_csv(metadata_file).fillna("").to_dict(orient="records")


def resolve_image_path(data_dir: str | Path, row: Dict[str, str]) -> Path:
    data_dir = Path(data_dir)
    candidates = [
        data_dir / row["filename"],
        data_dir / row["split"] / row["filename"],
        data_dir.parent / row["split"] / row["filename"],
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve image for row: {row}")


def load_processed_image(image_path: Path, target_size: Tuple[int, int]) -> torch.Tensor:
    if CV2_AVAILABLE:
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            image = np.array(Image.open(image_path).convert("L"))
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    else:
        image = np.array(Image.open(image_path).convert("L").resize(target_size, Image.Resampling.LANCZOS))

    if float(image.mean()) > 127.0:
        image = 255 - image

    tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0) / 255.0
    return tensor


def slice_tensor_positions(
    image_tensor: torch.Tensor,
    max_length: int,
    segment_width: int,
    overlap_ratio: float,
) -> torch.Tensor:
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)

    _, _, height, width = image_tensor.shape
    base_width = width / max_length
    margin = max(2, int(round(base_width * overlap_ratio)))
    segments: List[torch.Tensor] = []

    for position in range(max_length):
        left = int(round(position * base_width))
        right = int(round((position + 1) * base_width))
        start = max(0, left - margin)
        end = min(width, right + margin)
        crop = image_tensor[:, :, :, start:end]
        resized = torch.nn.functional.interpolate(
            crop,
            size=(height, segment_width),
            mode="bilinear",
            align_corners=False,
        )
        segments.append(resized)

    return torch.stack(segments, dim=1)


def build_vocab(rows: Sequence[Dict[str, str]]) -> Tuple[Dict[str, int], Dict[int, str]]:
    chars = sorted({char for row in rows for char in str(row["text"])})
    char_to_idx = {char: index for index, char in enumerate(chars)}
    idx_to_char = {index: char for char, index in char_to_idx.items()}
    return char_to_idx, idx_to_char


class CharacterCropDataset(Dataset):
    """Character-level dataset created from fixed-position CAPTCHA crops."""

    def __init__(
        self,
        rows: List[Dict[str, str]],
        data_dir: str,
        char_to_idx: Dict[str, int],
        max_length: int,
        target_size: Tuple[int, int],
        segment_width: int,
        overlap_ratio: float,
    ):
        self.rows = rows
        self.data_dir = data_dir
        self.char_to_idx = char_to_idx
        self.max_length = max_length
        self.target_size = target_size
        self.segment_width = segment_width
        self.overlap_ratio = overlap_ratio

    def __len__(self) -> int:
        return len(self.rows) * self.max_length

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index // self.max_length]
        position = index % self.max_length
        image_path = resolve_image_path(self.data_dir, row)
        image_tensor = load_processed_image(image_path, self.target_size)
        segments = slice_tensor_positions(image_tensor, self.max_length, self.segment_width, self.overlap_ratio)
        crop = segments[0, position]
        label = self.char_to_idx[row["text"][position]]
        return crop, torch.tensor(label, dtype=torch.long)


class CharClassifier(nn.Module):
    """Shared character classifier."""

    def __init__(self, num_classes: int, hidden_size: int = 256, dropout: float = 0.1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, hidden_size, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_size),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def evaluate_character_accuracy(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images).argmax(dim=-1)
            correct += int((predictions == labels).sum().item())
            total += int(labels.numel())
    return correct / max(total, 1)


def evaluate_captcha_accuracy(
    model: nn.Module,
    rows: Sequence[Dict[str, str]],
    data_dir: str,
    idx_to_char: Dict[int, str],
    device: torch.device,
    max_length: int,
    target_size: Tuple[int, int],
    segment_width: int,
    overlap_ratio: float,
    batch_size: int = 128,
) -> Dict[str, float]:
    model.eval()
    correct = 0
    char_correct = 0
    total_chars = 0
    all_predictions: List[str] = []
    all_targets: List[str] = []

    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start:start + batch_size]
            images = [
                load_processed_image(resolve_image_path(data_dir, row), target_size)
                for row in batch_rows
            ]
            image_tensor = torch.stack(images).to(device)
            segments = slice_tensor_positions(image_tensor, max_length, segment_width, overlap_ratio)
            flat_segments = segments.reshape(len(batch_rows) * max_length, 1, target_size[1], segment_width)
            logits = model(flat_segments).reshape(len(batch_rows), max_length, -1)
            predictions = logits.argmax(dim=-1).cpu().tolist()

            for row, pred_ids in zip(batch_rows, predictions):
                prediction = "".join(idx_to_char[int(idx)] for idx in pred_ids)
                target = str(row["text"])
                all_predictions.append(prediction)
                all_targets.append(target)
                correct += int(prediction == target)
                for pred_char, target_char in zip(prediction, target):
                    char_correct += int(pred_char == target_char)
                    total_chars += 1

    exact_accuracy = correct / max(len(rows), 1)
    character_accuracy = char_correct / max(total_chars, 1)
    return {
        "accuracy": exact_accuracy,
        "char_accuracy": character_accuracy,
        "num_samples": len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a fixed-slot character classifier")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing CAPTCHA data")
    parser.add_argument("--metadata-file", type=str, required=True, help="Metadata CSV file")
    parser.add_argument("--save-dir", type=str, required=True, help="Directory for checkpoints and reports")
    parser.add_argument("--batch-size", type=int, default=512, help="Character batch size")
    parser.add_argument("--num-epochs", type=int, default=15, help="Training epochs")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", help="auto, cuda, mps, or cpu")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--image-width", type=int, default=200, help="Normalized CAPTCHA width")
    parser.add_argument("--image-height", type=int, default=80, help="Normalized CAPTCHA height")
    parser.add_argument("--segment-width", type=int, default=48, help="Per-character crop width")
    parser.add_argument("--overlap-ratio", type=float, default=0.25, help="Overlap around each slot")
    parser.add_argument("--hidden-size", type=int, default=256, help="Character model width")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=3,
        help="Stop after this many non-improving validation epochs (0 disables early stopping)",
    )

    args = parser.parse_args()

    set_global_seed(args.seed)
    device = choose_device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    rows = load_metadata(args.metadata_file)
    max_length = max(len(str(row["text"])) for row in rows)
    if len({len(str(row["text"])) for row in rows}) != 1:
        raise ValueError("Slot classifier currently expects fixed-length CAPTCHAs")

    char_to_idx, idx_to_char = build_vocab(rows)
    print(f"Using device: {device}")
    print(f"Vocabulary size: {len(char_to_idx)}")
    print(f"Max length: {max_length}")

    split_rows = {
        split: [row for row in rows if row["split"] == split]
        for split in ("train", "val", "test")
    }

    train_dataset = CharacterCropDataset(
        rows=split_rows["train"],
        data_dir=args.data_dir,
        char_to_idx=char_to_idx,
        max_length=max_length,
        target_size=(args.image_width, args.image_height),
        segment_width=args.segment_width,
        overlap_ratio=args.overlap_ratio,
    )
    val_dataset = CharacterCropDataset(
        rows=split_rows["val"],
        data_dir=args.data_dir,
        char_to_idx=char_to_idx,
        max_length=max_length,
        target_size=(args.image_width, args.image_height),
        segment_width=args.segment_width,
        overlap_ratio=args.overlap_ratio,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = CharClassifier(
        num_classes=len(char_to_idx),
        hidden_size=args.hidden_size,
        dropout=args.dropout,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    history: List[Dict[str, float]] = []
    best_val_exact = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    best_model_path = save_dir / "best_slot_classifier.pth"

    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.num_epochs}")

        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_char_acc = evaluate_character_accuracy(model, train_loader, device)
        val_char_acc = evaluate_character_accuracy(model, val_loader, device)
        val_captcha_metrics = evaluate_captcha_accuracy(
            model,
            split_rows["val"],
            args.data_dir,
            idx_to_char,
            device,
            max_length,
            (args.image_width, args.image_height),
            args.segment_width,
            args.overlap_ratio,
        )

        record = {
            "epoch": epoch + 1,
            "train_loss": total_loss / max(len(train_loader), 1),
            "train_char_accuracy": train_char_acc,
            "val_char_accuracy": val_char_acc,
            "val_captcha_accuracy": val_captcha_metrics["accuracy"],
            "val_captcha_char_accuracy": val_captcha_metrics["char_accuracy"],
        }
        history.append(record)

        print(
            f"\nEpoch {epoch + 1}/{args.num_epochs}: "
            f"train_loss={record['train_loss']:.4f}, "
            f"train_char={train_char_acc:.4f}, "
            f"val_char={val_char_acc:.4f}, "
            f"val_exact={val_captcha_metrics['accuracy']:.4f}"
        )

        if val_captcha_metrics["accuracy"] > best_val_exact:
            best_val_exact = val_captcha_metrics["accuracy"]
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "char_to_idx": char_to_idx,
                    "idx_to_char": idx_to_char,
                    "history": history,
                    "config": {
                        "hidden_size": args.hidden_size,
                        "dropout": args.dropout,
                        "image_width": args.image_width,
                        "image_height": args.image_height,
                        "segment_width": args.segment_width,
                        "overlap_ratio": args.overlap_ratio,
                        "max_length": max_length,
                    },
                },
                best_model_path,
            )
            print(f"Saved new best model with validation exact accuracy {best_val_exact:.4f}")
        else:
            epochs_without_improvement += 1
            if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
                print(
                    "Early stopping triggered after "
                    f"{epochs_without_improvement} non-improving epochs. "
                    f"Best epoch was {best_epoch}."
                )
                break

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate_captcha_accuracy(
        model,
        split_rows["test"],
        args.data_dir,
        checkpoint["idx_to_char"],
        device,
        max_length,
        (args.image_width, args.image_height),
        args.segment_width,
        args.overlap_ratio,
    )

    summary = {
        "best_epoch": best_epoch,
        "best_val_exact_accuracy": best_val_exact,
        "test_exact_accuracy": test_metrics["accuracy"],
        "test_char_accuracy": test_metrics["char_accuracy"],
        "num_test_samples": test_metrics["num_samples"],
        "epochs_completed": len(history),
        "history": history,
        "checkpoint": str(best_model_path),
    }

    with open(save_dir / "slot_classifier_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)

    print("\nFinal results")
    print(f"Validation exact accuracy: {best_val_exact:.4f}")
    print(f"Test exact accuracy:       {test_metrics['accuracy']:.4f}")
    print(f"Test char accuracy:        {test_metrics['char_accuracy']:.4f}")
    print(f"Summary written to:        {save_dir / 'slot_classifier_summary.json'}")


if __name__ == "__main__":
    main()
