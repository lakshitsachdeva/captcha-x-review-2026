#!/usr/bin/env python3
"""
Train the text CAPTCHA recognizer.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from .model import (
        CaptchaCNN,
        decode_predictions,
        decode_targets,
        PositionSequenceLoss,
        summarize_sequence_metrics,
    )
    from .preprocess import (
        CaptchaDataset,
        CaptchaPreprocessor,
        collate_fn,
        create_char_mapping,
        infer_max_length,
    )
except ImportError:
    from model import CaptchaCNN, PositionSequenceLoss, decode_predictions, decode_targets, summarize_sequence_metrics
    from preprocess import (
        CaptchaDataset,
        CaptchaPreprocessor,
        collate_fn,
        create_char_mapping,
        infer_max_length,
    )


def set_global_seed(seed: int) -> None:
    """Make training as reproducible as possible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Trainer:
    """Research-friendly training loop with reproducible artifacts."""

    def __init__(
        self,
        model: CaptchaCNN,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        save_dir: str,
        char_to_idx: Dict[str, int],
        idx_to_char: Dict[int, str],
        training_config: Dict[str, object],
        num_epochs: int = 30,
        patience: int = 10,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.save_dir = Path(save_dir)
        self.char_to_idx = char_to_idx
        self.idx_to_char = idx_to_char
        self.training_config = training_config
        self.num_epochs = num_epochs
        self.patience = patience

        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=0.5,
            patience=max(2, patience // 2),
        )

        self.history: List[Dict[str, float]] = []
        self.best_score = float("-inf")
        self.best_metrics: Dict[str, float] = {}
        self.best_model_path = self.save_dir / "best_model.pth"
        self.latest_model_path = self.save_dir / "latest_model.pth"
        self.patience_counter = 0

    def _run_epoch(self, dataloader: DataLoader, epoch: int, train: bool) -> Dict[str, float]:
        self.model.train(mode=train)
        phase = "train" if train else "val"
        total_loss = 0.0
        batch_count = 0
        all_predictions: List[str] = []
        all_targets: List[str] = []

        progress_bar = tqdm(dataloader, desc=f"{phase.capitalize()} {epoch + 1}/{self.num_epochs}")
        for images, texts, lengths in progress_bar:
            if images.size(0) == 0 or torch.all(lengths == 0):
                continue

            images = images.to(self.device)
            texts = texts.to(self.device)
            lengths = lengths.to(self.device)

            with torch.set_grad_enabled(train):
                outputs = self.model(images)
                loss = self.criterion(outputs, texts, lengths)

                if not torch.isfinite(loss):
                    continue

                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

            total_loss += float(loss.item())
            batch_count += 1

            predictions = decode_predictions(outputs.detach(), self.idx_to_char)
            targets = decode_targets(texts.detach().cpu(), lengths.detach().cpu(), self.idx_to_char)
            all_predictions.extend(predictions)
            all_targets.extend(targets)

            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        metrics = summarize_sequence_metrics(all_predictions, all_targets)
        metrics["loss"] = total_loss / max(batch_count, 1)
        return metrics

    def _checkpoint_payload(self, epoch: int) -> Dict[str, object]:
        return {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "history": self.history,
            "best_score": self.best_score,
            "best_metrics": self.best_metrics,
            "char_to_idx": self.char_to_idx,
            "idx_to_char": self.idx_to_char,
            "model_config": self.model.get_config(),
            "training_config": self.training_config,
        }

    def _save_checkpoint(self, epoch: int, is_best: bool) -> None:
        payload = self._checkpoint_payload(epoch)
        torch.save(payload, self.latest_model_path)
        if is_best:
            torch.save(payload, self.best_model_path)

    def _save_training_artifacts(self) -> None:
        history_path = self.save_dir / "training_history.json"
        summary_path = self.save_dir / "training_summary.json"

        with open(history_path, "w") as handle:
            json.dump(self.history, handle, indent=2)

        with open(summary_path, "w") as handle:
            json.dump(
                {
                    "best_score": self.best_score,
                    "best_metrics": self.best_metrics,
                    "num_epochs_completed": len(self.history),
                    "model_path": str(self.best_model_path),
                    "training_config": self.training_config,
                },
                handle,
                indent=2,
            )

    def plot_training_history(self) -> None:
        """Save research-friendly training curves."""
        if not self.history:
            return

        epochs = [entry["epoch"] for entry in self.history]
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        axes[0, 0].plot(epochs, [entry["train_loss"] for entry in self.history], label="Train")
        axes[0, 0].plot(epochs, [entry["val_loss"] for entry in self.history], label="Validation")
        axes[0, 0].set_title("Sequence Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(epochs, [entry["train_accuracy"] for entry in self.history], label="Train exact")
        axes[0, 1].plot(epochs, [entry["val_accuracy"] for entry in self.history], label="Val exact")
        axes[0, 1].plot(epochs, [entry["train_char_accuracy"] for entry in self.history], label="Train char")
        axes[0, 1].plot(epochs, [entry["val_char_accuracy"] for entry in self.history], label="Val char")
        axes[0, 1].set_title("Sequence Accuracy")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(epochs, [entry["train_avg_edit_distance"] for entry in self.history], label="Train")
        axes[1, 0].plot(epochs, [entry["val_avg_edit_distance"] for entry in self.history], label="Validation")
        axes[1, 0].set_title("Average Edit Distance")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(epochs, [entry["learning_rate"] for entry in self.history], color="green")
        axes[1, 1].set_title("Learning Rate")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_yscale("log")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.save_dir / "training_history.png", dpi=250, bbox_inches="tight")
        plt.close(fig)

    def train(self) -> Dict[str, float]:
        print(f"Starting training for {self.num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(param.numel() for param in self.model.parameters()):,}")

        start_time = time.time()
        for epoch in range(self.num_epochs):
            epoch_start = time.time()
            train_metrics = self._run_epoch(self.train_loader, epoch, train=True)
            val_metrics = self._run_epoch(self.val_loader, epoch, train=False)

            score = float(val_metrics["accuracy"]) + (0.05 * float(val_metrics["char_accuracy"]))
            is_best = score > self.best_score or (
                np.isclose(score, self.best_score) and val_metrics["accuracy"] > self.best_metrics.get("accuracy", -1.0)
            )

            if is_best:
                self.best_score = score
                self.best_metrics = dict(val_metrics)
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            self.scheduler.step(score)

            record = {
                "epoch": epoch + 1,
                "train_loss": float(train_metrics["loss"]),
                "val_loss": float(val_metrics["loss"]),
                "train_accuracy": float(train_metrics["accuracy"]),
                "val_accuracy": float(val_metrics["accuracy"]),
                "train_char_accuracy": float(train_metrics["char_accuracy"]),
                "val_char_accuracy": float(val_metrics["char_accuracy"]),
                "train_avg_edit_distance": float(train_metrics["avg_edit_distance"]),
                "val_avg_edit_distance": float(val_metrics["avg_edit_distance"]),
                "train_normalized_edit_distance": float(train_metrics["normalized_edit_distance"]),
                "val_normalized_edit_distance": float(val_metrics["normalized_edit_distance"]),
                "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
                "epoch_seconds": float(time.time() - epoch_start),
            }
            self.history.append(record)
            self._save_checkpoint(epoch, is_best=is_best)

            print(
                f"\nEpoch {epoch + 1}/{self.num_epochs} "
                f"({record['epoch_seconds']:.1f}s)"
            )
            print(
                f"  Train: loss={train_metrics['loss']:.4f}, "
                f"exact={train_metrics['accuracy']:.4f}, "
                f"char={train_metrics['char_accuracy']:.4f}, "
                f"ned={train_metrics['normalized_edit_distance']:.4f}"
            )
            print(
                f"  Val:   loss={val_metrics['loss']:.4f}, "
                f"exact={val_metrics['accuracy']:.4f}, "
                f"char={val_metrics['char_accuracy']:.4f}, "
                f"ned={val_metrics['normalized_edit_distance']:.4f}"
            )

            if self.patience_counter >= self.patience:
                print(f"Early stopping triggered after epoch {epoch + 1}")
                break

        self.plot_training_history()
        self._save_training_artifacts()

        total_seconds = time.time() - start_time
        print(f"\nTraining completed in {total_seconds:.1f}s")
        print(f"Best validation metrics: {self.best_metrics}")
        print(f"Best model saved to: {self.best_model_path}")
        return self.best_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the CAPTCHA text recognizer")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing CAPTCHA images")
    parser.add_argument("--metadata-file", type=str, required=True, help="Metadata CSV file")
    parser.add_argument("--save-dir", type=str, default="models/text_captcha", help="Directory for checkpoints")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--num-epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden-size", type=int, default=192, help="Model hidden size")
    parser.add_argument("--num-layers", type=int, default=2, help="Transformer layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout")
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, or cuda")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-length", type=int, default=0, help="Override max text length")
    parser.add_argument("--image-width", type=int, default=200, help="Input width")
    parser.add_argument("--image-height", type=int, default=80, help="Input height")
    parser.add_argument("--augment", action="store_true", help="Enable preprocessing augmentation")
    parser.add_argument("--num-heads", type=int, default=4, help="Transformer attention heads")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Cross-entropy label smoothing")

    args = parser.parse_args()

    set_global_seed(args.seed)

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")
    metadata = pd.read_csv(args.metadata_file)
    all_texts = metadata["text"].astype(str).tolist()
    max_length = args.max_length or infer_max_length(all_texts)
    char_to_idx, idx_to_char = create_char_mapping(all_texts)

    preprocessor = CaptchaPreprocessor(
        target_size=(args.image_width, args.image_height),
        normalize=True,
        augment=args.augment,
    )

    train_dataset = CaptchaDataset(
        data_dir=args.data_dir,
        metadata_file=args.metadata_file,
        preprocessor=preprocessor,
        char_to_idx=char_to_idx,
        max_length=max_length,
        split="train",
    )
    val_dataset = CaptchaDataset(
        data_dir=args.data_dir,
        metadata_file=args.metadata_file,
        preprocessor=CaptchaPreprocessor(
            target_size=(args.image_width, args.image_height),
            normalize=True,
            augment=False,
        ),
        char_to_idx=char_to_idx,
        max_length=max_length,
        split="val",
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Max sequence length: {max_length}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = CaptchaCNN(
        num_classes=len(char_to_idx),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        max_length=max_length,
        num_heads=args.num_heads,
    ).to(device)

    criterion = PositionSequenceLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    training_config = {
        "data_dir": args.data_dir,
        "metadata_file": args.metadata_file,
        "save_dir": args.save_dir,
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "num_heads": args.num_heads,
        "label_smoothing": args.label_smoothing,
        "seed": args.seed,
        "max_length": max_length,
        "image_width": args.image_width,
        "image_height": args.image_height,
        "augment": args.augment,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
    }

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        save_dir=args.save_dir,
        char_to_idx=char_to_idx,
        idx_to_char=idx_to_char,
        training_config=training_config,
        num_epochs=args.num_epochs,
        patience=args.patience,
    )
    trainer.train()


if __name__ == "__main__":
    main()
