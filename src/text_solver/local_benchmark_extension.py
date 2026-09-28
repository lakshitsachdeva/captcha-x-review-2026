#!/usr/bin/env python3
"""
Local-only experimental extension for CAPTCHA-X.

This runner adds:
- multiple local/open-source generator backends
- difficulty-wise evaluation
- cross-generator transfer matrices
- calibration, entropy-based OOD analysis, and abstention curves
- human-baseline study scaffolding
- publication-ready plots, tables, and a paper handoff bundle
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageDraw
from tqdm import tqdm

try:
    from .gen_captchas import DIFFICULTY_PROFILES
    from .local_generators import GeneratorDatasetConfig, create_renderer
    from .reliability import (
        abstention_curve,
        calibration_table,
        entropy_ood_summary,
        expected_calibration_error,
        fit_temperature,
        sequence_confidence,
        sequence_entropy,
    )
    from .train_slot_classifier import (
        CharClassifier,
        choose_device,
        load_processed_image,
        resolve_image_path,
        set_global_seed,
        slice_tensor_positions,
    )
except ImportError:
    from gen_captchas import DIFFICULTY_PROFILES
    from local_generators import GeneratorDatasetConfig, create_renderer
    from reliability import (
        abstention_curve,
        calibration_table,
        entropy_ood_summary,
        expected_calibration_error,
        fit_temperature,
        sequence_confidence,
        sequence_entropy,
    )
    from train_slot_classifier import (
        CharClassifier,
        choose_device,
        load_processed_image,
        resolve_image_path,
        set_global_seed,
        slice_tensor_positions,
    )


DIFFICULTY_ORDER = ["clean", "easy", "medium", "hard"]


def load_config(config_path: str) -> Dict[str, object]:
    with open(config_path, "r") as handle:
        return json.load(handle)


def build_char_mapping(characters: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    unique_chars = list(dict.fromkeys(characters))
    char_to_idx = {char: index for index, char in enumerate(unique_chars)}
    idx_to_char = {index: char for char, index in char_to_idx.items()}
    return char_to_idx, idx_to_char


def load_rows(metadata_file: str | Path, split: str | None = None) -> List[Dict[str, str]]:
    rows = pd.read_csv(metadata_file).fillna("").to_dict(orient="records")
    if split is None:
        return rows
    return [row for row in rows if row.get("split") == split]


def dataset_is_complete(dataset_dir: str | Path, expected_samples: int) -> bool:
    """Return True when an existing generated dataset matches the requested scale."""
    dataset_path = Path(dataset_dir)
    metadata_file = dataset_path / "metadata.csv"
    manifest_file = dataset_path / "dataset_manifest.json"
    if not metadata_file.exists() or not manifest_file.exists():
        return False

    try:
        metadata_df = pd.read_csv(metadata_file)
    except Exception:
        return False

    required_columns = {"filename", "text", "split", "difficulty", "generator_name", "generator_backend"}
    if len(metadata_df) != expected_samples or not required_columns.issubset(metadata_df.columns):
        return False

    split_counts = metadata_df["split"].value_counts().to_dict()
    png_count = 0
    for split_name in ("train", "val", "test"):
        split_dir = dataset_path / split_name
        if not split_dir.exists():
            return False
        observed_count = sum(1 for _ in split_dir.glob("*.png"))
        png_count += observed_count
        if observed_count != int(split_counts.get(split_name, 0)):
            return False

    return png_count == expected_samples


def targets_from_rows(rows: Sequence[Dict[str, str]], char_to_idx: Dict[str, int], max_length: int) -> torch.Tensor:
    targets = torch.zeros(len(rows), max_length, dtype=torch.long)
    for row_index, row in enumerate(rows):
        text = str(row["text"])[:max_length]
        targets[row_index, : len(text)] = torch.tensor([char_to_idx[char] for char in text], dtype=torch.long)
    return targets


def build_segment_batch(
    rows: Sequence[Dict[str, str]],
    data_dir: str | Path,
    char_to_idx: Dict[str, int],
    *,
    device: torch.device,
    max_length: int,
    target_size: Tuple[int, int],
    segment_width: int,
    overlap_ratio: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load each CAPTCHA once and return flattened character crops plus labels."""
    images = [load_processed_image(resolve_image_path(data_dir, row), target_size) for row in rows]
    image_tensor = torch.stack(images).to(device)
    segments = slice_tensor_positions(image_tensor, max_length, segment_width, overlap_ratio)
    flat_segments = segments.reshape(len(rows) * max_length, 1, target_size[1], segment_width)
    labels = targets_from_rows(rows, char_to_idx, max_length).reshape(-1).to(device)
    return flat_segments, labels


def to_cpu_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Store checkpoints on CPU so they reload cleanly across CPU/CUDA/MPS devices."""
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def move_optimizer_state_to_device(optimizer: optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def predict_rows(
    model: nn.Module,
    rows: Sequence[Dict[str, str]],
    data_dir: str | Path,
    idx_to_char: Dict[int, str],
    char_to_idx: Dict[str, int],
    device: torch.device,
    *,
    max_length: int,
    target_size: Tuple[int, int],
    segment_width: int,
    overlap_ratio: float,
    batch_size: int = 128,
    temperature: float = 1.0,
    return_tensors: bool = False,
) -> Tuple[Dict[str, float], pd.DataFrame, pd.DataFrame, Dict[str, torch.Tensor] | None]:
    model.eval()
    total_correct = 0
    total_char_correct = 0
    total_chars = 0

    detail_rows: List[Dict[str, object]] = []
    all_logits: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []

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

            confidences = sequence_confidence(logits, temperature=temperature)
            entropies = sequence_entropy(logits, temperature=temperature)
            pred_ids = logits.argmax(dim=-1).detach().cpu().numpy()

            batch_targets = targets_from_rows(batch_rows, char_to_idx, max_length)

            if return_tensors:
                all_logits.append(logits.detach().cpu())
                all_targets.append(batch_targets.detach().cpu())

            for row, pred_row, target_row, confidence, entropy in zip(
                batch_rows,
                pred_ids,
                batch_targets.numpy(),
                confidences,
                entropies,
            ):
                prediction = "".join(idx_to_char[int(idx)] for idx in pred_row)
                target = str(row["text"])
                exact = prediction == target
                char_matches = sum(int(pred_char == tgt_char) for pred_char, tgt_char in zip(prediction, target))
                char_accuracy = char_matches / max(len(target), 1)

                total_correct += int(exact)
                total_char_correct += char_matches
                total_chars += len(target)

                detail_rows.append(
                    {
                        "filename": row["filename"],
                        "target": target,
                        "prediction": prediction,
                        "exact_correct": exact,
                        "char_accuracy_sample": char_accuracy,
                        "confidence": float(confidence),
                        "entropy": float(entropy),
                        "generator_name": row.get("generator_name", ""),
                        "generator_backend": row.get("generator_backend", ""),
                        "difficulty": row.get("difficulty", ""),
                        "split": row.get("split", ""),
                    }
                )

    details_df = pd.DataFrame(detail_rows)
    exact_accuracy = total_correct / max(len(rows), 1)
    char_accuracy = total_char_correct / max(total_chars, 1)
    ece = expected_calibration_error(details_df["confidence"].tolist(), details_df["exact_correct"].tolist())
    calibration_df = pd.DataFrame(
        calibration_table(details_df["confidence"].tolist(), details_df["exact_correct"].tolist())
    )

    summary = {
        "exact_accuracy": float(exact_accuracy),
        "char_accuracy": float(char_accuracy),
        "ece": float(ece),
        "avg_confidence": float(details_df["confidence"].mean()) if not details_df.empty else 0.0,
        "avg_entropy": float(details_df["entropy"].mean()) if not details_df.empty else 0.0,
        "num_samples": int(len(rows)),
    }

    tensors = None
    if return_tensors:
        tensors = {
            "logits": torch.cat(all_logits, dim=0) if all_logits else torch.empty(0),
            "targets": torch.cat(all_targets, dim=0) if all_targets else torch.empty(0, dtype=torch.long),
        }

    return summary, details_df, calibration_df, tensors


def train_slot_model(
    train_rows: Sequence[Dict[str, str]],
    val_rows: Sequence[Dict[str, str]],
    data_dir: str | Path,
    *,
    device: torch.device,
    char_to_idx: Dict[str, int],
    idx_to_char: Dict[int, str],
    max_length: int,
    target_size: Tuple[int, int],
    training_cfg: Dict[str, object],
    save_dir: Path,
) -> Dict[str, object]:
    save_dir.mkdir(parents=True, exist_ok=True)

    model = CharClassifier(
        num_classes=len(char_to_idx),
        hidden_size=int(training_cfg["hidden_size"]),
        dropout=float(training_cfg["dropout"]),
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=float(training_cfg["learning_rate"]))
    criterion = nn.CrossEntropyLoss()

    checkpoint_path = save_dir / "best_model.pth"
    latest_checkpoint_path = save_dir / "latest_checkpoint.pth"
    total_epochs = int(training_cfg["epochs"])
    train_batch_size = int(training_cfg["batch_size"])
    eval_batch_size = int(training_cfg["eval_batch_size"])
    segment_width = int(training_cfg["segment_width"])
    overlap_ratio = float(training_cfg["overlap_ratio"])

    history: List[Dict[str, float]] = []
    best_state = None
    best_exact = float("-inf")
    start_epoch = 0

    if checkpoint_path.exists() and (save_dir / "training_history.csv").exists():
        history_df = pd.read_csv(save_dir / "training_history.csv")
        if len(history_df) >= total_epochs:
            complete_state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            model.load_state_dict(complete_state["model_state_dict"])
            return {
                "model": model,
                "checkpoint_path": str(checkpoint_path),
                "history": history_df.to_dict(orient="records"),
                "best_val_exact": float(history_df["val_exact_accuracy"].max()),
            }

    if latest_checkpoint_path.exists():
        latest_state = torch.load(latest_checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(latest_state["model_state_dict"])
        if "optimizer_state_dict" in latest_state:
            optimizer.load_state_dict(latest_state["optimizer_state_dict"])
            move_optimizer_state_to_device(optimizer, device)
        history = list(latest_state.get("history", []))
        best_state = latest_state.get("best_state")
        best_exact = float(latest_state.get("best_exact", best_exact))
        start_epoch = int(latest_state.get("epoch", 0))

    train_rows_list = list(train_rows)
    val_rows_list = list(val_rows)

    for epoch in range(start_epoch, total_epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        epoch_indices = np.arange(len(train_rows_list))
        np.random.default_rng(epoch).shuffle(epoch_indices)
        progress_bar = tqdm(
            range(0, len(epoch_indices), train_batch_size),
            desc=f"Train {save_dir.name} epoch {epoch + 1}",
            leave=False,
        )
        for start in progress_bar:
            batch_indices = epoch_indices[start:start + train_batch_size]
            batch_rows = [train_rows_list[int(index)] for index in batch_indices]
            crops, labels = build_segment_batch(
                batch_rows,
                data_dir,
                char_to_idx,
                device=device,
                max_length=max_length,
                target_size=target_size,
                segment_width=segment_width,
                overlap_ratio=overlap_ratio,
            )

            logits = model(crops)
            loss = criterion(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += float(loss.item()) * crops.size(0)
            preds = logits.argmax(dim=-1)
            running_correct += int((preds == labels).sum().item())
            running_total += int(labels.numel())
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / max(running_total, 1)
        train_char_acc = running_correct / max(running_total, 1)

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for start in range(0, len(val_rows_list), eval_batch_size):
                batch_rows = val_rows_list[start:start + eval_batch_size]
                crops, labels = build_segment_batch(
                    batch_rows,
                    data_dir,
                    char_to_idx,
                    device=device,
                    max_length=max_length,
                    target_size=target_size,
                    segment_width=segment_width,
                    overlap_ratio=overlap_ratio,
                )
                preds = model(crops).argmax(dim=-1)
                val_correct += int((preds == labels).sum().item())
                val_total += int(labels.numel())
        val_crop_acc = val_correct / max(val_total, 1)

        val_summary, _, _, _ = predict_rows(
            model,
            val_rows,
            data_dir,
            idx_to_char,
            char_to_idx,
            device,
            max_length=max_length,
            target_size=target_size,
            segment_width=segment_width,
            overlap_ratio=overlap_ratio,
            batch_size=eval_batch_size,
            temperature=1.0,
            return_tensors=False,
        )
        val_exact = float(val_summary["exact_accuracy"])
        val_char_acc = float(val_summary["char_accuracy"])

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_crop_accuracy": train_char_acc,
                "val_crop_accuracy": val_crop_acc,
                "val_exact_accuracy": val_exact,
                "val_char_accuracy": val_char_acc,
            }
        )

        if val_exact > best_exact:
            best_exact = val_exact
            best_state = {
                "model_state_dict": to_cpu_state_dict(model),
                "training_cfg": training_cfg,
                "char_to_idx": char_to_idx,
                "idx_to_char": idx_to_char,
                "max_length": max_length,
                "target_size": list(target_size),
            }
            torch.save(best_state, checkpoint_path)

        latest_state = {
            "epoch": epoch + 1,
            "model_state_dict": to_cpu_state_dict(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "best_exact": best_exact,
            "best_state": best_state,
            "training_cfg": training_cfg,
            "char_to_idx": char_to_idx,
            "idx_to_char": idx_to_char,
            "max_length": max_length,
            "target_size": list(target_size),
        }
        torch.save(latest_state, latest_checkpoint_path)

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint")

    torch.save(best_state, checkpoint_path)
    with open(save_dir / "training_history.json", "w") as handle:
        json.dump(history, handle, indent=2)

    history_df = pd.DataFrame(history)
    history_df.to_csv(save_dir / "training_history.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history_df["epoch"], history_df["train_loss"], marker="o")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history_df["epoch"], history_df["val_exact_accuracy"], marker="o", label="Val exact")
    axes[1].plot(history_df["epoch"], history_df["val_char_accuracy"], marker="o", label="Val char")
    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "training_curves.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    model.load_state_dict(best_state["model_state_dict"])
    return {
        "model": model,
        "checkpoint_path": str(checkpoint_path),
        "history": history,
        "best_val_exact": best_exact,
    }


def make_montage(dataset_index: Dict[Tuple[str, str], Path], output_path: Path, samples_per_condition: int = 1) -> None:
    generator_names = sorted({generator for generator, _ in dataset_index})
    difficulties = [difficulty for difficulty in DIFFICULTY_ORDER if any((generator, difficulty) in dataset_index for generator in generator_names)]
    thumb_width = 200
    thumb_height = 80
    margin = 20
    header_height = 50
    cell_height = thumb_height + 40
    canvas_width = margin * 2 + len(difficulties) * (thumb_width + margin)
    canvas_height = margin * 2 + len(generator_names) * cell_height + header_height
    canvas = Image.new("RGB", (canvas_width, canvas_height), (250, 248, 243))
    draw = ImageDraw.Draw(canvas)

    for column, difficulty in enumerate(difficulties):
        x = margin + column * (thumb_width + margin)
        draw.text((x + 50, 10), difficulty.title(), fill=(32, 41, 55))

    for row_index, generator in enumerate(generator_names):
        y = header_height + margin + row_index * cell_height
        draw.text((10, y + 24), generator, fill=(32, 41, 55))
        for column, difficulty in enumerate(difficulties):
            dataset_dir = dataset_index.get((generator, difficulty))
            if dataset_dir is None:
                continue
            metadata_path = dataset_dir / "metadata.csv"
            if not metadata_path.exists():
                continue
            rows = load_rows(metadata_path, split="test")
            if not rows:
                continue
            sample_row = rows[0]
            image_path = resolve_image_path(dataset_dir, sample_row)
            image = Image.open(image_path).convert("RGB").resize((thumb_width, thumb_height))
            x = margin + column * (thumb_width + margin)
            canvas.paste(image, (x, y))
            draw.text((x, y + thumb_height + 8), sample_row["text"], fill=(75, 85, 99))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, "PNG")


def write_methods_addendum(
    config: Dict[str, object],
    evaluation_df: pd.DataFrame,
    ood_df: pd.DataFrame,
    output_path: Path,
) -> None:
    matched = evaluation_df[evaluation_df["train_generator"] == evaluation_df["test_generator"]].copy()
    matched["difficulty"] = pd.Categorical(matched["difficulty"], categories=DIFFICULTY_ORDER, ordered=True)
    matched = matched.sort_values(["train_generator", "difficulty"])

    matched_lines = "\n".join(
        f"| {row.train_generator} | {row.difficulty} | {row.exact_accuracy:.4f} | {row.char_accuracy:.4f} | {row.ece:.4f} |"
        for row in matched.itertuples()
    )
    if not matched_lines:
        matched_lines = "| - | - | - | - | - |"

    ood_lines = "\n".join(
        f"| {row.train_generator} | {row.ood_condition} | {row.auroc:.4f} | {row.id_mean_entropy:.4f} | {row.ood_mean_entropy:.4f} |"
        for row in ood_df.itertuples()
    )
    if not ood_lines:
        ood_lines = "| - | - | - | - | - |"

    text = f"""# Methods Addendum For Workshop Integration

## Experimental Setup

This local-only extension augments CAPTCHA-X with a cross-generator transfer benchmark over multiple open-source text CAPTCHA renderers that are configured and executed entirely offline. No live websites, third-party endpoints, or production CAPTCHA services are queried.

- Generators: {", ".join(generator['name'] for generator in config['generators'])}
- Difficulty profiles: {", ".join(config['difficulties'])}
- Train difficulty for model fitting: `{config['train_difficulty']}`
- Samples per generator/difficulty dataset: `{config['num_samples_per_dataset']}`
- Image size: `{config['image_size'][0]} x {config['image_size'][1]}`
- CAPTCHA length: `{config['captcha_length']}`
- Character set size: `{len(config['characters'])}`

## Reliability Extensions

The extension evaluates four reliability components:

1. Post-hoc temperature scaling on the held-out validation split of the training generator.
2. Expected calibration error (ECE) computed from sequence confidence and exact correctness.
3. Entropy-based OOD detection using sequence entropy as the uncertainty score.
4. Abstention curves computed by sweeping confidence thresholds and measuring coverage versus selective accuracy.

## Matched-Generator Results

| Train generator | Difficulty | Exact accuracy | Character accuracy | ECE |
| --- | --- | --- | --- | --- |
{matched_lines}

## Entropy-Based OOD Summary

| Train generator | OOD condition | AUROC | ID mean entropy | OOD mean entropy |
| --- | --- | --- | --- | --- |
{ood_lines}

## Paper Integration Notes

This extension is suitable for a workshop-style paper update in three places:

1. Experimental design: describe the local-only cross-generator transfer setup, the four difficulty profiles, and the defensive framing.
2. Results: add the matched-generator difficulty plots, the transfer matrix, the calibration-versus-difficulty plot, and the abstention curves.
3. Limitations and responsible use: state explicitly that all data are self-generated or locally rendered and that no live service evaluation is performed.
"""
    output_path.write_text(text)


def create_human_baseline_pack(
    dataset_index: Dict[Tuple[str, str], Path],
    output_dir: Path,
    *,
    per_condition: int = 4,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stimuli_dir = output_dir / "stimuli"
    stimuli_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: List[Dict[str, object]] = []
    response_rows: List[Dict[str, object]] = []

    for generator_name, difficulty in sorted(dataset_index):
        dataset_dir = dataset_index[(generator_name, difficulty)]
        rows = load_rows(dataset_dir / "metadata.csv", split="test")[:per_condition]
        for row_index, row in enumerate(rows):
            stimulus_id = f"{generator_name}_{difficulty}_{row_index:03d}"
            source_path = resolve_image_path(dataset_dir, row)
            target_path = stimuli_dir / f"{stimulus_id}.png"
            shutil.copy2(source_path, target_path)

            manifest_rows.append(
                {
                    "stimulus_id": stimulus_id,
                    "generator_name": generator_name,
                    "difficulty": difficulty,
                    "image_path": str(target_path.relative_to(output_dir)),
                    "ground_truth": row["text"],
                    "split": row["split"],
                }
            )
            response_rows.append(
                {
                    "participant_id": "",
                    "stimulus_id": stimulus_id,
                    "response_text": "",
                    "solve_time_seconds": "",
                    "difficulty_rating_1_to_5": "",
                    "notes": "",
                }
            )

    pd.DataFrame(manifest_rows).to_csv(output_dir / "stimulus_manifest.csv", index=False)
    pd.DataFrame(response_rows).to_csv(output_dir / "participant_response_template.csv", index=False)
    (output_dir / "STUDY_PROTOCOL.md").write_text(
        """# Small Human-Baseline Study Protocol

This template is intended for local, IRB-aware pilot studies only.

## Recommended captured fields

- `response_text`: participant transcription of the CAPTCHA
- `solve_time_seconds`: wall-clock time taken for the attempt
- `difficulty_rating_1_to_5`: subjective difficulty rating after each item

## Suggested aggregate metrics

- solve rate by generator and difficulty
- median solve time by generator and difficulty
- mean difficulty rating by generator and difficulty

## Notes

- All stimuli in this folder were generated locally by the benchmark suite.
- No external services or third-party CAPTCHA endpoints are involved.
"""
    )


def plot_exact_vs_difficulty(matched_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(8, 4.8))
    sns.lineplot(
        data=matched_df,
        x="difficulty",
        y="exact_accuracy",
        hue="train_generator",
        marker="o",
    )
    plt.ylim(0.0, 1.05)
    plt.title("Exact String Accuracy vs Difficulty")
    plt.ylabel("Exact accuracy")
    plt.xlabel("Difficulty")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close()


def plot_char_vs_difficulty(matched_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(8, 4.8))
    sns.lineplot(
        data=matched_df,
        x="difficulty",
        y="char_accuracy",
        hue="train_generator",
        marker="o",
    )
    plt.ylim(0.0, 1.05)
    plt.title("Character Accuracy vs Difficulty")
    plt.ylabel("Character accuracy")
    plt.xlabel("Difficulty")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close()


def plot_ece_vs_difficulty(matched_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(8, 4.8))
    sns.lineplot(
        data=matched_df,
        x="difficulty",
        y="ece",
        hue="train_generator",
        marker="o",
    )
    plt.ylim(bottom=0.0)
    plt.title("Calibration Error vs Difficulty")
    plt.ylabel("Expected calibration error")
    plt.xlabel("Difficulty")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close()


def plot_abstention_curves(abstention_df: pd.DataFrame, output_path: Path) -> None:
    matched = abstention_df[abstention_df["train_generator"] == abstention_df["test_generator"]].copy()
    generators = sorted(matched["train_generator"].unique())
    fig, axes = plt.subplots(1, max(len(generators), 1), figsize=(6 * max(len(generators), 1), 4), squeeze=False)
    for axis, generator in zip(axes.flat, generators):
        subset = matched[matched["train_generator"] == generator]
        sns.lineplot(
            data=subset,
            x="coverage",
            y="selective_accuracy",
            hue="difficulty",
            marker="o",
            ax=axis,
        )
        axis.set_title(f"Abstention Curve: {generator}")
        axis.set_xlabel("Coverage")
        axis.set_ylabel("Selective accuracy")
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.05)
        axis.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_transfer_matrix(evaluation_df: pd.DataFrame, difficulty: str, output_path: Path, csv_path: Path) -> None:
    matrix_df = evaluation_df[evaluation_df["difficulty"] == difficulty].pivot(
        index="train_generator",
        columns="test_generator",
        values="exact_accuracy",
    )
    matrix_df.to_csv(csv_path)
    plt.figure(figsize=(6.4, 5.2))
    sns.heatmap(matrix_df, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0.0, vmax=1.0)
    plt.title(f"Train-on-A / Test-on-B Exact Accuracy ({difficulty})")
    plt.ylabel("Train generator")
    plt.xlabel("Test generator")
    plt.tight_layout()
    plt.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close()


def write_summary_markdown(
    config: Dict[str, object],
    evaluation_df: pd.DataFrame,
    temperature_df: pd.DataFrame,
    ood_df: pd.DataFrame,
    output_path: Path,
) -> None:
    matched = evaluation_df[evaluation_df["train_generator"] == evaluation_df["test_generator"]].copy()
    matched["difficulty"] = pd.Categorical(matched["difficulty"], categories=DIFFICULTY_ORDER, ordered=True)
    matched = matched.sort_values(["train_generator", "difficulty"])

    matched_lines = "\n".join(
        f"| {row.train_generator} | {row.difficulty} | {row.exact_accuracy:.4f} | {row.char_accuracy:.4f} | {row.ece:.4f} |"
        for row in matched.itertuples()
    ) or "| - | - | - | - | - |"

    temp_lines = "\n".join(
        f"| {row.train_generator} | {row.temperature:.4f} | {row.nll_before:.4f} | {row.nll_after:.4f} |"
        for row in temperature_df.itertuples()
    ) or "| - | - | - | - |"

    ood_lines = "\n".join(
        f"| {row.train_generator} | {row.ood_condition} | {row.auroc:.4f} |"
        for row in ood_df.itertuples()
    ) or "| - | - | - |"

    output_path.write_text(
        f"""# Local Benchmark Extension Summary

## Defensive Scope

This run was executed entirely on local, self-generated CAPTCHA datasets. It does not evaluate, query, scrape, or interact with any live or third-party CAPTCHA service.

## Config Snapshot

- Train difficulty: `{config['train_difficulty']}`
- Transfer matrix difficulty: `{config['transfer_matrix_difficulty']}`
- Samples per dataset: `{config['num_samples_per_dataset']}`
- Generators: {", ".join(generator['name'] for generator in config['generators'])}

## Matched-Generator Metrics

| Train generator | Difficulty | Exact accuracy | Character accuracy | ECE |
| --- | --- | --- | --- | --- |
{matched_lines}

## Temperature Scaling

| Train generator | Temperature | NLL before | NLL after |
| --- | --- | --- | --- |
{temp_lines}

## Entropy-Based OOD Summary

| Train generator | OOD condition | AUROC |
| --- | --- | --- |
{ood_lines}

## Generated Paper Assets

- `plots/exact_accuracy_vs_difficulty.png`
- `plots/char_accuracy_vs_difficulty.png`
- `plots/calibration_error_vs_difficulty.png`
- `plots/abstention_curves.png`
- `plots/transfer_matrix_exact_accuracy.png`
- `plots/generator_examples.png`
- `tables/evaluation_summary.csv`
- `tables/transfer_matrix_exact_accuracy.csv`
- `docs/methods_addendum.md`
- `human_baseline/participant_response_template.csv`

## Recommended Paper Integration

Add this extension as a benchmark-expansion subsection:

1. Describe the local-only multi-generator setup and the four audited difficulty profiles.
2. Add the exact-accuracy, character-accuracy, and ECE-versus-difficulty figures.
3. Add the generator-transfer matrix as a robustness table or heatmap figure.
4. Mention the human-baseline template as future evaluation scaffolding rather than a completed user study.
"""
    )


def bundle_outputs(output_root: Path, bundle_dir: Path, reference_paper: str | None) -> Path:
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    for name in ("plots", "tables", "docs", "human_baseline", "configs"):
        source = output_root / name
        if source.exists():
            shutil.copytree(source, bundle_dir / name)

    for name in ("resolved_config.json",):
        source = output_root / name
        if source.exists():
            shutil.copy2(source, bundle_dir / name)

    if reference_paper and Path(reference_paper).exists():
        paper_dir = bundle_dir / "current_paper_reference"
        paper_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(reference_paper, paper_dir / "current_paper_reference.tex")

    manifest_lines = []
    for path in sorted(bundle_dir.rglob("*")):
        if path.is_file():
            manifest_lines.append(str(path.relative_to(bundle_dir)))
    (bundle_dir / "bundle_manifest.txt").write_text("\n".join(manifest_lines) + "\n")

    archive_path = shutil.make_archive(str(bundle_dir), "gztar", root_dir=bundle_dir.parent, base_dir=bundle_dir.name)
    return Path(archive_path)


def run_experiment(config: Dict[str, object], config_path: str, reference_paper: str | None = None) -> Path:
    seed = int(config.get("seed", 42))
    set_global_seed(seed)
    output_root = Path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)

    directories = {
        "datasets": output_root / "datasets",
        "models": output_root / "models",
        "results": output_root / "results",
        "plots": output_root / "plots",
        "tables": output_root / "tables",
        "docs": output_root / "docs",
        "human_baseline": output_root / "human_baseline",
        "configs": output_root / "configs",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    shutil.copy2(config_path, directories["configs"] / Path(config_path).name)
    with open(output_root / "resolved_config.json", "w") as handle:
        json.dump(config, handle, indent=2)

    device = choose_device(str(config.get("device", "auto")))
    target_size = (int(config["image_size"][0]), int(config["image_size"][1]))
    characters = str(config["characters"])
    max_length = int(config["captcha_length"])
    char_to_idx, idx_to_char = build_char_mapping(characters)

    dataset_index: Dict[Tuple[str, str], Path] = {}
    difficulties = [difficulty for difficulty in DIFFICULTY_ORDER if difficulty in config["difficulties"]]
    generators = config["generators"]

    for generator_index, generator_cfg in enumerate(generators):
        for difficulty_index, difficulty in enumerate(difficulties):
            dataset_dir = directories["datasets"] / generator_cfg["name"] / difficulty
            renderer_cfg = GeneratorDatasetConfig(
                name=str(generator_cfg["name"]),
                backend=str(generator_cfg["backend"]),
                width=int(config["image_size"][0]),
                height=int(config["image_size"][1]),
                captcha_length=max_length,
                difficulty=difficulty,
                num_samples=int(config["num_samples_per_dataset"]),
                split_ratio=tuple(config["split_ratio"]),
                seed=seed + generator_index * 1000 + difficulty_index * 100,
                characters=characters,
                font_sizes=tuple(config.get("font_sizes", [40, 45, 50, 55, 60])),
                font_paths=generator_cfg.get("font_paths"),
            )
            if dataset_is_complete(dataset_dir, int(config["num_samples_per_dataset"])):
                print(f"Dataset already complete, reusing: {dataset_dir}")
            else:
                renderer = create_renderer(renderer_cfg)
                renderer.generate_dataset(dataset_dir)
            dataset_index[(generator_cfg["name"], difficulty)] = dataset_dir

    make_montage(dataset_index, directories["plots"] / "generator_examples.png")
    create_human_baseline_pack(dataset_index, directories["human_baseline"], per_condition=int(config["human_study"]["samples_per_condition"]))

    training_cfg = dict(config["training"])
    temperature_rows: List[Dict[str, object]] = []
    evaluation_rows: List[Dict[str, object]] = []
    calibration_rows: List[Dict[str, object]] = []
    abstention_rows: List[Dict[str, object]] = []
    all_detail_frames: List[pd.DataFrame] = []
    entropy_by_condition: Dict[Tuple[str, str, str], np.ndarray] = {}

    for generator_cfg in generators:
        train_generator = str(generator_cfg["name"])
        train_difficulty = str(config["train_difficulty"])
        train_dataset_dir = dataset_index[(train_generator, train_difficulty)]
        metadata_file = train_dataset_dir / "metadata.csv"

        train_rows = load_rows(metadata_file, split="train")
        val_rows = load_rows(metadata_file, split="val")
        model_dir = directories["models"] / train_generator

        training_result = train_slot_model(
            train_rows,
            val_rows,
            train_dataset_dir,
            device=device,
            char_to_idx=char_to_idx,
            idx_to_char=idx_to_char,
            max_length=max_length,
            target_size=target_size,
            training_cfg=training_cfg,
            save_dir=model_dir,
        )
        model = training_result["model"]

        val_summary, _, _, val_tensors = predict_rows(
            model,
            val_rows,
            train_dataset_dir,
            idx_to_char,
            char_to_idx,
            device,
            max_length=max_length,
            target_size=target_size,
            segment_width=int(training_cfg["segment_width"]),
            overlap_ratio=float(training_cfg["overlap_ratio"]),
            batch_size=int(training_cfg["eval_batch_size"]),
            temperature=1.0,
            return_tensors=True,
        )
        assert val_tensors is not None
        temp_result = fit_temperature(val_tensors["logits"], val_tensors["targets"])
        temperature_rows.append(
            {
                "train_generator": train_generator,
                "temperature": temp_result.temperature,
                "nll_before": temp_result.nll_before,
                "nll_after": temp_result.nll_after,
                "val_exact_accuracy": val_summary["exact_accuracy"],
            }
        )

        with open(model_dir / "temperature_scaling.json", "w") as handle:
            json.dump(asdict(temp_result), handle, indent=2)

        for test_generator_cfg in generators:
            test_generator = str(test_generator_cfg["name"])
            for difficulty in difficulties:
                dataset_dir = dataset_index[(test_generator, difficulty)]
                rows = load_rows(dataset_dir / "metadata.csv", split="test")
                summary, details_df, calibration_df, _ = predict_rows(
                    model,
                    rows,
                    dataset_dir,
                    idx_to_char,
                    char_to_idx,
                    device,
                    max_length=max_length,
                    target_size=target_size,
                    segment_width=int(training_cfg["segment_width"]),
                    overlap_ratio=float(training_cfg["overlap_ratio"]),
                    batch_size=int(training_cfg["eval_batch_size"]),
                    temperature=temp_result.temperature,
                    return_tensors=False,
                )

                details_df.insert(0, "train_generator", train_generator)
                details_df.insert(1, "test_generator", test_generator)
                details_df.to_csv(
                    directories["results"] / f"details_{train_generator}_to_{test_generator}_{difficulty}.csv",
                    index=False,
                )
                all_detail_frames.append(details_df)
                entropy_by_condition[(train_generator, test_generator, difficulty)] = details_df["entropy"].to_numpy()

                calibration_df.insert(0, "train_generator", train_generator)
                calibration_df.insert(1, "test_generator", test_generator)
                calibration_df.insert(2, "difficulty", difficulty)
                calibration_rows.extend(calibration_df.to_dict(orient="records"))

                evaluation_rows.append(
                    {
                        "train_generator": train_generator,
                        "test_generator": test_generator,
                        "difficulty": difficulty,
                        **summary,
                    }
                )

                curve_rows = abstention_curve(
                    details_df["confidence"].tolist(),
                    details_df["exact_correct"].tolist(),
                    thresholds=config["reliability"]["abstention_thresholds"],
                )
                for row in curve_rows:
                    abstention_rows.append(
                        {
                            "train_generator": train_generator,
                            "test_generator": test_generator,
                            "difficulty": difficulty,
                            **row,
                        }
                    )

    evaluation_df = pd.DataFrame(evaluation_rows)
    temperature_df = pd.DataFrame(temperature_rows)
    calibration_df = pd.DataFrame(calibration_rows)
    abstention_df = pd.DataFrame(abstention_rows)
    details_df = pd.concat(all_detail_frames, ignore_index=True) if all_detail_frames else pd.DataFrame()

    evaluation_df["difficulty"] = pd.Categorical(evaluation_df["difficulty"], categories=DIFFICULTY_ORDER, ordered=True)
    evaluation_df = evaluation_df.sort_values(["train_generator", "test_generator", "difficulty"])
    evaluation_df.to_csv(directories["tables"] / "evaluation_summary.csv", index=False)
    temperature_df.to_csv(directories["tables"] / "temperature_scaling_summary.csv", index=False)
    calibration_df.to_csv(directories["tables"] / "calibration_bins.csv", index=False)
    abstention_df.to_csv(directories["tables"] / "abstention_curves.csv", index=False)
    details_df.to_csv(directories["tables"] / "all_prediction_details.csv", index=False)

    ood_rows: List[Dict[str, object]] = []
    transfer_difficulty = str(config["transfer_matrix_difficulty"])
    for generator_cfg in generators:
        train_generator = str(generator_cfg["name"])
        id_scores = entropy_by_condition.get((train_generator, train_generator, str(config["train_difficulty"])), np.array([]))
        if id_scores.size == 0:
            continue
        for test_generator_cfg in generators:
            test_generator = str(test_generator_cfg["name"])
            if test_generator == train_generator:
                continue
            ood_scores = entropy_by_condition.get((train_generator, test_generator, transfer_difficulty), np.array([]))
            summary = entropy_ood_summary(id_scores, ood_scores)
            ood_rows.append(
                {
                    "train_generator": train_generator,
                    "ood_condition": f"{test_generator}@{transfer_difficulty}",
                    **summary,
                }
            )
        if "hard" in difficulties and str(config["train_difficulty"]) != "hard":
            hard_scores = entropy_by_condition.get((train_generator, train_generator, "hard"), np.array([]))
            summary = entropy_ood_summary(id_scores, hard_scores)
            ood_rows.append(
                {
                    "train_generator": train_generator,
                    "ood_condition": f"{train_generator}@hard",
                    **summary,
                }
            )

    ood_df = pd.DataFrame(ood_rows)
    ood_df.to_csv(directories["tables"] / "ood_entropy_summary.csv", index=False)

    matched_df = evaluation_df[evaluation_df["train_generator"] == evaluation_df["test_generator"]].copy()
    plot_exact_vs_difficulty(matched_df, directories["plots"] / "exact_accuracy_vs_difficulty.png")
    plot_char_vs_difficulty(matched_df, directories["plots"] / "char_accuracy_vs_difficulty.png")
    plot_ece_vs_difficulty(matched_df, directories["plots"] / "calibration_error_vs_difficulty.png")
    plot_abstention_curves(abstention_df, directories["plots"] / "abstention_curves.png")
    plot_transfer_matrix(
        evaluation_df,
        transfer_difficulty,
        directories["plots"] / "transfer_matrix_exact_accuracy.png",
        directories["tables"] / "transfer_matrix_exact_accuracy.csv",
    )

    write_methods_addendum(config, evaluation_df, ood_df, directories["docs"] / "methods_addendum.md")
    write_summary_markdown(
        config,
        evaluation_df,
        temperature_df,
        ood_df,
        directories["docs"] / "local_extension_report.md",
    )

    bundle_path = bundle_outputs(
        output_root,
        output_root / "paper_bundle",
        reference_paper,
    )
    print(f"Experiment completed. Bundle written to: {bundle_path}")
    return bundle_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local-only CAPTCHA-X benchmark extension")
    parser.add_argument("--config", type=str, required=True, help="Path to the JSON experiment config")
    parser.add_argument(
        "--reference-paper",
        type=str,
        default="",
        help="Optional LaTeX paper path to copy into the final bundle for handoff context",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run_experiment(config, args.config, reference_paper=args.reference_paper or None)


if __name__ == "__main__":
    main()
