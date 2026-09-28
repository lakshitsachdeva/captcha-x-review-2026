#!/usr/bin/env python3
"""
Evaluate trained text CAPTCHA models and export paper-ready artifacts.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from .model import (
        CaptchaCNN,
        calculate_edit_distance,
        calculate_prediction_confidences,
        decode_predictions,
        decode_targets,
        summarize_sequence_metrics,
    )
    from .preprocess import CaptchaDataset, CaptchaPreprocessor, collate_fn, infer_max_length
except ImportError:
    from model import (
        CaptchaCNN,
        calculate_edit_distance,
        calculate_prediction_confidences,
        decode_predictions,
        decode_targets,
        summarize_sequence_metrics,
    )
    from preprocess import CaptchaDataset, CaptchaPreprocessor, collate_fn, infer_max_length


class CaptchaEvaluator:
    """Run evaluation and export analysis artifacts."""

    def __init__(self, model: CaptchaCNN, device: torch.device, idx_to_char: Dict[int, str]):
        self.model = model
        self.device = device
        self.idx_to_char = idx_to_char
        self.model.eval()

    def evaluate_dataset(
        self,
        dataloader: DataLoader,
        metadata_records: List[Dict[str, str]],
        dataset_name: str = "test",
    ) -> Tuple[Dict[str, float], pd.DataFrame]:
        print(f"Evaluating {dataset_name} split...")

        all_predictions: List[str] = []
        all_targets: List[str] = []
        all_confidences: List[float] = []
        all_inference_times: List[float] = []

        with torch.no_grad():
            progress_bar = tqdm(dataloader, desc=f"Evaluating {dataset_name}")
            for images, texts, lengths in progress_bar:
                if images.size(0) == 0 or torch.all(lengths == 0):
                    continue

                images = images.to(self.device)
                texts = texts.to(self.device)
                lengths = lengths.to(self.device)

                start_time = time.perf_counter()
                outputs = self.model(images)
                inference_seconds = (time.perf_counter() - start_time) / images.size(0)

                input_lengths = torch.full(
                    (outputs.size(0),),
                    outputs.size(1),
                    dtype=torch.long,
                    device=self.device,
                )
                predictions = decode_predictions(outputs, self.idx_to_char, input_lengths)
                targets = decode_targets(texts.detach().cpu(), lengths.detach().cpu(), self.idx_to_char)
                confidences = calculate_prediction_confidences(outputs, input_lengths)

                all_predictions.extend(predictions)
                all_targets.extend(targets)
                all_confidences.extend(confidences)
                all_inference_times.extend([inference_seconds] * len(predictions))

        metrics = summarize_sequence_metrics(all_predictions, all_targets)
        metrics["avg_confidence"] = float(np.mean(all_confidences)) if all_confidences else 0.0
        metrics["std_confidence"] = float(np.std(all_confidences)) if all_confidences else 0.0
        metrics["avg_inference_time"] = float(np.mean(all_inference_times)) if all_inference_times else 0.0
        metrics["std_inference_time"] = float(np.std(all_inference_times)) if all_inference_times else 0.0
        metrics["num_samples"] = len(all_predictions)

        detailed_rows = []
        for index, (prediction, target, confidence, metadata) in enumerate(
            zip(all_predictions, all_targets, all_confidences, metadata_records)
        ):
            edit_distance = calculate_edit_distance(prediction, target)
            normalized_edit_distance = edit_distance / max(len(target), 1)
            row = dict(metadata)
            row.update(
                {
                    "row_index": index,
                    "prediction": prediction,
                    "target": target,
                    "confidence": confidence,
                    "correct": prediction == target,
                    "edit_distance": edit_distance,
                    "normalized_edit_distance": normalized_edit_distance,
                }
            )
            detailed_rows.append(row)

        detailed_results = pd.DataFrame(detailed_rows)

        print(f"\n{dataset_name.capitalize()} metrics")
        print(f"  Exact accuracy:           {metrics['accuracy']:.4f}")
        print(f"  Character accuracy:       {metrics['char_accuracy']:.4f}")
        print(f"  Average edit distance:    {metrics['avg_edit_distance']:.4f}")
        print(f"  Normalized edit distance: {metrics['normalized_edit_distance']:.4f}")
        print(f"  Average confidence:       {metrics['avg_confidence']:.4f}")
        print(f"  Avg inference time:       {metrics['avg_inference_time']:.4f}s")

        return metrics, detailed_results

    def generate_error_analysis(self, detailed_results: pd.DataFrame, save_dir: Path) -> None:
        """Create CSV, figures, and a markdown summary."""
        detailed_results.to_csv(save_dir / "detailed_results.csv", index=False)
        detailed_results[~detailed_results["correct"]].to_csv(save_dir / "error_analysis.csv", index=False)

        if detailed_results.empty:
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        conf_bins = np.linspace(0, 1, 11)
        conf_centers = (conf_bins[:-1] + conf_bins[1:]) / 2
        binned_accuracy = []
        for lower, upper in zip(conf_bins[:-1], conf_bins[1:]):
            bin_df = detailed_results[
                (detailed_results["confidence"] >= lower) & (detailed_results["confidence"] < upper)
            ]
            binned_accuracy.append(float(bin_df["correct"].mean()) if not bin_df.empty else 0.0)
        axes[0, 0].bar(conf_centers, binned_accuracy, width=0.08, color="#2E86AB")
        axes[0, 0].set_title("Accuracy by Confidence")
        axes[0, 0].set_xlabel("Confidence bin")
        axes[0, 0].set_ylabel("Accuracy")
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].hist(detailed_results["edit_distance"], bins=15, color="#F18F01", alpha=0.8)
        axes[0, 1].set_title("Edit Distance Distribution")
        axes[0, 1].set_xlabel("Edit distance")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].grid(True, alpha=0.3)

        length_accuracy = (
            detailed_results.groupby("length")["correct"].mean().sort_index()
            if "length" in detailed_results.columns
            else pd.Series(dtype=float)
        )
        if not length_accuracy.empty:
            axes[0, 2].bar(length_accuracy.index.astype(str), length_accuracy.values, color="#6A994E")
            axes[0, 2].set_title("Accuracy by CAPTCHA Length")
            axes[0, 2].set_xlabel("Length")
            axes[0, 2].set_ylabel("Accuracy")
            axes[0, 2].grid(True, alpha=0.3)
        else:
            axes[0, 2].axis("off")

        if "difficulty" in detailed_results.columns:
            difficulty_accuracy = detailed_results.groupby("difficulty")["correct"].mean()
            axes[1, 0].bar(difficulty_accuracy.index, difficulty_accuracy.values, color="#BC4749")
            axes[1, 0].set_title("Accuracy by Difficulty")
            axes[1, 0].set_ylabel("Accuracy")
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].axis("off")

        axes[1, 1].hist(detailed_results["confidence"], bins=20, color="#7B2CBF", alpha=0.8)
        axes[1, 1].set_title("Confidence Distribution")
        axes[1, 1].set_xlabel("Confidence")
        axes[1, 1].set_ylabel("Frequency")
        axes[1, 1].grid(True, alpha=0.3)

        error_types = []
        for _, row in detailed_results.iterrows():
            if row["correct"]:
                error_types.append("Correct")
            elif len(str(row["prediction"])) != len(str(row["target"])):
                error_types.append("Length mismatch")
            elif not str(row["prediction"]):
                error_types.append("Empty prediction")
            else:
                error_types.append("Character substitution")

        error_counts = pd.Series(error_types).value_counts()
        axes[1, 2].pie(error_counts.values, labels=error_counts.index, autopct="%1.1f%%")
        axes[1, 2].set_title("Error Type Distribution")

        plt.tight_layout()
        plt.savefig(save_dir / "error_analysis.png", dpi=250, bbox_inches="tight")
        plt.close(fig)

        incorrect = detailed_results[~detailed_results["correct"]].copy()
        if not incorrect.empty:
            hardest = incorrect.sort_values(
                by=["normalized_edit_distance", "confidence"],
                ascending=[False, True],
            ).head(20)
        else:
            hardest = detailed_results.head(5)

        hardest.to_csv(save_dir / "hard_examples.csv", index=False)

    def write_markdown_report(
        self,
        metrics: Dict[str, float],
        detailed_results: pd.DataFrame,
        output_dir: Path,
        dataset_name: str,
        model_path: str,
    ) -> None:
        """Create a paper-friendly narrative summary."""
        length_section = ""
        if "length" in detailed_results.columns and not detailed_results.empty:
            length_accuracy = detailed_results.groupby("length")["correct"].mean().sort_index()
            length_lines = "\n".join(
                f"| {length} | {accuracy:.4f} |"
                for length, accuracy in length_accuracy.items()
            )
            length_section = (
                "## Accuracy By Length\n\n"
                "| CAPTCHA length | Exact accuracy |\n"
                "| --- | --- |\n"
                f"{length_lines}\n\n"
            )

        hardest = detailed_results[~detailed_results["correct"]].sort_values(
            by=["normalized_edit_distance", "confidence"],
            ascending=[False, True],
        ).head(10)
        hardest_lines = "\n".join(
            f"| {row.filename} | {row.target} | {row.prediction} | {row.confidence:.3f} | {row.normalized_edit_distance:.3f} |"
            for row in hardest.itertuples()
        )
        if not hardest_lines:
            hardest_lines = "| None | - | - | - | - |"

        report = f"""# CAPTCHA-X Evaluation Report

## Experiment Summary

- Split: `{dataset_name}`
- Samples: `{int(metrics['num_samples'])}`
- Model checkpoint: `{model_path}`

## Core Metrics

| Metric | Value |
| --- | --- |
| Exact accuracy | {metrics['accuracy']:.4f} |
| Character accuracy | {metrics['char_accuracy']:.4f} |
| Average edit distance | {metrics['avg_edit_distance']:.4f} |
| Normalized edit distance | {metrics['normalized_edit_distance']:.4f} |
| Average confidence | {metrics['avg_confidence']:.4f} |
| Average inference time (s) | {metrics['avg_inference_time']:.4f} |

{length_section}## Hardest Examples

| Filename | Target | Prediction | Confidence | Normalized edit distance |
| --- | --- | --- | --- | --- |
{hardest_lines}

## Interpretation Notes

- Exact accuracy is the strict metric for end-to-end CAPTCHA solving.
- Character accuracy and normalized edit distance are better early-training indicators and better paper metrics for ablations.
- Confidence should be interpreted as a decoding heuristic, not a calibrated probability.
"""

        with open(output_dir / "paper_report.md", "w") as handle:
            handle.write(report)


def load_model(checkpoint_path: str, device: torch.device) -> Tuple[CaptchaCNN, Dict[str, int], Dict[int, str], Dict[str, object]]:
    """Load a model checkpoint with its saved architecture config."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    char_to_idx = checkpoint["char_to_idx"]
    idx_to_char = checkpoint["idx_to_char"]
    model_config = checkpoint.get("model_config", {"num_classes": len(char_to_idx)})

    model = CaptchaCNN(**model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, char_to_idx, idx_to_char, checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained CAPTCHA model")
    parser.add_argument("--model-path", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--data-dir", type=str, required=True, help="Dataset directory")
    parser.add_argument("--metadata-file", type=str, required=True, help="Metadata CSV path")
    parser.add_argument("--output-dir", type=str, default="evaluation_results", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, or cuda")
    parser.add_argument("--split", type=str, default="test", help="Split to evaluate")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--max-length", type=int, default=0, help="Override max text length")
    parser.add_argument("--image-width", type=int, default=200, help="Input width")
    parser.add_argument("--image-height", type=int, default=80, help="Input height")

    args = parser.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, char_to_idx, idx_to_char, checkpoint = load_model(args.model_path, device)
    training_config = checkpoint.get("training_config", {})

    metadata = pd.read_csv(args.metadata_file)
    max_length = args.max_length or int(training_config.get("max_length", infer_max_length(metadata["text"].astype(str))))
    image_width = int(training_config.get("image_width", args.image_width))
    image_height = int(training_config.get("image_height", args.image_height))

    dataset = CaptchaDataset(
        data_dir=args.data_dir,
        metadata_file=args.metadata_file,
        preprocessor=CaptchaPreprocessor(target_size=(image_width, image_height), normalize=True, augment=False),
        char_to_idx=char_to_idx,
        max_length=max_length,
        split=args.split,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )

    evaluator = CaptchaEvaluator(model, device, idx_to_char)
    metrics, detailed_results = evaluator.evaluate_dataset(
        dataloader,
        metadata_records=dataset.get_metadata_records(),
        dataset_name=args.split,
    )

    evaluator.generate_error_analysis(detailed_results, output_dir)
    evaluator.write_markdown_report(metrics, detailed_results, output_dir, args.split, args.model_path)

    summary = {
        "dataset_split": args.split,
        "dataset_size": int(metrics["num_samples"]),
        "model_path": args.model_path,
        "metrics": metrics,
        "training_config": training_config,
    }

    with open(output_dir / "evaluation_results.json", "w") as handle:
        json.dump(summary, handle, indent=2, default=str)

    print(f"\nEvaluation artifacts written to {output_dir}")
    print(f"Report: {output_dir / 'paper_report.md'}")


if __name__ == "__main__":
    main()
