#!/usr/bin/env python3
"""Shared sequence metrics for CAPTCHA-X evaluation scripts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Sequence

import pandas as pd


def edit_distance(prediction: str, target: str) -> int:
    """Levenshtein edit distance."""
    prediction = str(prediction)
    target = str(target)
    if not prediction:
        return len(target)
    if not target:
        return len(prediction)

    previous = list(range(len(target) + 1))
    for i, pred_char in enumerate(prediction, start=1):
        current = [i]
        for j, target_char in enumerate(target, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + int(pred_char != target_char),
                )
            )
        previous = current
    return previous[-1]


def exact_accuracy(predictions: Sequence[str], targets: Sequence[str]) -> float:
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have equal length")
    return sum(str(pred) == str(tgt) for pred, tgt in zip(predictions, targets)) / max(len(targets), 1)


def character_accuracy(predictions: Sequence[str], targets: Sequence[str]) -> float:
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have equal length")
    edits = sum(edit_distance(str(pred), str(tgt)) for pred, tgt in zip(predictions, targets))
    total = sum(max(len(str(tgt)), 1) for tgt in targets)
    return max(0.0, 1.0 - edits / max(total, 1))


def summarize_predictions(predictions: Sequence[str], targets: Sequence[str]) -> Dict[str, float]:
    edit_distances = [edit_distance(str(pred), str(tgt)) for pred, tgt in zip(predictions, targets)]
    return {
        "exact_accuracy": float(exact_accuracy(predictions, targets)),
        "char_accuracy": float(character_accuracy(predictions, targets)),
        "avg_edit_distance": float(sum(edit_distances) / max(len(edit_distances), 1)),
        "num_samples": int(len(targets)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute CAPTCHA sequence metrics from a prediction CSV")
    parser.add_argument("--predictions-csv", required=True, help="CSV containing target and prediction columns")
    parser.add_argument("--target-col", default="target")
    parser.add_argument("--prediction-col", default="prediction")
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.predictions_csv).fillna("")
    summary = summarize_predictions(df[args.prediction_col].astype(str), df[args.target_col].astype(str))
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

