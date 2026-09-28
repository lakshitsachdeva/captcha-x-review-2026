#!/usr/bin/env python3
"""Failure taxonomy for incorrect CAPTCHA predictions."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


ERROR_TYPES = ("CORRECT", "SUBSTITUTION", "DELETION", "INSERTION", "COLLAPSE", "TRANSPOSITION")


def _edit_counts(prediction: str, target: str) -> Dict[str, int]:
    """Backtrace Levenshtein alignment into operation counts."""
    prediction = str(prediction)
    target = str(target)
    rows = len(prediction) + 1
    cols = len(target) + 1
    dp = [[0] * cols for _ in range(rows)]
    op = [[""] * cols for _ in range(rows)]

    for i in range(1, rows):
        dp[i][0] = i
        op[i][0] = "INSERTION"
    for j in range(1, cols):
        dp[0][j] = j
        op[0][j] = "DELETION"

    for i in range(1, rows):
        for j in range(1, cols):
            substitution_cost = int(prediction[i - 1] != target[j - 1])
            choices: List[Tuple[int, str]] = [
                (dp[i - 1][j] + 1, "INSERTION"),
                (dp[i][j - 1] + 1, "DELETION"),
                (dp[i - 1][j - 1] + substitution_cost, "MATCH" if substitution_cost == 0 else "SUBSTITUTION"),
            ]
            dp[i][j], op[i][j] = min(choices, key=lambda item: item[0])

    counts = Counter({"SUBSTITUTION": 0, "DELETION": 0, "INSERTION": 0})
    i, j = len(prediction), len(target)
    while i > 0 or j > 0:
        current = op[i][j]
        if current == "MATCH":
            i -= 1
            j -= 1
        elif current == "SUBSTITUTION":
            counts["SUBSTITUTION"] += 1
            i -= 1
            j -= 1
        elif current == "INSERTION":
            counts["INSERTION"] += 1
            i -= 1
        elif current == "DELETION":
            counts["DELETION"] += 1
            j -= 1
        else:
            break
    return dict(counts)


def classify_failure(prediction: str, target: str) -> Dict[str, object]:
    prediction = str(prediction)
    target = str(target)
    if prediction == target:
        return {"error_type": "CORRECT", "edit_counts": {"SUBSTITUTION": 0, "DELETION": 0, "INSERTION": 0}}

    if prediction and len(set(prediction)) == 1 and len(prediction) >= max(3, len(target) - 1):
        primary = "COLLAPSE"
    elif len(prediction) == len(target) and sorted(prediction) == sorted(target):
        primary = "TRANSPOSITION"
    elif len(prediction) < len(target):
        primary = "DELETION"
    elif len(prediction) > len(target):
        primary = "INSERTION"
    else:
        primary = "SUBSTITUTION"

    return {"error_type": primary, "edit_counts": _edit_counts(prediction, target)}


def build_taxonomy(df: pd.DataFrame, target_col: str, prediction_col: str) -> Tuple[pd.DataFrame, Dict[str, object]]:
    rows = []
    group_counts: Dict[str, Counter[str]] = defaultdict(Counter)

    for index, row in df.iterrows():
        result = classify_failure(str(row[prediction_col]), str(row[target_col]))
        detail = row.to_dict()
        detail["row_index"] = int(index)
        detail["error_type"] = result["error_type"]
        for edit_name, count in result["edit_counts"].items():
            detail[f"{edit_name.lower()}_count"] = int(count)
        rows.append(detail)

        generator = str(row.get("generator_name", row.get("test_generator", "unknown")))
        difficulty = str(row.get("difficulty", "unknown"))
        group_counts[f"{generator}::{difficulty}"][str(result["error_type"])] += 1

    details = pd.DataFrame(rows)
    summary: Dict[str, object] = {
        "overall_counts": details["error_type"].value_counts().reindex(ERROR_TYPES, fill_value=0).to_dict(),
        "by_generator_difficulty": {},
    }
    for group, counts in group_counts.items():
        total = sum(counts.values())
        summary["by_generator_difficulty"][group] = {
            error_type: {
                "count": int(counts.get(error_type, 0)),
                "proportion": float(counts.get(error_type, 0) / max(total, 1)),
            }
            for error_type in ERROR_TYPES
        }
    return details, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify CAPTCHA prediction failures")
    parser.add_argument("--predictions-csv", required=True)
    parser.add_argument("--target-col", default="target")
    parser.add_argument("--prediction-col", default="prediction")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.predictions_csv).fillna("")
    details, summary = build_taxonomy(df, args.target_col, args.prediction_col)
    details.to_csv(output_dir / "failure_taxonomy_details.csv", index=False)
    (output_dir / "failure_taxonomy.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["overall_counts"], indent=2))


if __name__ == "__main__":
    main()

