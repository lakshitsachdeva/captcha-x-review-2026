#!/usr/bin/env python3
"""Abstention policies and coverage/selective-accuracy curves."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd


def area_under_curve(x_values: Sequence[float], y_values: Sequence[float]) -> float:
    if len(x_values) < 2:
        return 0.0
    order = np.argsort(np.asarray(x_values, dtype=np.float64))
    x = np.asarray(x_values, dtype=np.float64)[order]
    y = np.asarray(y_values, dtype=np.float64)[order]
    return float(np.trapezoid(y, x))


def confidence_policy_scores(df: pd.DataFrame) -> np.ndarray:
    return pd.to_numeric(df["confidence"], errors="coerce").fillna(0.0).clip(0.0, 1.0).to_numpy()


def entropy_policy_scores(df: pd.DataFrame) -> np.ndarray:
    entropy = pd.to_numeric(df["entropy"], errors="coerce").fillna(1.0).to_numpy()
    return 1.0 - np.clip(entropy, 0.0, 1.0)


def ensemble_disagreement_scores(df: pd.DataFrame) -> np.ndarray:
    if "ensemble_disagreement" in df.columns:
        disagreement = pd.to_numeric(df["ensemble_disagreement"], errors="coerce").fillna(1.0).to_numpy()
        return 1.0 - np.clip(disagreement, 0.0, 1.0)
    return confidence_policy_scores(df)


def policy_curve(
    scores: Sequence[float],
    correctness: Sequence[bool | int],
    thresholds: Iterable[float],
) -> List[Dict[str, float]]:
    score_arr = np.asarray(scores, dtype=np.float64)
    correct = np.asarray(correctness, dtype=np.float64)
    rows: List[Dict[str, float]] = []
    for threshold in thresholds:
        keep = score_arr >= float(threshold)
        coverage = float(keep.mean()) if keep.size else 0.0
        selective_accuracy = float(correct[keep].mean()) if keep.any() else 0.0
        rows.append(
            {
                "threshold": float(threshold),
                "coverage": coverage,
                "selective_accuracy": selective_accuracy,
                "selected_samples": int(keep.sum()),
                "total_samples": int(keep.size),
            }
        )
    acac = area_under_curve([row["coverage"] for row in rows], [row["selective_accuracy"] for row in rows])
    for row in rows:
        row["acac"] = acac
    return rows


def build_abstention_report(df: pd.DataFrame, thresholds: Iterable[float]) -> pd.DataFrame:
    if "exact_correct" in df.columns:
        correct = df["exact_correct"].astype(str).str.lower().isin({"1", "true", "yes"}).to_numpy()
    else:
        correct = (df["target"].astype(str) == df["prediction"].astype(str)).to_numpy()

    policies = {
        "max_softmax_threshold": confidence_policy_scores(df),
        "string_entropy_threshold": entropy_policy_scores(df),
        "ensemble_disagreement": ensemble_disagreement_scores(df),
    }
    rows = []
    for policy_name, scores in policies.items():
        for row in policy_curve(scores, correct, thresholds):
            rows.append({"policy": policy_name, **row})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute abstention-policy curves")
    parser.add_argument("--predictions-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--thresholds", default="0,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99")
    args = parser.parse_args()

    df = pd.read_csv(args.predictions_csv).fillna("")
    thresholds = [float(value) for value in args.thresholds.split(",") if value.strip()]
    report = build_abstention_report(df, thresholds)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False)
    print(report.groupby("policy")["acac"].first().to_string())


if __name__ == "__main__":
    main()
