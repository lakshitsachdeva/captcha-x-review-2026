#!/usr/bin/env python3
"""
Helpers for scoring small local human-baseline pilot studies.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def score_human_baseline(manifest_csv: str, responses_csv: str, output_dir: str) -> Path:
    manifest = pd.read_csv(manifest_csv).fillna("")
    responses = pd.read_csv(responses_csv).fillna("")

    merged = manifest.merge(responses, on="stimulus_id", how="left", suffixes=("_manifest", "_response"))
    merged["response_text"] = merged["response_text"].astype(str)
    merged["ground_truth"] = merged["ground_truth"].astype(str)
    merged["correct"] = merged["response_text"] == merged["ground_truth"]

    solve_time = pd.to_numeric(merged["solve_time_seconds"], errors="coerce")
    difficulty_rating = pd.to_numeric(merged["difficulty_rating_1_to_5"], errors="coerce")

    summary = (
        merged.assign(
            solve_time_seconds=solve_time,
            difficulty_rating_1_to_5=difficulty_rating,
        )
        .groupby(["generator_name", "difficulty"], dropna=False)
        .agg(
            solve_rate=("correct", "mean"),
            median_solve_time_seconds=("solve_time_seconds", "median"),
            mean_difficulty_rating=("difficulty_rating_1_to_5", "mean"),
            num_trials=("stimulus_id", "count"),
        )
        .reset_index()
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path / "human_baseline_scored_trials.csv", index=False)
    summary.to_csv(output_path / "human_baseline_summary.csv", index=False)
    return output_path / "human_baseline_summary.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a local human-baseline CAPTCHA pilot study")
    parser.add_argument("--manifest-csv", type=str, required=True, help="Stimulus manifest CSV")
    parser.add_argument("--responses-csv", type=str, required=True, help="Participant response CSV")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for scored outputs")
    args = parser.parse_args()

    summary_path = score_human_baseline(args.manifest_csv, args.responses_csv, args.output_dir)
    print(f"Human-baseline summary written to: {summary_path}")


if __name__ == "__main__":
    main()
