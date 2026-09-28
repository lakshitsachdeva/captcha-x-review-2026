#!/usr/bin/env python3
"""Aggregate seed-sweep results into mean/std tables and curves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


METRICS = ("exact_accuracy", "char_accuracy", "ece", "ood_auroc")


def aggregate(seed_sweep_json: str, output_dir: str) -> Dict[str, object]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    payload = json.loads(Path(seed_sweep_json).read_text())
    df = pd.DataFrame(payload["results"])
    df.to_csv(output_path / "per_seed_results.csv", index=False)

    rows: List[Dict[str, object]] = []
    grouped = df.groupby("model_family", dropna=False)
    for model_family, group in grouped:
        row: Dict[str, object] = {"model_family": model_family, "num_seeds": int(len(group))}
        for metric in METRICS:
            if metric in group.columns and group[metric].notna().any():
                row[f"{metric}_mean"] = float(group[metric].mean())
                row[f"{metric}_std"] = float(group[metric].std(ddof=0))
            else:
                row[f"{metric}_mean"] = None
                row[f"{metric}_std"] = None
        rows.append(row)
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(output_path / "seed_sweep_summary.csv", index=False)
    plot_seed_bars(df, output_path / "seed_accuracy_bars.png")
    result = {"summary": summary_df.to_dict(orient="records")}
    (output_path / "seed_sweep_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def plot_seed_bars(df: pd.DataFrame, output_path: Path) -> None:
    if df.empty or "exact_accuracy" not in df:
        return
    fig, axis = plt.subplots(figsize=(8, 4.5))
    labels = [f"{row.model_family}\nseed {row.seed}" for row in df.itertuples()]
    axis.bar(labels, df["exact_accuracy"], color="#0072B2", alpha=0.82)
    axis.set_ylim(0, 1.0)
    axis.set_ylabel("Exact accuracy")
    axis.set_title("Per-seed exact accuracy")
    axis.grid(True, axis="y", alpha=0.25)
    axis.tick_params(axis="x", labelrotation=35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate CAPTCHA-X seed-sweep results")
    parser.add_argument("--seed-sweep-json", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(aggregate(args.seed_sweep_json, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
