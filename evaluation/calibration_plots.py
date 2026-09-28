#!/usr/bin/env python3
"""Calibration metrics and reliability diagrams for CAPTCHA-X."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def calibration_bins(confidences: Sequence[float], correctness: Sequence[bool | int], n_bins: int = 15) -> pd.DataFrame:
    conf = np.asarray(confidences, dtype=np.float64)
    corr = np.asarray(correctness, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (conf >= lower) & (conf <= upper if upper == 1.0 else conf < upper)
        count = int(mask.sum())
        rows.append(
            {
                "bin_lower": float(lower),
                "bin_upper": float(upper),
                "bin_center": float((lower + upper) / 2.0),
                "count": count,
                "accuracy": float(corr[mask].mean()) if count else 0.0,
                "confidence": float(conf[mask].mean()) if count else 0.0,
            }
        )
    return pd.DataFrame(rows)


def calibration_metrics(confidences: Sequence[float], correctness: Sequence[bool | int], n_bins: int = 15) -> Dict[str, float]:
    bins = calibration_bins(confidences, correctness, n_bins=n_bins)
    total = max(int(bins["count"].sum()), 1)
    gaps = (bins["accuracy"] - bins["confidence"]).abs()
    weights = bins["count"] / total
    non_empty = bins[bins["count"] > 0]
    adaptive_gap = 0.0
    if not non_empty.empty:
        adaptive_gap = float((non_empty["accuracy"] - non_empty["confidence"]).abs().mean())
    return {
        "ece": float((weights * gaps).sum()),
        "mce": float(gaps.max()) if len(gaps) else 0.0,
        "ace": adaptive_gap,
    }


def plot_reliability_diagram(
    bins: pd.DataFrame,
    output_png: Path,
    output_svg: Path | None = None,
    *,
    title: str = "Reliability Diagram",
) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    if output_svg is not None:
        output_svg.parent.mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(figsize=(6.5, 5.2))
    width = float((bins["bin_upper"] - bins["bin_lower"]).median()) if not bins.empty else 0.05
    gaps = bins["accuracy"] - bins["confidence"]
    colors = ["#0072B2" if gap >= 0 else "#D55E00" for gap in gaps]
    axis.bar(bins["bin_center"], bins["accuracy"], width=width * 0.92, color=colors, alpha=0.78, label="Empirical accuracy")
    axis.plot([0, 1], [0, 1], color="#111827", linestyle="--", linewidth=1.2, label="Perfect calibration")
    axis.fill_between([0, 1], [0, 1], [1, 1], color="#0072B2", alpha=0.08, label="Underconfident")
    axis.fill_between([0, 1], [0, 0], [0, 1], color="#D55E00", alpha=0.08, label="Overconfident")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xlabel("Confidence")
    axis.set_ylabel("Accuracy")
    axis.set_title(title)
    axis.grid(True, alpha=0.25)
    axis.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    if output_svg is not None:
        fig.savefig(output_svg, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build reliability diagram and calibration metrics")
    parser.add_argument("--predictions-csv", required=True)
    parser.add_argument("--confidence-col", default="confidence")
    parser.add_argument("--correct-col", default="exact_correct")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bins", type=int, default=15)
    parser.add_argument("--title", default="CAPTCHA-X Reliability Diagram")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.predictions_csv).fillna(0)
    confidences = pd.to_numeric(df[args.confidence_col], errors="coerce").fillna(0).clip(0, 1)
    correctness = df[args.correct_col].astype(str).str.lower().isin({"1", "true", "yes"})
    bins = calibration_bins(confidences, correctness, n_bins=args.bins)
    metrics = calibration_metrics(confidences, correctness, n_bins=args.bins)
    bins.to_csv(output_dir / "calibration_bins.csv", index=False)
    (output_dir / "calibration_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    plot_reliability_diagram(bins, output_dir / "reliability_diagram.png", output_dir / "svg" / "reliability_diagram.svg", title=args.title)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
