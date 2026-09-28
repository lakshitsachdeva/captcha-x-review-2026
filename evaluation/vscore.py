#!/usr/bin/env python3
"""Computable CAPTCHA-X reliability scorecard.

The reported paper score deliberately has three terms: exact accuracy,
calibration quality, and normalized prediction entropy. OOD AUROC is reported
as a separate diagnostic because it is not available for every matched row and
including an implicit missing-value penalty would make the score depend on
whether an auxiliary experiment was run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Mapping

import numpy as np
import pandas as pd


DEFAULT_WEIGHTS = {
    "accuracy": 0.4,
    "calibration": 0.2,
    "uncertainty": 0.2,
}

# These are the five configurations printed in the manuscript table. Every
# configuration is evaluated on all sixteen matched generator--difficulty rows.
SENSITIVITY_CONFIGS = (
    ("default", {"accuracy": 0.4, "calibration": 0.2, "uncertainty": 0.2}),
    ("accuracy_only", {"accuracy": 1.0, "calibration": 0.0, "uncertainty": 0.0}),
    ("calibration_heavy", {"accuracy": 0.0, "calibration": 0.8, "uncertainty": 0.2}),
    ("entropy_heavy", {"accuracy": 0.0, "calibration": 0.2, "uncertainty": 0.8}),
    ("equal_weight", {"accuracy": 1 / 3, "calibration": 1 / 3, "uncertainty": 1 / 3}),
)


def normalize_weights(weights: Mapping[str, float]) -> Dict[str, float]:
    merged = {**DEFAULT_WEIGHTS, **dict(weights)}
    cleaned = {key: max(float(value), 0.0) for key, value in merged.items()}
    if sum(cleaned.values()) <= 0:
        return dict(DEFAULT_WEIGHTS)
    # The paper deliberately leaves 0.2 mass outside the three reported terms;
    # the attainable upper bound is therefore 0.6 rather than 1.0.
    return cleaned


def compute_vscore_row(row: Mapping[str, object], weights: Mapping[str, float] | None = None) -> Dict[str, float]:
    """Return the three-term V-Score and its components.

    The score is a descriptive reliability summary, not a deployment
    vulnerability probability. OOD metrics are intentionally not included;
    see ``ood_entropy_summary.csv`` for that separate analysis.
    """
    w = normalize_weights(weights or DEFAULT_WEIGHTS)
    exact = float(row.get("exact_accuracy", row.get("accuracy", 0.0)))
    ece = float(row.get("ece", 0.0))
    uncertainty = float(row.get("avg_entropy", row.get("uncertainty", 0.0)))
    calibration_quality = 1.0 - min(max(ece, 0.0), 1.0)

    reliability = (
        w["accuracy"] * exact
        + w["calibration"] * calibration_quality
        - w["uncertainty"] * uncertainty
    )
    reliability = float(np.clip(reliability, 0.0, 1.0))
    return {
        "v_score": reliability,
        "captcha_robustness_score": float(1.0 - reliability),
        "accuracy_component": exact,
        "calibration_component": calibration_quality,
        "uncertainty_component": uncertainty,
    }


def add_vscore_columns(df: pd.DataFrame, weights: Mapping[str, float] | None = None) -> pd.DataFrame:
    records = []
    for row in df.to_dict(orient="records"):
        records.append({**row, **compute_vscore_row(row, weights=weights)})
    return pd.DataFrame(records)


def sensitivity_weights(step: float = 0.2) -> Iterable[Dict[str, float]]:
    """Yield the named configurations reported in the paper.

    ``step`` is retained for CLI compatibility; the paper uses the explicit
    configurations above rather than silently expanding the score with an OOD
    weight or an unreported simplex grid.
    """
    del step
    for _, weights in SENSITIVITY_CONFIGS:
        yield dict(weights)


def sensitivity_analysis(df: pd.DataFrame, step: float = 0.2) -> pd.DataFrame:
    rows = []
    id_columns = [column for column in ("model", "train_generator", "test_generator", "generator_name", "difficulty") if column in df.columns]
    for configuration, weight_set in SENSITIVITY_CONFIGS:
        scored = add_vscore_columns(df, weights=weight_set)
        for _, row in scored.iterrows():
            record = {column: row[column] for column in id_columns}
            record["configuration"] = configuration
            record.update({f"w_{key}": value for key, value in weight_set.items()})
            record["v_score"] = float(row["v_score"])
            record["captcha_robustness_score"] = float(row["captcha_robustness_score"])
            rows.append(record)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute CAPTCHA-X three-term V-Score table")
    parser.add_argument("--summary-csv", required=True, help="CSV with exact_accuracy, ece, and avg_entropy")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--sensitivity-csv", required=True)
    parser.add_argument("--weights-json", default="", help="Optional JSON object overriding default weights")
    parser.add_argument("--sensitivity-step", type=float, default=0.2, help="Retained for CLI compatibility; named configurations are used")
    args = parser.parse_args()

    weights = DEFAULT_WEIGHTS
    if args.weights_json:
        weights = normalize_weights(json.loads(args.weights_json))

    df = pd.read_csv(args.summary_csv).fillna(0)
    scored = add_vscore_columns(df, weights=weights)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_csv, index=False)
    sensitivity = sensitivity_analysis(df, step=args.sensitivity_step)
    Path(args.sensitivity_csv).parent.mkdir(parents=True, exist_ok=True)
    sensitivity.to_csv(args.sensitivity_csv, index=False)
    print(scored.to_string(index=False))


if __name__ == "__main__":
    main()
