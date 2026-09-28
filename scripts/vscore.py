#!/usr/bin/env python3
"""Computable CAPTCHA-X vulnerability/reliability scorecard."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Dict, Iterable, Mapping

import numpy as np
import pandas as pd


DEFAULT_WEIGHTS = {
    "accuracy": 0.4,
    "calibration": 0.2,
    "uncertainty": 0.2,
    "ood_exposure": 0.2,
}


def normalize_weights(weights: Mapping[str, float]) -> Dict[str, float]:
    merged = {**DEFAULT_WEIGHTS, **dict(weights)}
    total = sum(max(float(value), 0.0) for value in merged.values())
    if total <= 0:
        return dict(DEFAULT_WEIGHTS)
    return {key: max(float(value), 0.0) / total for key, value in merged.items()}


def compute_vscore_row(row: Mapping[str, object], weights: Mapping[str, float] | None = None) -> Dict[str, float]:
    """
    Compute the roadmap score R.

    High R means the solver evidence is strong and reliable on that condition:
    high solve rate, good calibration, low uncertainty, and low OOD penalty.
    For CAPTCHA-design robustness, use `captcha_robustness_score = 1 - R`.
    """
    w = normalize_weights(weights or DEFAULT_WEIGHTS)
    exact = float(row.get("exact_accuracy", row.get("accuracy", 0.0)))
    ece = float(row.get("ece", 0.0))
    uncertainty = float(row.get("avg_entropy", row.get("uncertainty", 0.0)))
    if "ood_auroc" in row:
        ood_exposure = 1.0 - float(row.get("ood_auroc", 0.0))
    else:
        ood_exposure = float(row.get("ood_exposure", 0.0))
    calibration_quality = 1.0 - min(max(ece, 0.0), 1.0)

    reliability = (
        w["accuracy"] * exact
        + w["calibration"] * calibration_quality
        - w["uncertainty"] * uncertainty
        - w["ood_exposure"] * ood_exposure
    )
    reliability = float(np.clip(reliability, 0.0, 1.0))
    vulnerability = reliability
    robustness = float(1.0 - vulnerability)
    return {
        "v_score": vulnerability,
        "captcha_robustness_score": robustness,
        "accuracy_component": exact,
        "calibration_component": calibration_quality,
        "uncertainty_component": uncertainty,
        "ood_exposure_component": ood_exposure,
    }


def add_vscore_columns(df: pd.DataFrame, weights: Mapping[str, float] | None = None) -> pd.DataFrame:
    records = []
    for row in df.to_dict(orient="records"):
        records.append({**row, **compute_vscore_row(row, weights=weights)})
    return pd.DataFrame(records)


def sensitivity_weights(step: float = 0.2) -> Iterable[Dict[str, float]]:
    values = np.arange(0.0, 1.0 + 1e-9, step)
    keys = list(DEFAULT_WEIGHTS)
    for combo in itertools.product(values, repeat=len(keys)):
        if abs(sum(combo) - 1.0) <= 1e-9:
            yield dict(zip(keys, combo))


def sensitivity_analysis(df: pd.DataFrame, step: float = 0.2) -> pd.DataFrame:
    rows = []
    id_columns = [column for column in ("model", "train_generator", "test_generator", "generator_name", "difficulty") if column in df.columns]
    for weight_set in sensitivity_weights(step=step):
        scored = add_vscore_columns(df, weights=weight_set)
        for _, row in scored.iterrows():
            record = {column: row[column] for column in id_columns}
            record.update({f"w_{key}": value for key, value in weight_set.items()})
            record["v_score"] = float(row["v_score"])
            record["captcha_robustness_score"] = float(row["captcha_robustness_score"])
            rows.append(record)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute CAPTCHA-X V-Score table")
    parser.add_argument("--summary-csv", required=True, help="CSV with exact_accuracy, ece, avg_entropy, and optional ood_auroc")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--sensitivity-csv", required=True)
    parser.add_argument("--weights-json", default="", help="Optional JSON object overriding default weights")
    parser.add_argument("--sensitivity-step", type=float, default=0.2)
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
    sensitivity.to_csv(args.sensitivity_csv, index=False)
    print(scored.to_string(index=False))


if __name__ == "__main__":
    main()
