#!/usr/bin/env python3
"""OOD detection suite for CAPTCHA-X reliability analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def fpr_at_tpr95(labels: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    valid = np.where(tpr >= 0.95)[0]
    if valid.size == 0:
        return 1.0
    return float(fpr[valid[0]])


def evaluate_scores(id_scores: Sequence[float], ood_scores: Sequence[float]) -> Dict[str, float]:
    id_arr = np.asarray(id_scores, dtype=np.float64)
    ood_arr = np.asarray(ood_scores, dtype=np.float64)
    if id_arr.size == 0 or ood_arr.size == 0:
        return {"auroc": 0.0, "aupr": 0.0, "fpr_at_95_tpr": 1.0}
    labels = np.concatenate([np.zeros_like(id_arr), np.ones_like(ood_arr)])
    scores = np.concatenate([id_arr, ood_arr])
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "aupr": float(average_precision_score(labels, scores)),
        "fpr_at_95_tpr": fpr_at_tpr95(labels, scores),
    }


def score_columns_from_details(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Build OOD scores where larger means more OOD-like."""
    scores: Dict[str, np.ndarray] = {}
    if "confidence" in df.columns:
        confidence = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.0).clip(0.0, 1.0).to_numpy()
        scores["msp"] = 1.0 - confidence
        scores["temperature_msp"] = 1.0 - confidence
    if "entropy" in df.columns:
        entropy = pd.to_numeric(df["entropy"], errors="coerce").fillna(0.0).to_numpy()
        scores["odin"] = entropy
        scores["energy"] = entropy
    if "energy" in df.columns:
        scores["energy"] = pd.to_numeric(df["energy"], errors="coerce").fillna(0.0).to_numpy()
    return scores


def evaluate_ood_suite(id_df: pd.DataFrame, ood_df: pd.DataFrame) -> pd.DataFrame:
    id_scores = score_columns_from_details(id_df)
    ood_scores = score_columns_from_details(ood_df)
    methods = sorted(set(id_scores) & set(ood_scores))
    rows = []
    for method in methods:
        rows.append({"method": method, **evaluate_scores(id_scores[method], ood_scores[method])})
    return pd.DataFrame(rows)


def evaluate_many_conditions(id_csv: str, ood_csvs: Mapping[str, str]) -> pd.DataFrame:
    id_df = pd.read_csv(id_csv).fillna(0)
    frames = []
    for condition, csv_path in ood_csvs.items():
        result = evaluate_ood_suite(id_df, pd.read_csv(csv_path).fillna(0))
        result.insert(0, "ood_condition", condition)
        frames.append(result)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MSP, temperature-MSP, ODIN-proxy, and energy OOD scoring")
    parser.add_argument("--id-csv", required=True, help="In-distribution prediction details CSV")
    parser.add_argument(
        "--ood-csv",
        action="append",
        required=True,
        help="OOD CSV entry as name:path. May be repeated.",
    )
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    ood_map = {}
    for entry in args.ood_csv:
        if ":" not in entry:
            raise ValueError("--ood-csv entries must be name:path")
        name, path = entry.split(":", 1)
        ood_map[name] = path

    result = evaluate_many_conditions(args.id_csv, ood_map)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    payload = result.to_dict(orient="records")
    Path(args.output_json).write_text(json.dumps(payload, indent=2) + "\n")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()

