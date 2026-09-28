#!/usr/bin/env python3
"""Statistical tests and confidence intervals for CAPTCHA-X tables."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.stats import binomtest, chi2
except ImportError:  # pragma: no cover - scipy is listed as a dependency.
    binomtest = None
    chi2 = None


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if total <= 0:
        return 0.0, 0.0
    z = 1.959963984540054 if abs(confidence - 0.95) < 1e-9 else 1.959963984540054
    phat = successes / total
    denominator = 1.0 + z * z / total
    center = (phat + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * total)) / total) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def bootstrap_ci(values: Sequence[float], n_resamples: int = 10000, confidence: float = 0.95, seed: int = 42) -> Tuple[float, float]:
    """Non-parametric bootstrap confidence interval for a mean."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    means = arr[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


def mcnemar_test(correct_a: Iterable[bool], correct_b: Iterable[bool]) -> Dict[str, float]:
    """Paired McNemar test for two classifiers on the same examples."""
    a = np.asarray(list(correct_a), dtype=bool)
    b = np.asarray(list(correct_b), dtype=bool)
    if a.shape != b.shape:
        raise ValueError("McNemar inputs must have matching shapes")
    b01 = int(np.logical_and(~a, b).sum())
    b10 = int(np.logical_and(a, ~b).sum())
    discordant = b01 + b10

    if discordant == 0:
        p_value = 1.0
        statistic = 0.0
    elif binomtest is not None:
        p_value = float(binomtest(min(b01, b10), discordant, p=0.5, alternative="two-sided").pvalue)
        statistic = float((abs(b01 - b10) - 1.0) ** 2 / discordant)
    else:
        statistic = float((abs(b01 - b10) - 1.0) ** 2 / discordant)
        p_value = float(1.0 - chi2.cdf(statistic, 1)) if chi2 is not None else 1.0

    return {
        "a_wrong_b_right": b01,
        "a_right_b_wrong": b10,
        "discordant_pairs": discordant,
        "mcnemar_statistic": statistic,
        "p_value": p_value,
    }


def summarize_binary_metric(correct: Sequence[bool], n_resamples: int = 10000) -> Dict[str, float]:
    arr = np.asarray(correct, dtype=np.float64)
    successes = int(arr.sum())
    total = int(arr.size)
    wilson_low, wilson_high = wilson_interval(successes, total)
    boot_low, boot_high = bootstrap_ci(arr, n_resamples=n_resamples)
    return {
        "mean": float(arr.mean()) if total else 0.0,
        "successes": successes,
        "total": total,
        "wilson_low": wilson_low,
        "wilson_high": wilson_high,
        "bootstrap_low": boot_low,
        "bootstrap_high": boot_high,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute CIs and optional McNemar tests")
    parser.add_argument("--predictions-a", required=True, help="Prediction CSV for model A")
    parser.add_argument("--predictions-b", help="Optional paired prediction CSV for model B")
    parser.add_argument("--target-col", default="target")
    parser.add_argument("--prediction-col", default="prediction")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    args = parser.parse_args()

    df_a = pd.read_csv(args.predictions_a).fillna("")
    correct_a = df_a[args.target_col].astype(str) == df_a[args.prediction_col].astype(str)
    result = {"model_a_exact": summarize_binary_metric(correct_a.tolist(), args.bootstrap_resamples)}

    if args.predictions_b:
        df_b = pd.read_csv(args.predictions_b).fillna("")
        correct_b = df_b[args.target_col].astype(str) == df_b[args.prediction_col].astype(str)
        result["model_b_exact"] = summarize_binary_metric(correct_b.tolist(), args.bootstrap_resamples)
        result["mcnemar"] = mcnemar_test(correct_a.tolist(), correct_b.tolist())

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
