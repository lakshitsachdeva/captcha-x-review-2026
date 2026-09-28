#!/usr/bin/env python3
"""
Reliability utilities for local-only CAPTCHA benchmarking.

These helpers intentionally operate on offline model outputs from synthetic or
locally hosted datasets. They are designed for reproducible benchmark auditing,
not for interacting with third-party systems.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F

try:
    from sklearn.metrics import roc_auc_score
except ImportError:  # pragma: no cover - optional during lightweight use.
    roc_auc_score = None


@dataclass
class TemperatureScalingResult:
    """Outcome of post-hoc temperature scaling."""

    temperature: float
    nll_before: float
    nll_after: float


def _flatten_for_nll(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.dim() != 3:
        raise ValueError(f"Expected logits with shape [N, L, C], got {tuple(logits.shape)}")
    if targets.dim() != 2:
        raise ValueError(f"Expected targets with shape [N, L], got {tuple(targets.shape)}")
    return logits.reshape(-1, logits.size(-1)), targets.reshape(-1)


def compute_position_nll(logits: torch.Tensor, targets: torch.Tensor, temperature: float = 1.0, ignore_index: int = 0) -> float:
    """Cross-entropy on flattened position logits."""
    flat_logits, flat_targets = _flatten_for_nll(logits, targets, ignore_index=ignore_index)
    scaled_logits = flat_logits / max(float(temperature), 1e-4)
    return float(F.cross_entropy(scaled_logits, flat_targets, ignore_index=ignore_index).item())


def fit_temperature(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    max_iter: int = 200,
    lr: float = 0.05,
    ignore_index: int = 0,
) -> TemperatureScalingResult:
    """
    Learn a single post-hoc temperature on validation logits.

    The optimization objective is position-wise cross-entropy over the held-out
    validation set, which is standard for temperature scaling in classifiers.
    """

    flat_logits, flat_targets = _flatten_for_nll(logits.detach(), targets.detach(), ignore_index=ignore_index)
    flat_logits = flat_logits.float()
    flat_targets = flat_targets.long()

    log_temperature = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32, device=flat_logits.device))
    optimizer = torch.optim.Adam([log_temperature], lr=lr)

    nll_before = float(F.cross_entropy(flat_logits, flat_targets, ignore_index=ignore_index).item())

    for _ in range(max_iter):
        optimizer.zero_grad(set_to_none=True)
        temperature = torch.exp(log_temperature).clamp(min=1e-4, max=100.0)
        loss = F.cross_entropy(flat_logits / temperature, flat_targets, ignore_index=ignore_index)
        loss.backward()
        optimizer.step()

    learned_temperature = float(torch.exp(log_temperature.detach()).clamp(min=1e-4, max=100.0).item())
    nll_after = compute_position_nll(flat_logits.reshape(logits.shape), targets, temperature=learned_temperature, ignore_index=ignore_index)
    return TemperatureScalingResult(
        temperature=learned_temperature,
        nll_before=nll_before,
        nll_after=nll_after,
    )


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Scale logits without mutating the original tensor."""
    return logits / max(float(temperature), 1e-4)


def sequence_confidence(logits: torch.Tensor, temperature: float = 1.0) -> np.ndarray:
    """
    Geometric-mean confidence across positions.

    This is more conservative than a simple arithmetic mean and better reflects
    sequence-level correctness, where a single weak character can spoil the full
    CAPTCHA string.
    """

    scaled = apply_temperature(logits, temperature)
    probabilities = torch.softmax(scaled, dim=-1)
    max_probs = probabilities.max(dim=-1).values.clamp(min=1e-8)
    log_confidence = torch.log(max_probs).mean(dim=-1)
    return torch.exp(log_confidence).detach().cpu().numpy()


def sequence_entropy(logits: torch.Tensor, temperature: float = 1.0, normalize: bool = True) -> np.ndarray:
    """Average per-position entropy."""
    scaled = apply_temperature(logits, temperature)
    probabilities = torch.softmax(scaled, dim=-1)
    log_probabilities = torch.log(probabilities.clamp(min=1e-8))
    entropy = -(probabilities * log_probabilities).sum(dim=-1).mean(dim=-1)
    if normalize:
        normalizer = float(np.log(logits.size(-1))) if logits.size(-1) > 1 else 1.0
        entropy = entropy / max(normalizer, 1e-8)
    return entropy.detach().cpu().numpy()


def expected_calibration_error(
    confidences: Sequence[float],
    correctness: Sequence[bool | int],
    *,
    n_bins: int = 10,
) -> float:
    """Standard expected calibration error."""
    if len(confidences) != len(correctness):
        raise ValueError("Confidences and correctness must have the same length")
    if not confidences:
        return 0.0

    conf_arr = np.asarray(confidences, dtype=np.float64)
    correct_arr = np.asarray(correctness, dtype=np.float64)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    ece = 0.0
    total = len(conf_arr)
    for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
        if upper == 1.0:
            mask = (conf_arr >= lower) & (conf_arr <= upper)
        else:
            mask = (conf_arr >= lower) & (conf_arr < upper)
        if not np.any(mask):
            continue
        bin_accuracy = float(correct_arr[mask].mean())
        bin_confidence = float(conf_arr[mask].mean())
        ece += (mask.sum() / total) * abs(bin_accuracy - bin_confidence)
    return float(ece)


def calibration_table(
    confidences: Sequence[float],
    correctness: Sequence[bool | int],
    *,
    n_bins: int = 10,
) -> List[Dict[str, float]]:
    """Detailed per-bin calibration summary."""
    conf_arr = np.asarray(confidences, dtype=np.float64)
    correct_arr = np.asarray(correctness, dtype=np.float64)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: List[Dict[str, float]] = []

    for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
        if upper == 1.0:
            mask = (conf_arr >= lower) & (conf_arr <= upper)
        else:
            mask = (conf_arr >= lower) & (conf_arr < upper)
        if np.any(mask):
            accuracy = float(correct_arr[mask].mean())
            confidence = float(conf_arr[mask].mean())
            count = int(mask.sum())
        else:
            accuracy = 0.0
            confidence = 0.0
            count = 0
        rows.append(
            {
                "bin_lower": float(lower),
                "bin_upper": float(upper),
                "count": count,
                "accuracy": accuracy,
                "avg_confidence": confidence,
                "gap": abs(accuracy - confidence),
            }
        )
    return rows


def abstention_curve(
    confidences: Sequence[float],
    correctness: Sequence[bool | int],
    *,
    thresholds: Iterable[float],
) -> List[Dict[str, float]]:
    """Coverage/accuracy trade-off under confidence-threshold abstention."""
    conf_arr = np.asarray(confidences, dtype=np.float64)
    correct_arr = np.asarray(correctness, dtype=np.float64)
    total = len(conf_arr)
    rows: List[Dict[str, float]] = []

    for threshold in thresholds:
        keep_mask = conf_arr >= float(threshold)
        coverage = float(keep_mask.mean()) if total else 0.0
        selected = int(keep_mask.sum())
        selective_accuracy = float(correct_arr[keep_mask].mean()) if selected else 0.0
        abstention_rate = 1.0 - coverage
        rows.append(
            {
                "threshold": float(threshold),
                "coverage": coverage,
                "selective_accuracy": selective_accuracy,
                "abstention_rate": abstention_rate,
                "selected_samples": selected,
                "total_samples": total,
            }
        )
    return rows


def entropy_ood_summary(in_distribution_scores: Sequence[float], ood_scores: Sequence[float]) -> Dict[str, float]:
    """
    Summarize entropy-based OOD separation.

    Higher entropy implies greater uncertainty, so OOD examples are expected to
    have larger scores than in-distribution examples.
    """

    in_scores = np.asarray(in_distribution_scores, dtype=np.float64)
    ood = np.asarray(ood_scores, dtype=np.float64)
    if in_scores.size == 0 or ood.size == 0:
        return {"auroc": 0.0, "id_mean_entropy": 0.0, "ood_mean_entropy": 0.0}

    labels = np.concatenate([np.zeros_like(in_scores), np.ones_like(ood)])
    scores = np.concatenate([in_scores, ood])
    auroc = float(roc_auc_score(labels, scores)) if roc_auc_score is not None else 0.0
    return {
        "auroc": auroc,
        "id_mean_entropy": float(in_scores.mean()),
        "ood_mean_entropy": float(ood.mean()),
    }

