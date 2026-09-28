#!/usr/bin/env python3
"""Small smoke checks for the anonymized review artifact."""
from __future__ import annotations

import torch
import numpy as np


def main() -> int:
    from evaluation.failure_taxonomy import classify_failure
    from evaluation.vscore import compute_vscore_row
    from models.crnn_clean import CRNNClean
    from tools.benchmark_verifier import fft_artifact_score

    assert classify_failure("ABCDE", "ABCDE")["error_type"] == "CORRECT"
    row = compute_vscore_row({"accuracy": 0.5, "ece": 0.1, "uncertainty": 0.2, "ood_exposure": 0.0})
    assert 0.0 <= row["v_score"] <= 1.0
    model = CRNNClean(num_classes=34, hidden_size=32)
    output = model(torch.randn(1, 1, 80, 200))
    assert output.shape[0] == 1
    assert isinstance(fft_artifact_score(np.zeros((32, 32), dtype=np.uint8)), float)
    print("CAPTCHA-X artifact smoke checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
