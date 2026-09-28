#!/usr/bin/env python3
"""Run the local-only cross-generator transfer matrix experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.text_solver.local_benchmark_extension import load_config, run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CAPTCHA-X cross-generator benchmark")
    parser.add_argument("--config", default="configs/journal_smoke.json")
    parser.add_argument("--reference-paper", default="docs/paper_ieee.tex")
    args = parser.parse_args()
    run_experiment(load_config(args.config), args.config, reference_paper=args.reference_paper)


if __name__ == "__main__":
    main()
