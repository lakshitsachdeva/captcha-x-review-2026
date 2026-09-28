#!/usr/bin/env python3
"""Run repeated cross-generator CAPTCHA-X experiments across random seeds."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.text_solver.local_benchmark_extension import load_config, run_experiment


def parse_seeds(seed_text: str, config: Dict[str, object]) -> List[int]:
    if seed_text:
        return [int(seed.strip()) for seed in seed_text.split(",") if seed.strip()]
    seed_cfg = config.get("seed_sweep", {})
    if isinstance(seed_cfg, dict) and seed_cfg.get("seeds"):
        return [int(seed) for seed in seed_cfg["seeds"]]
    return [int(config.get("seed", 42))]


def seed_config(base_config: Dict[str, object], seed: int, output_root: Path) -> Dict[str, object]:
    config = deepcopy(base_config)
    seed_root = output_root / f"seed_{seed}"
    config["seed"] = seed
    config["output_root"] = str(seed_root)
    config["dataset_root"] = str(seed_root)
    return config


def write_seed_config(config: Dict[str, object], output_root: Path, seed: int) -> Path:
    config_dir = output_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"seed_{seed}.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    return config_path


def collect_seed_result(seed: int, seed_root: Path) -> Dict[str, object]:
    summary_csv = seed_root / "tables" / "evaluation_summary.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(f"Missing evaluation summary for seed {seed}: {summary_csv}")
    df = pd.read_csv(summary_csv)
    matched = df[df["train_generator"] == df["test_generator"]].copy()
    medium_matched = matched[matched["difficulty"] == "medium"].copy()
    best_medium = medium_matched.sort_values("exact_accuracy", ascending=False).head(1)
    return {
        "seed": seed,
        "artifact_dir": str(seed_root),
        "num_conditions": int(len(df)),
        "matched_medium_mean_exact": float(medium_matched["exact_accuracy"].mean()) if not medium_matched.empty else None,
        "matched_medium_min_exact": float(medium_matched["exact_accuracy"].min()) if not medium_matched.empty else None,
        "best_medium_generator": str(best_medium.iloc[0]["train_generator"]) if not best_medium.empty else "",
        "best_medium_exact": float(best_medium.iloc[0]["exact_accuracy"]) if not best_medium.empty else None,
    }


def prune_seed_artifacts(seed_root: Path) -> None:
    """Drop bulky reproducible per-seed files after metrics are written."""
    for name in ("datasets", "human_baseline", "paper_bundle"):
        path = seed_root / name
        if path.exists():
            shutil.rmtree(path)
    archive = seed_root / "paper_bundle.tar.gz"
    if archive.exists():
        archive.unlink()


def aggregate_seed_tables(output_root: Path, seeds: List[int]) -> Dict[str, object]:
    frames = []
    for seed in seeds:
        seed_root = output_root / f"seed_{seed}"
        summary_csv = seed_root / "tables" / "evaluation_summary.csv"
        if not summary_csv.exists():
            continue
        df = pd.read_csv(summary_csv)
        df.insert(0, "seed", seed)
        frames.append(df)

    tables_dir = output_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    if not frames:
        payload = {"num_completed_seeds": 0, "completed_seeds": []}
        (tables_dir / "seed_sweep_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
        return payload

    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(tables_dir / "all_seed_evaluation_summary.csv", index=False)

    metric_cols = ["exact_accuracy", "char_accuracy", "ece", "avg_confidence", "avg_entropy", "num_samples"]
    grouped = all_df.groupby(["train_generator", "test_generator", "difficulty"], dropna=False)
    rows = []
    for keys, group in grouped:
        row = {
            "train_generator": keys[0],
            "test_generator": keys[1],
            "difficulty": keys[2],
            "num_seeds": int(group["seed"].nunique()),
        }
        for metric in metric_cols:
            if metric in group.columns:
                row[f"{metric}_mean"] = float(group[metric].mean())
                row[f"{metric}_std"] = float(group[metric].std(ddof=0))
                row[f"{metric}_min"] = float(group[metric].min())
                row[f"{metric}_max"] = float(group[metric].max())
        rows.append(row)

    aggregate_df = pd.DataFrame(rows)
    aggregate_df.to_csv(tables_dir / "seed_sweep_aggregate.csv", index=False)

    matched = aggregate_df[aggregate_df["train_generator"] == aggregate_df["test_generator"]].copy()
    matched.to_csv(tables_dir / "seed_sweep_matched_aggregate.csv", index=False)

    payload = {
        "num_completed_seeds": int(all_df["seed"].nunique()),
        "completed_seeds": sorted(int(seed) for seed in all_df["seed"].unique()),
        "num_conditions_per_seed": int(len(all_df) / max(all_df["seed"].nunique(), 1)),
        "matched_medium": matched[matched["difficulty"] == "medium"].to_dict(orient="records"),
    }
    (tables_dir / "seed_sweep_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a cross-generator seed sweep")
    parser.add_argument("--config", default="configs/journal_seed_sweep_core.json")
    parser.add_argument("--output-root", default="artifacts/journal_seed_sweep_core")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--reference-paper", default="docs/paper_ieee.tex")
    parser.add_argument("--skip-existing", action="store_true", help="Do not rerun seeds that already have evaluation_summary.csv")
    parser.add_argument("--prune-datasets", action="store_true", help="Remove bulky generated images after each seed completes")
    args = parser.parse_args()

    base_config = load_config(args.config)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(args.seeds, base_config)
    seed_results = []

    for seed in seeds:
        config = seed_config(base_config, seed, output_root)
        seed_root = Path(config["output_root"])
        if args.skip_existing and (seed_root / "tables" / "evaluation_summary.csv").exists():
            print(f"Seed {seed} already complete, skipping: {seed_root}")
        else:
            config_path = write_seed_config(config, output_root, seed)
            run_experiment(config, str(config_path), reference_paper=args.reference_paper)
        seed_results.append(collect_seed_result(seed, seed_root))
        if args.prune_datasets:
            prune_seed_artifacts(seed_root)

    payload = {
        "base_config": args.config,
        "output_root": str(output_root),
        "seeds": seeds,
        "seed_results": seed_results,
        "aggregate": aggregate_seed_tables(output_root, seeds),
    }
    (output_root / "seed_sweep_results.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
