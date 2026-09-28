#!/usr/bin/env python3
"""Generate all configured CAPTCHA-X journal datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.text_solver.local_generators import GeneratorDatasetConfig, create_renderer


def generate_from_config(config_path: str) -> None:
    with open(config_path, "r") as handle:
        config = json.load(handle)

    output_root = Path(config.get("dataset_root", config.get("output_root", "artifacts/journal_datasets"))) / "datasets"
    output_root.mkdir(parents=True, exist_ok=True)
    seed = int(config.get("seed", 42))
    difficulties = config.get("difficulties", ["clean", "easy", "medium", "hard"])
    generators = config.get("generators", [])

    for generator_index, generator_cfg in enumerate(generators):
        for difficulty_index, difficulty in enumerate(difficulties):
            dataset_dir = output_root / str(generator_cfg["name"]) / str(difficulty)
            renderer_cfg = GeneratorDatasetConfig(
                name=str(generator_cfg["name"]),
                backend=str(generator_cfg["backend"]),
                width=int(config.get("image_size", [200, 80])[0]),
                height=int(config.get("image_size", [200, 80])[1]),
                captcha_length=int(config.get("captcha_length", 5)),
                difficulty=str(difficulty),
                num_samples=int(config.get("num_samples_per_dataset", 1000)),
                split_ratio=tuple(config.get("split_ratio", [0.7, 0.15, 0.15])),
                seed=seed + generator_index * 1000 + difficulty_index * 100,
                characters=str(config.get("characters", "23456789ABCDEFGHJKLMNPQRSTUVWXYZ")),
                font_sizes=tuple(config.get("font_sizes", [40, 45, 50, 55, 60])),
                font_paths=generator_cfg.get("font_paths"),
            )
            print(f"Generating {generator_cfg['name']} / {difficulty} -> {dataset_dir}")
            create_renderer(renderer_cfg).generate_dataset(dataset_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate configured CAPTCHA-X datasets")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    generate_from_config(args.config)


if __name__ == "__main__":
    main()
