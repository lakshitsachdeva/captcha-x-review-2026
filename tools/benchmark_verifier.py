#!/usr/bin/env python3
"""Pixel-level benchmark verification for CAPTCHA-X datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


DISTORTION_COLUMNS = ("rotation", "shear", "line_count", "dot_count", "blur_radius", "noise_std", "x_jitter", "y_jitter")


def resolve_image_path(data_dir: Path, row: Dict[str, object]) -> Path:
    filename = str(row["filename"])
    split = str(row.get("split", ""))
    for candidate in (data_dir / filename, data_dir / split / filename, data_dir.parent / split / filename):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve image for {filename}")


def fft_artifact_score(gray: np.ndarray) -> float:
    centered = gray.astype(np.float64) - float(gray.mean())
    spectrum = np.fft.fftshift(np.fft.fft2(centered))
    magnitude = np.abs(spectrum)
    h, w = magnitude.shape
    yy, xx = np.ogrid[:h, :w]
    radius = np.sqrt((yy - h / 2) ** 2 + (xx - w / 2) ** 2)
    high = magnitude[radius > min(h, w) * 0.18].sum()
    total = magnitude.sum()
    return float(high / max(total, 1e-8))


def background_fft_artifact_score(gray: np.ndarray) -> float:
    """Estimate high-frequency artifacts after suppressing foreground text edges."""
    background_threshold = np.percentile(gray, 82)
    background_mask = gray >= background_threshold
    if not background_mask.any():
        return fft_artifact_score(gray)
    background_value = float(np.median(gray[background_mask]))
    background_only = gray.astype(np.float64).copy()
    background_only[~background_mask] = background_value
    return fft_artifact_score(background_only.astype(np.uint8))


def background_std(gray: np.ndarray) -> float:
    bright_pixels = gray[gray >= np.percentile(gray, 85)]
    if bright_pixels.size == 0:
        return 0.0
    return float(bright_pixels.std())


def declared_distortion(row: Dict[str, object]) -> float:
    total = 0.0
    for column in DISTORTION_COLUMNS:
        try:
            total += abs(float(row.get(column, 0.0)))
        except (TypeError, ValueError):
            continue
    return float(total)


def verify_dataset(data_dir: str, metadata_file: str, output_dir: str, *, max_samples: int | None = None) -> pd.DataFrame:
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(metadata_file).fillna("").to_dict(orient="records")
    if max_samples is not None:
        rows = rows[:max_samples]

    report_rows: List[Dict[str, object]] = []
    for row in rows:
        image_path = resolve_image_path(data_path, row)
        image = Image.open(image_path).convert("L").resize((200, 80))
        gray = np.asarray(image, dtype=np.uint8)
        bg_std = background_std(gray)
        fft_score = fft_artifact_score(gray)
        bg_fft_score = background_fft_artifact_score(gray)
        fg_fraction = float((gray < np.percentile(gray, 70)).mean())
        declared = declared_distortion(row)
        clean_declared = declared < 1e-8
        suspicious_clean = bool(clean_declared and (bg_std > 2.5 or bg_fft_score > 0.18))
        report_rows.append(
            {
                "filename": row["filename"],
                "split": row.get("split", ""),
                "text": row.get("text", ""),
                "generator_name": row.get("generator_name", ""),
                "difficulty": row.get("difficulty", ""),
                "pixel_mean": float(gray.mean()),
                "pixel_std": float(gray.std()),
                "background_std": bg_std,
                "foreground_fraction": fg_fraction,
                "fft_high_frequency_ratio": fft_score,
                "background_fft_high_frequency_ratio": bg_fft_score,
                "declared_distortion_magnitude": declared,
                "declared_clean": clean_declared,
                "suspicious_clean_mismatch": suspicious_clean,
                "integrity_score": float(
                    np.clip(1.0 - (0.08 * bg_std + 1.5 * bg_fft_score + 0.5 * int(suspicious_clean)), 0.0, 1.0)
                ),
            }
        )

    df = pd.DataFrame(report_rows)
    df.to_csv(output_path / "verification_report.csv", index=False)
    summary = {
        "num_images": int(len(df)),
        "mean_integrity_score": float(df["integrity_score"].mean()) if not df.empty else 0.0,
        "suspicious_clean_mismatch_count": int(df["suspicious_clean_mismatch"].sum()) if not df.empty else 0,
        "mean_background_std": float(df["background_std"].mean()) if not df.empty else 0.0,
        "mean_fft_high_frequency_ratio": float(df["fft_high_frequency_ratio"].mean()) if not df.empty else 0.0,
        "mean_background_fft_high_frequency_ratio": float(df["background_fft_high_frequency_ratio"].mean()) if not df.empty else 0.0,
    }
    (output_path / "verification_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_histograms(df, output_path / "pixel_integrity_histograms.png")
    return df


def plot_histograms(df: pd.DataFrame, output_path: Path) -> None:
    if df.empty:
        return
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    axes[0].hist(df["background_std"], bins=30, color="#0072B2", alpha=0.8)
    axes[0].set_title("Background Std")
    axes[1].hist(df["fft_high_frequency_ratio"], bins=30, color="#009E73", alpha=0.8)
    axes[1].set_title("Full FFT Ratio")
    axes[2].hist(df["background_fft_high_frequency_ratio"], bins=30, color="#CC79A7", alpha=0.8)
    axes[2].set_title("Background FFT Ratio")
    axes[3].hist(df["integrity_score"], bins=30, color="#D55E00", alpha=0.8)
    axes[3].set_title("Integrity Score")
    for axis in axes:
        axis.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify pixel-level benchmark integrity")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--metadata-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    df = verify_dataset(
        args.data_dir,
        args.metadata_file,
        args.output_dir,
        max_samples=args.max_samples or None,
    )
    print(f"Wrote verification for {len(df)} images to {args.output_dir}")


if __name__ == "__main__":
    main()
