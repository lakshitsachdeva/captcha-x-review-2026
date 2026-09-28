#!/usr/bin/env python3
"""
Synthetic text CAPTCHA generation with controllable rendering difficulty.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from tqdm import tqdm


DEFAULT_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Courier New Bold.ttf",
    "/System/Library/Fonts/Supplemental/Georgia Bold.ttf",
    "/System/Library/Fonts/Supplemental/Impact.ttf",
    "/System/Library/Fonts/Supplemental/AmericanTypewriter.ttc",
]


DIFFICULTY_PROFILES: Dict[str, Dict[str, float]] = {
    "clean": {
        "max_rotation": 0.0,
        "max_shear": 0.0,
        "min_lines": 0,
        "max_lines": 0,
        "min_dots": 0,
        "max_dots": 0,
        "max_blur": 0.0,
        "max_noise": 0.0,
        "max_x_jitter": 0.0,
        "max_y_jitter": 0.0,
    },
    "easy": {
        "max_rotation": 3.0,
        "max_shear": 0.04,
        "min_lines": 0,
        "max_lines": 1,
        "min_dots": 5,
        "max_dots": 20,
        "max_blur": 0.3,
        "max_noise": 6.0,
        "max_x_jitter": 3.0,
        "max_y_jitter": 4.0,
    },
    "medium": {
        "max_rotation": 7.0,
        "max_shear": 0.08,
        "min_lines": 1,
        "max_lines": 3,
        "min_dots": 20,
        "max_dots": 60,
        "max_blur": 0.7,
        "max_noise": 12.0,
        "max_x_jitter": 6.0,
        "max_y_jitter": 8.0,
    },
    "hard": {
        "max_rotation": 12.0,
        "max_shear": 0.12,
        "min_lines": 2,
        "max_lines": 5,
        "min_dots": 50,
        "max_dots": 110,
        "max_blur": 1.2,
        "max_noise": 18.0,
        "max_x_jitter": 10.0,
        "max_y_jitter": 12.0,
    },
}


class TextCaptchaGenerator:
    """Generate synthetic CAPTCHA data with recorded nuisance parameters."""

    def __init__(
        self,
        width: int = 200,
        height: int = 80,
        font_sizes: Optional[List[int]] = None,
        font_paths: Optional[List[str]] = None,
        length_range: Tuple[int, int] = (4, 6),
        difficulty: str = "medium",
        seed: Optional[int] = 42,
        characters: str = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ",
    ):
        if difficulty not in DIFFICULTY_PROFILES:
            raise ValueError(f"Unknown difficulty '{difficulty}'. Choose from {sorted(DIFFICULTY_PROFILES)}")

        self.width = width
        self.height = height
        self.font_sizes = font_sizes or [40, 45, 50, 55, 60]
        self.length_range = length_range
        self.difficulty = difficulty
        self.profile = DIFFICULTY_PROFILES[difficulty]
        self.characters = characters
        self.random = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.font_paths = self._resolve_font_paths(font_paths)
        self.font_cache: Dict[Tuple[str, int], ImageFont.FreeTypeFont] = {}

    def _resolve_font_paths(self, font_paths: Optional[List[str]]) -> List[Path]:
        candidates = [Path(path) for path in (font_paths or DEFAULT_FONT_CANDIDATES)]
        resolved = [path for path in candidates if path.exists()]
        if not resolved:
            print("Warning: no system fonts found; falling back to PIL default font")
        return resolved

    def _get_font(self, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        if not self.font_paths:
            return ImageFont.load_default()

        font_path = self.random.choice(self.font_paths)
        cache_key = (str(font_path), size)
        if cache_key not in self.font_cache:
            self.font_cache[cache_key] = ImageFont.truetype(str(font_path), size=size)
        return self.font_cache[cache_key]

    def generate_random_text(self, length: Optional[int] = None) -> str:
        if length is None:
            length = self.random.randint(self.length_range[0], self.length_range[1])
        return "".join(self.random.choices(self.characters, k=length))

    def _sample_render_params(self) -> Dict[str, float | int]:
        return {
            "rotation": self.random.uniform(-self.profile["max_rotation"], self.profile["max_rotation"]),
            "shear": self.random.uniform(-self.profile["max_shear"], self.profile["max_shear"]),
            "line_count": self.random.randint(int(self.profile["min_lines"]), int(self.profile["max_lines"])),
            "dot_count": self.random.randint(int(self.profile["min_dots"]), int(self.profile["max_dots"])),
            "blur_radius": self.random.uniform(0.0, self.profile["max_blur"]),
            "noise_std": self.random.uniform(0.0, self.profile["max_noise"]),
            "x_jitter": int(round(self.random.uniform(0.0, self.profile["max_x_jitter"]))),
            "y_jitter": int(round(self.random.uniform(0.0, self.profile["max_y_jitter"]))),
        }

    def generate_captcha(self, text: Optional[str] = None) -> Tuple[Image.Image, str, Dict[str, float | int]]:
        if text is None:
            text = self.generate_random_text()

        params = self._sample_render_params()
        image = self._render_text_image(text, params)
        image = self._apply_distortions(image, params)
        return image, text, params

    def _render_text_image(self, text: str, params: Dict[str, float | int]) -> Image.Image:
        canvas = Image.new("RGBA", (self.width, self.height), (255, 255, 255, 255))
        slot_width = self.width / max(len(text), 1)
        draw_margin = max(8, int(round(slot_width * 0.15)))

        for index, char in enumerate(text):
            font_size = self.random.choice(self.font_sizes)
            font = self._get_font(font_size)
            color_value = int(self.np_rng.integers(20, 90))
            glyph = self._render_character(char, font, (color_value, color_value, color_value, 255), slot_width)

            char_rotation = float(self.random.uniform(-float(params["rotation"]), float(params["rotation"])))
            if abs(char_rotation) > 1e-6:
                glyph = glyph.rotate(
                    char_rotation,
                    resample=Image.Resampling.BICUBIC,
                    expand=True,
                )

            x_center = int(round((index + 0.5) * slot_width))
            y_center = self.height // 2

            if int(params["x_jitter"]) > 0:
                x_center += self.random.randint(-int(params["x_jitter"]), int(params["x_jitter"]))
            if int(params["y_jitter"]) > 0:
                y_center += self.random.randint(-int(params["y_jitter"]), int(params["y_jitter"]))

            min_x = int(index * slot_width) + draw_margin
            max_x = int((index + 1) * slot_width) - draw_margin
            x = int(np.clip(x_center - glyph.width // 2, min_x - glyph.width // 3, max_x - (2 * glyph.width) // 3))
            y = int(np.clip(y_center - glyph.height // 2, 4, max(4, self.height - glyph.height - 4)))

            canvas.alpha_composite(glyph, (x, y))

        return canvas.convert("RGB")

    def _render_character(
        self,
        char: str,
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
        color: Tuple[int, int, int, int],
        slot_width: float,
    ) -> Image.Image:
        tile_width = max(int(round(slot_width * 1.8)), 80)
        tile_height = max(int(round(self.height * 1.6)), 96)
        tile = Image.new("RGBA", (tile_width, tile_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(tile)
        bbox = draw.textbbox((0, 0), char, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (tile_width - text_width) / 2 - bbox[0]
        y = (tile_height - text_height) / 2 - bbox[1]
        draw.text((x, y), char, font=font, fill=color)
        crop_box = tile.getbbox()
        return tile.crop(crop_box) if crop_box else tile

    def _apply_distortions(self, image: Image.Image, params: Dict[str, float | int]) -> Image.Image:
        shear = float(params["shear"])
        x_shift = abs(shear) * self.height
        affine_matrix = (1, shear, -x_shift if shear > 0 else 0, 0, 1, 0)
        distorted = image.transform(
            (self.width, self.height),
            Image.Transform.AFFINE,
            affine_matrix,
            resample=Image.Resampling.BICUBIC,
            fillcolor=(255, 255, 255),
        )

        draw = ImageDraw.Draw(distorted)
        for _ in range(int(params["line_count"])):
            start = (self.random.randint(0, self.width), self.random.randint(0, self.height))
            end = (self.random.randint(0, self.width), self.random.randint(0, self.height))
            color = tuple(int(v) for v in self.np_rng.integers(30, 180, size=3))
            thickness = self.random.randint(1, 2)
            draw.line([start, end], fill=color, width=thickness)

        for _ in range(int(params["dot_count"])):
            x = self.random.randint(0, self.width - 1)
            y = self.random.randint(0, self.height - 1)
            color = tuple(int(v) for v in self.np_rng.integers(60, 220, size=3))
            draw.point((x, y), fill=color)

        if float(params["blur_radius"]) > 0:
            distorted = distorted.filter(ImageFilter.GaussianBlur(radius=float(params["blur_radius"])))

        noisy = np.array(distorted, dtype=np.float32)
        if float(params["noise_std"]) > 0:
            noise = self.np_rng.normal(0.0, float(params["noise_std"]), noisy.shape)
            noisy = np.clip(noisy + noise, 0, 255)

        return Image.fromarray(noisy.astype(np.uint8))

    def generate_dataset(
        self,
        num_samples: int,
        output_dir: str,
        split_ratio: Tuple[float, float, float] = (0.7, 0.2, 0.1),
    ) -> str:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        train_size = int(num_samples * split_ratio[0])
        val_size = int(num_samples * split_ratio[1])
        test_size = num_samples - train_size - val_size

        split_labels = ["train"] * train_size + ["val"] * val_size + ["test"] * test_size
        self.random.shuffle(split_labels)

        for split_name in ("train", "val", "test"):
            split_path = output_path / split_name
            if split_path.exists():
                shutil.rmtree(split_path)
            split_path.mkdir(exist_ok=True)

        captcha_data = []
        print(f"Generating {num_samples} CAPTCHAs with difficulty='{self.difficulty}'")
        print(f"Character set: {self.characters}")
        print(f"Splits - Train: {train_size}, Val: {val_size}, Test: {test_size}")

        for index in tqdm(range(num_samples), desc="Generating CAPTCHAs"):
            split = split_labels[index]
            text = self.generate_random_text()
            image, _, params = self.generate_captcha(text)

            filename = f"captcha_{index:06d}.png"
            image_path = output_path / split / filename
            image.save(image_path, "PNG")

            captcha_data.append(
                {
                    "id": index,
                    "filename": filename,
                    "text": text,
                    "split": split,
                    "length": len(text),
                    "difficulty": self.difficulty,
                    "rotation": round(float(params["rotation"]), 3),
                    "shear": round(float(params["shear"]), 4),
                    "line_count": int(params["line_count"]),
                    "dot_count": int(params["dot_count"]),
                    "blur_radius": round(float(params["blur_radius"]), 3),
                    "noise_std": round(float(params["noise_std"]), 3),
                }
            )

        df = pd.DataFrame(captcha_data)
        df.to_csv(output_path / "metadata.csv", index=False)

        with open(output_path / "metadata.json", "w") as handle:
            json.dump(captcha_data, handle, indent=2)

        all_chars = "".join(df["text"].astype(str).tolist())
        char_stats = {char: all_chars.count(char) for char in self.characters}
        with open(output_path / "char_stats.json", "w") as handle:
            json.dump(
                {
                    "character_set": self.characters,
                    "character_frequencies": char_stats,
                    "total_samples": len(df),
                    "avg_length": float(df["length"].mean()) if not df.empty else 0.0,
                    "length_distribution": {
                        str(length): int((df["length"] == length).sum())
                        for length in range(self.length_range[0], self.length_range[1] + 1)
                    },
                    "difficulty": self.difficulty,
                },
                handle,
                indent=2,
            )

        print(f"\nDataset generated successfully at {output_path}")
        return str(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic text CAPTCHAs")
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of CAPTCHAs to generate")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/synthetic_text_captchas",
        help="Output directory for generated CAPTCHAs",
    )
    parser.add_argument("--width", type=int, default=200, help="Width of CAPTCHA images")
    parser.add_argument("--height", type=int, default=80, help="Height of CAPTCHA images")
    parser.add_argument("--min-length", type=int, default=4, help="Minimum CAPTCHA length")
    parser.add_argument("--max-length", type=int, default=6, help="Maximum CAPTCHA length")
    parser.add_argument(
        "--difficulty",
        type=str,
        choices=sorted(DIFFICULTY_PROFILES),
        default="medium",
        help="Generation difficulty profile",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    generator = TextCaptchaGenerator(
        width=args.width,
        height=args.height,
        length_range=(args.min_length, args.max_length),
        difficulty=args.difficulty,
        seed=args.seed,
    )

    dataset_path = generator.generate_dataset(args.num_samples, args.output_dir)
    print("\n=== Generation Complete ===")
    print(f"Dataset saved to: {dataset_path}")
    print("To train a model, run:")
    print(
        "python -m src.text_solver.train "
        f"--data-dir {dataset_path} --metadata-file {dataset_path}/metadata.csv"
    )


if __name__ == "__main__":
    main()
