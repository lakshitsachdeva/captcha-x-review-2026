#!/usr/bin/env python3
"""
Local-only text CAPTCHA generator backends for benchmark transfer studies.

All generators in this module are intended for reproducible offline experiments
that operate on local assets only. They do not contact live services and are
designed for benchmark auditing, cross-generator transfer analysis, and human
baseline studies on self-generated data.
"""

from __future__ import annotations

import json
import math
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageFont

try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:  # pragma: no cover - optional fallback.
    cv2 = None
    CV2_AVAILABLE = False

from captcha.image import ImageCaptcha

from .gen_captchas import DEFAULT_FONT_CANDIDATES, DIFFICULTY_PROFILES, TextCaptchaGenerator


DEFAULT_CHARACTERS = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"


@dataclass
class GeneratorDatasetConfig:
    """Resolved dataset configuration for a single local backend."""

    name: str
    backend: str
    width: int = 200
    height: int = 80
    captcha_length: int = 5
    difficulty: str = "medium"
    num_samples: int = 800
    split_ratio: Tuple[float, float, float] = (0.6, 0.2, 0.2)
    seed: int = 42
    characters: str = DEFAULT_CHARACTERS
    font_sizes: Tuple[int, ...] = (40, 45, 50, 55, 60)
    font_paths: Optional[List[str]] = None


def resolve_font_paths(font_paths: Optional[Sequence[str]] = None) -> List[str]:
    candidates = [Path(path) for path in (font_paths or DEFAULT_FONT_CANDIDATES)]
    return [str(path) for path in candidates if path.exists()]


def sample_render_params(
    rng: random.Random,
    profile: Dict[str, float],
) -> Dict[str, float | int]:
    return {
        "rotation": rng.uniform(-float(profile["max_rotation"]), float(profile["max_rotation"])),
        "shear": rng.uniform(-float(profile["max_shear"]), float(profile["max_shear"])),
        "line_count": rng.randint(int(profile["min_lines"]), int(profile["max_lines"])),
        "dot_count": rng.randint(int(profile["min_dots"]), int(profile["max_dots"])),
        "blur_radius": rng.uniform(0.0, float(profile["max_blur"])),
        "noise_std": rng.uniform(0.0, float(profile["max_noise"])),
        "x_jitter": int(round(rng.uniform(0.0, float(profile["max_x_jitter"])))),
        "y_jitter": int(round(rng.uniform(0.0, float(profile["max_y_jitter"])))),
    }


class BaseLocalRenderer:
    """Shared generator interface with auditable metadata emission."""

    backend_name = "base"

    def __init__(self, config: GeneratorDatasetConfig):
        if config.difficulty not in DIFFICULTY_PROFILES:
            raise ValueError(f"Unknown difficulty '{config.difficulty}'")
        self.config = config
        self.profile = DIFFICULTY_PROFILES[config.difficulty]
        self.random = random.Random(config.seed)
        self.np_rng = np.random.default_rng(config.seed)
        self.font_paths = resolve_font_paths(config.font_paths)
        self.font_cache: Dict[Tuple[str, int], ImageFont.FreeTypeFont] = {}

    def generate_text(self) -> str:
        return "".join(self.random.choices(self.config.characters, k=self.config.captcha_length))

    def _get_font(self, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        if not self.font_paths:
            return ImageFont.load_default()
        font_path = self.random.choice(self.font_paths)
        cache_key = (font_path, size)
        if cache_key not in self.font_cache:
            self.font_cache[cache_key] = ImageFont.truetype(font_path, size=size)
        return self.font_cache[cache_key]

    def render(self, text: str) -> Tuple[Image.Image, Dict[str, float | int]]:
        raise NotImplementedError

    def generate_dataset(self, output_dir: str | Path) -> str:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        train_size = int(self.config.num_samples * self.config.split_ratio[0])
        val_size = int(self.config.num_samples * self.config.split_ratio[1])
        test_size = self.config.num_samples - train_size - val_size
        split_labels = ["train"] * train_size + ["val"] * val_size + ["test"] * test_size
        self.random.shuffle(split_labels)

        for split_name in ("train", "val", "test"):
            split_dir = output_path / split_name
            if split_dir.exists():
                shutil.rmtree(split_dir)
            split_dir.mkdir(parents=True, exist_ok=True)

        metadata_rows: List[Dict[str, object]] = []

        for index in range(self.config.num_samples):
            split = split_labels[index]
            text = self.generate_text()
            image, params = self.render(text)
            filename = f"{self.config.name}_{self.config.difficulty}_{index:06d}.png"
            image_path = output_path / split / filename
            image.save(image_path, "PNG")

            metadata_row: Dict[str, object] = {
                "id": index,
                "filename": filename,
                "text": text,
                "split": split,
                "length": len(text),
                "difficulty": self.config.difficulty,
                "generator_name": self.config.name,
                "generator_backend": self.backend_name,
                "rotation": round(float(params["rotation"]), 3),
                "shear": round(float(params["shear"]), 4),
                "line_count": int(params["line_count"]),
                "dot_count": int(params["dot_count"]),
                "blur_radius": round(float(params["blur_radius"]), 3),
                "noise_std": round(float(params["noise_std"]), 3),
                "x_jitter": int(params["x_jitter"]),
                "y_jitter": int(params["y_jitter"]),
            }
            for key, value in params.items():
                if key in metadata_row:
                    continue
                if isinstance(value, float):
                    metadata_row[key] = round(value, 4)
                else:
                    metadata_row[key] = value
            metadata_rows.append(metadata_row)

        df = pd.DataFrame(metadata_rows)
        df.to_csv(output_path / "metadata.csv", index=False)

        with open(output_path / "metadata.json", "w") as handle:
            json.dump(metadata_rows, handle, indent=2)

        all_chars = "".join(df["text"].astype(str).tolist())
        char_stats = {char: all_chars.count(char) for char in self.config.characters}

        manifest = {
            "dataset_config": asdict(self.config),
            "resolved_font_paths": self.font_paths,
            "difficulty_profile": DIFFICULTY_PROFILES[self.config.difficulty],
            "character_stats": char_stats,
            "avg_length": float(df["length"].mean()) if not df.empty else 0.0,
        }
        with open(output_path / "dataset_manifest.json", "w") as handle:
            json.dump(manifest, handle, indent=2)

        return str(output_path)


class ControlledPILRenderer(BaseLocalRenderer):
    """Wrapper around the audited PIL renderer already present in CAPTCHA-X."""

    backend_name = "pil_controlled"

    def __init__(self, config: GeneratorDatasetConfig):
        super().__init__(config)
        self.generator = TextCaptchaGenerator(
            width=config.width,
            height=config.height,
            font_sizes=list(config.font_sizes),
            font_paths=self.font_paths or None,
            length_range=(config.captcha_length, config.captcha_length),
            difficulty=config.difficulty,
            seed=config.seed,
            characters=config.characters,
        )

    def render(self, text: str) -> Tuple[Image.Image, Dict[str, float | int]]:
        image, _, params = self.generator.generate_captcha(text)
        return image.convert("RGB"), params


class CaptchaPackageRenderer(BaseLocalRenderer):
    """Generator based on the open-source `captcha` package."""

    backend_name = "captcha_image"

    def __init__(self, config: GeneratorDatasetConfig):
        super().__init__(config)
        self.image_captcha = ImageCaptcha(
            width=config.width,
            height=config.height,
            fonts=self.font_paths or None,
            font_sizes=list(config.font_sizes),
        )

    def render(self, text: str) -> Tuple[Image.Image, Dict[str, float | int]]:
        params = sample_render_params(self.random, self.profile)
        base_image = self.image_captcha.generate_image(text).convert("RGB")
        distorted = apply_common_distortions(
            base_image,
            params,
            rng=self.random,
            np_rng=self.np_rng,
            translate_first=True,
        )
        return distorted, params


class OpenCVRenderer(BaseLocalRenderer):
    """Simple OpenCV-based text renderer with distinct style from the PIL backends."""

    backend_name = "opencv_text"

    cv2_fonts = (
        getattr(cv2, "FONT_HERSHEY_SIMPLEX", 0),
        getattr(cv2, "FONT_HERSHEY_DUPLEX", 0),
        getattr(cv2, "FONT_HERSHEY_COMPLEX", 0),
        getattr(cv2, "FONT_HERSHEY_TRIPLEX", 0),
    )

    def render(self, text: str) -> Tuple[Image.Image, Dict[str, float | int]]:
        params = sample_render_params(self.random, self.profile)
        canvas = np.full((self.config.height, self.config.width, 3), 255, dtype=np.uint8)
        slot_width = self.config.width / max(len(text), 1)

        if CV2_AVAILABLE:
            for index, char in enumerate(text):
                font = self.random.choice(self.cv2_fonts)
                font_scale = self.random.uniform(1.0, 1.45)
                thickness = self.random.randint(2, 3)
                (char_width, char_height), baseline = cv2.getTextSize(char, font, font_scale, thickness)
                x_center = int(round((index + 0.5) * slot_width))
                y_center = self.config.height // 2 + self.random.randint(-int(params["y_jitter"]), int(params["y_jitter"]))
                x_center += self.random.randint(-int(params["x_jitter"]), int(params["x_jitter"]))
                x = max(4, min(self.config.width - char_width - 4, x_center - char_width // 2))
                y = max(char_height + 4, min(self.config.height - 4, y_center + char_height // 2))
                color_value = int(self.np_rng.integers(25, 95))
                cv2.putText(
                    canvas,
                    char,
                    (x, y),
                    font,
                    font_scale,
                    (color_value, color_value, color_value),
                    thickness,
                    lineType=cv2.LINE_AA,
                )
            base_image = Image.fromarray(canvas, mode="RGB")
        else:
            # Fallback to a light-weight PIL draw if OpenCV is unavailable.
            base_image = Image.new("RGB", (self.config.width, self.config.height), (255, 255, 255))
            draw = ImageDraw.Draw(base_image)
            for index, char in enumerate(text):
                font = self._get_font(self.random.choice(self.config.font_sizes))
                bbox = draw.textbbox((0, 0), char, font=font)
                char_width = bbox[2] - bbox[0]
                char_height = bbox[3] - bbox[1]
                x_center = int(round((index + 0.5) * slot_width))
                x_center += self.random.randint(-int(params["x_jitter"]), int(params["x_jitter"]))
                y_center = self.config.height // 2 + self.random.randint(-int(params["y_jitter"]), int(params["y_jitter"]))
                x = max(4, min(self.config.width - char_width - 4, x_center - char_width // 2))
                y = max(4, min(self.config.height - char_height - 4, y_center - char_height // 2))
                color_value = int(self.np_rng.integers(25, 95))
                draw.text((x, y), char, font=font, fill=(color_value, color_value, color_value))

        distorted = apply_common_distortions(
            base_image,
            params,
            rng=self.random,
            np_rng=self.np_rng,
            translate_first=False,
        )
        return distorted, params


class WandTextRenderer(BaseLocalRenderer):
    """
    Wave/arc renderer inspired by ImageMagick/Wand CAPTCHA effects.

    The implementation stays pure-Python so the benchmark remains runnable on
    machines without a system ImageMagick install. It records Wand-like wave
    amplitude, frequency, and arc-angle metadata, making it suitable as the
    fourth generator family in the journal transfer taxonomy.
    """

    backend_name = "wand_text"

    def render(self, text: str) -> Tuple[Image.Image, Dict[str, float | int]]:
        params = sample_render_params(self.random, self.profile)
        params.update(
            {
                "wave_amplitude": self.random.uniform(1.0, 7.0 + 0.35 * float(params["dot_count"])),
                "wave_frequency": self.random.uniform(0.75, 2.5),
                "arc_angle": self.random.uniform(-16.0, 16.0),
            }
        )

        canvas = Image.new("RGB", (self.config.width, self.config.height), (255, 255, 255))
        char_layer = Image.new("RGBA", canvas.size, (255, 255, 255, 0))
        slot_width = self.config.width / max(len(text), 1)

        for index, char in enumerate(text):
            font = self._get_font(self.random.choice(self.config.font_sizes))
            temp = Image.new("RGBA", (64, 80), (255, 255, 255, 0))
            draw = ImageDraw.Draw(temp)
            bbox = draw.textbbox((0, 0), char, font=font)
            char_width = bbox[2] - bbox[0]
            char_height = bbox[3] - bbox[1]
            ink = int(self.np_rng.integers(20, 90))
            draw.text(
                ((temp.width - char_width) / 2, (temp.height - char_height) / 2 - bbox[1]),
                char,
                font=font,
                fill=(ink, ink, ink, 255),
            )

            local_rotation = float(params["rotation"]) + (index - (len(text) - 1) / 2) * float(params["arc_angle"]) / 8.0
            temp = temp.rotate(
                local_rotation,
                resample=Image.Resampling.BICUBIC,
                expand=False,
                fillcolor=(255, 255, 255, 0),
            )

            x_center = int(round((index + 0.5) * slot_width))
            x_center += self.random.randint(-int(params["x_jitter"]), int(params["x_jitter"])) if int(params["x_jitter"]) else 0
            normalized_position = (index - (len(text) - 1) / 2) / max((len(text) - 1) / 2, 1.0)
            arc_y = math.sin(normalized_position * math.pi / 2.0) * float(params["arc_angle"]) * 0.35
            y_center = self.config.height // 2 + int(round(arc_y))
            y_center += self.random.randint(-int(params["y_jitter"]), int(params["y_jitter"])) if int(params["y_jitter"]) else 0
            paste_x = max(0, min(self.config.width - temp.width, x_center - temp.width // 2))
            paste_y = max(0, min(self.config.height - temp.height, y_center - temp.height // 2))
            char_layer.alpha_composite(temp, (paste_x, paste_y))

        base_image = Image.alpha_composite(canvas.convert("RGBA"), char_layer).convert("RGB")
        waved = apply_wave_distortion(
            base_image,
            amplitude=float(params["wave_amplitude"]),
            frequency=float(params["wave_frequency"]),
        )
        distorted = apply_common_distortions(
            waved,
            params,
            rng=self.random,
            np_rng=self.np_rng,
            translate_first=False,
        )
        return distorted, params


def translate_image(image: Image.Image, x_offset: int, y_offset: int) -> Image.Image:
    if x_offset == 0 and y_offset == 0:
        return image
    matrix = (1, 0, x_offset, 0, 1, y_offset)
    return image.transform(
        image.size,
        Image.Transform.AFFINE,
        matrix,
        resample=Image.Resampling.BICUBIC,
        fillcolor=(255, 255, 255),
    )


def apply_wave_distortion(image: Image.Image, *, amplitude: float, frequency: float) -> Image.Image:
    """Apply an ImageMagick-style horizontal sine-wave warp."""
    if abs(amplitude) < 1e-6:
        return image

    source = np.array(image.convert("RGB"), dtype=np.uint8)
    height, width = source.shape[:2]
    warped = np.full_like(source, 255)

    for y in range(height):
        shift = int(round(amplitude * math.sin(2.0 * math.pi * frequency * y / max(height, 1))))
        if shift > 0:
            warped[y, shift:, :] = source[y, : width - shift, :]
        elif shift < 0:
            warped[y, : width + shift, :] = source[y, -shift:, :]
        else:
            warped[y, :, :] = source[y, :, :]

    return Image.fromarray(warped, mode="RGB")


def apply_common_distortions(
    image: Image.Image,
    params: Dict[str, float | int],
    *,
    rng: random.Random,
    np_rng: np.random.Generator,
    translate_first: bool,
) -> Image.Image:
    """Post-process a base rendering with the shared auditable difficulty controls."""
    distorted = image.convert("RGB")

    if int(params["x_jitter"]) > 0 or int(params["y_jitter"]) > 0:
        x_shift = rng.randint(-int(params["x_jitter"]), int(params["x_jitter"])) if int(params["x_jitter"]) else 0
        y_shift = rng.randint(-int(params["y_jitter"]), int(params["y_jitter"])) if int(params["y_jitter"]) else 0
        if translate_first:
            distorted = translate_image(distorted, x_shift, y_shift)

    if abs(float(params["rotation"])) > 1e-6:
        distorted = distorted.rotate(
            float(params["rotation"]),
            resample=Image.Resampling.BICUBIC,
            expand=False,
            fillcolor=(255, 255, 255),
        )

    shear = float(params["shear"])
    if abs(shear) > 1e-6:
        x_shift = abs(shear) * distorted.height
        affine_matrix = (1, shear, -x_shift if shear > 0 else 0, 0, 1, 0)
        distorted = distorted.transform(
            distorted.size,
            Image.Transform.AFFINE,
            affine_matrix,
            resample=Image.Resampling.BICUBIC,
            fillcolor=(255, 255, 255),
        )

    if not translate_first and (int(params["x_jitter"]) > 0 or int(params["y_jitter"]) > 0):
        x_shift = rng.randint(-int(params["x_jitter"]), int(params["x_jitter"])) if int(params["x_jitter"]) else 0
        y_shift = rng.randint(-int(params["y_jitter"]), int(params["y_jitter"])) if int(params["y_jitter"]) else 0
        distorted = translate_image(distorted, x_shift, y_shift)

    draw = ImageDraw.Draw(distorted)
    for _ in range(int(params["line_count"])):
        start = (rng.randint(0, distorted.width), rng.randint(0, distorted.height))
        end = (rng.randint(0, distorted.width), rng.randint(0, distorted.height))
        color = tuple(int(v) for v in np_rng.integers(60, 180, size=3))
        thickness = rng.randint(1, 2)
        draw.line([start, end], fill=color, width=thickness)

    for _ in range(int(params["dot_count"])):
        x = rng.randint(0, distorted.width - 1)
        y = rng.randint(0, distorted.height - 1)
        color = tuple(int(v) for v in np_rng.integers(90, 220, size=3))
        draw.point((x, y), fill=color)

    if float(params["blur_radius"]) > 0:
        distorted = distorted.filter(ImageFilter.GaussianBlur(radius=float(params["blur_radius"])))

    noisy = np.array(distorted, dtype=np.float32)
    if float(params["noise_std"]) > 0:
        noise = np_rng.normal(0.0, float(params["noise_std"]), noisy.shape)
        noisy = np.clip(noisy + noise, 0, 255)
    return Image.fromarray(noisy.astype(np.uint8), mode="RGB")


def create_renderer(config: GeneratorDatasetConfig) -> BaseLocalRenderer:
    if config.backend == "pil_controlled":
        return ControlledPILRenderer(config)
    if config.backend == "captcha_image":
        return CaptchaPackageRenderer(config)
    if config.backend == "opencv_text":
        return OpenCVRenderer(config)
    if config.backend == "wand_text":
        return WandTextRenderer(config)
    raise ValueError(f"Unknown generator backend '{config.backend}'")
