#!/usr/bin/env python3
"""
Preprocessing and dataset utilities for text CAPTCHA experiments.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    pd = None
    PANDAS_AVAILABLE = False


class CaptchaPreprocessor:
    """Normalize and optionally augment CAPTCHA images."""

    def __init__(
        self,
        target_size: Tuple[int, int] = (200, 80),
        normalize: bool = True,
        augment: bool = False,
    ):
        self.target_size = target_size
        self.normalize = normalize
        self.augment = augment

    def preprocess_image(self, image: np.ndarray | Image.Image, debug: bool = False) -> np.ndarray:
        """Convert an image into a model-ready grayscale tensor image."""
        if isinstance(image, Image.Image):
            image = np.array(image)

        if image.ndim == 3:
            if CV2_AVAILABLE:
                code = cv2.COLOR_BGR2GRAY if image.shape[2] == 3 else cv2.COLOR_BGRA2GRAY
                gray = cv2.cvtColor(image, code)
            else:
                gray = np.dot(image[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
        else:
            gray = image.astype(np.uint8)

        if CV2_AVAILABLE:
            resized = cv2.resize(gray, self.target_size, interpolation=cv2.INTER_AREA)
            blurred = cv2.GaussianBlur(resized, (3, 3), 0)
            normalized = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)
        else:
            pil_img = Image.fromarray(gray)
            pil_img = pil_img.resize(self.target_size, Image.Resampling.LANCZOS)
            normalized = np.array(pil_img, dtype=np.uint8)

        if float(np.mean(normalized)) > 127.0:
            normalized = 255 - normalized

        if self.augment:
            normalized = self._apply_augmentation(normalized)

        processed = normalized.astype(np.float32)
        if self.normalize:
            processed /= 255.0

        if debug:
            print(
                f"Preprocessed image: shape={processed.shape}, "
                f"min={processed.min():.3f}, max={processed.max():.3f}"
            )

        return processed

    def _apply_augmentation(self, image: np.ndarray) -> np.ndarray:
        """Small stochastic perturbations that keep CAPTCHA content legible."""
        if not CV2_AVAILABLE:
            return image

        augmented = image.copy()

        if np.random.random() < 0.5:
            angle = float(np.random.uniform(-7.0, 7.0))
            center = (augmented.shape[1] // 2, augmented.shape[0] // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            augmented = cv2.warpAffine(
                augmented,
                matrix,
                (augmented.shape[1], augmented.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )

        if np.random.random() < 0.4:
            noise = np.random.normal(0.0, 6.0, augmented.shape).astype(np.float32)
            augmented = np.clip(augmented.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        if np.random.random() < 0.3:
            kernel = np.ones((2, 2), np.uint8)
            augmented = cv2.morphologyEx(augmented, cv2.MORPH_CLOSE, kernel)

        return augmented


class CaptchaDataset(Dataset):
    """PyTorch dataset with resilient file resolution."""

    def __init__(
        self,
        data_dir: str,
        metadata_file: str,
        preprocessor: CaptchaPreprocessor,
        char_to_idx: Dict[str, int],
        max_length: int = 6,
        split: Optional[str] = None,
        debug: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.preprocessor = preprocessor
        self.char_to_idx = char_to_idx
        self.max_length = max_length
        self.debug = debug
        self.metadata = self._load_metadata(metadata_file)

        if split:
            self.metadata = [row for row in self.metadata if row.get("split") == split]

        self._validate_files()

    def _load_metadata(self, metadata_file: str) -> List[Dict[str, str]]:
        if PANDAS_AVAILABLE:
            df = pd.read_csv(metadata_file).fillna("")
            return df.to_dict(orient="records")

        with open(metadata_file, "r", newline="") as handle:
            reader = csv.DictReader(handle)
            return [dict(row) for row in reader]

    def _validate_files(self) -> None:
        valid_rows: List[Dict[str, str]] = []
        for row in self.metadata:
            candidate_paths = [
                self.data_dir / row["filename"],
                self.data_dir / row.get("split", "train") / row["filename"],
                self.data_dir.parent / row.get("split", "train") / row["filename"],
            ]

            for candidate in candidate_paths:
                if candidate.exists():
                    row["full_path"] = str(candidate)
                    valid_rows.append(row)
                    break
            else:
                if self.debug:
                    print(f"Warning: missing image for metadata row {row}")

        self.metadata = valid_rows

    def get_metadata_records(self) -> List[Dict[str, str]]:
        """Return metadata rows in dataset order."""
        return list(self.metadata)

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        row = self.metadata[idx]
        text = str(row["text"])
        image_path = Path(row["full_path"])

        try:
            if CV2_AVAILABLE:
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None:
                    image = np.array(Image.open(image_path))
            else:
                image = np.array(Image.open(image_path))

            processed = self.preprocessor.preprocess_image(image, debug=self.debug and idx < 2)
            image_tensor = torch.tensor(processed, dtype=torch.float32).unsqueeze(0)

            text_indices = [self.char_to_idx.get(char, self.char_to_idx.get("<UNK>", 1)) for char in text]
            text_tensor = torch.zeros(self.max_length, dtype=torch.long)
            text_length = min(len(text_indices), self.max_length)
            if text_length > 0:
                text_tensor[:text_length] = torch.tensor(text_indices[:text_length], dtype=torch.long)

            return image_tensor, text_tensor, text_length
        except Exception as exc:
            if self.debug:
                print(f"Failed to load sample {idx} from {image_path}: {exc}")
            dummy_image = torch.zeros(1, self.preprocessor.target_size[1], self.preprocessor.target_size[0])
            dummy_text = torch.zeros(self.max_length, dtype=torch.long)
            return dummy_image, dummy_text, 0


def infer_max_length(texts: Iterable[str]) -> int:
    """Infer the maximum sequence length from a corpus of texts."""
    return max((len(str(text)) for text in texts), default=0)


def create_char_mapping(
    texts: List[str],
    custom_chars: Optional[str] = None,
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Create stable vocabulary mappings."""
    if custom_chars:
        chars = sorted(set(custom_chars))
    else:
        charset = set()
        for text in texts:
            charset.update(str(text))
        chars = sorted(charset)

    char_to_idx = {"<PAD>": 0, "<UNK>": 1}
    idx_to_char = {0: "<PAD>", 1: "<UNK>"}

    for index, char in enumerate(chars, start=2):
        char_to_idx[char] = index
        idx_to_char[index] = char

    print(f"Created character mapping with {len(char_to_idx)} entries")
    print(f"Characters: {chars}")

    return char_to_idx, idx_to_char


def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, int]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate function that preserves sample order."""
    valid_batch = [(image, text, length) for image, text, length in batch if length > 0]

    if not valid_batch:
        return (
            torch.zeros(1, 1, 80, 200, dtype=torch.float32),
            torch.zeros(1, 6, dtype=torch.long),
            torch.ones(1, dtype=torch.long),
        )

    images, texts, lengths = zip(*valid_batch)
    return torch.stack(images), torch.stack(texts), torch.tensor(lengths, dtype=torch.long)


def main() -> None:
    """Minimal preprocessing smoke test."""
    random_image = np.random.randint(0, 255, size=(80, 200, 3), dtype=np.uint8)
    preprocessor = CaptchaPreprocessor(target_size=(200, 80), normalize=True, augment=True)
    processed = preprocessor.preprocess_image(random_image, debug=True)
    print(f"Processed image ready with shape {processed.shape}")


if __name__ == "__main__":
    main()
