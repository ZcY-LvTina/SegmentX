from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Protocol

try:
    from PIL import Image
except ImportError:  # pragma: no cover - optional dependency for test envs
    Image = None


class SliceProvider(Protocol):
    """Minimal slice provider interface for volume viewing."""

    id: str
    depth: int
    width: int
    height: int

    def get_slice(self, index: int) -> Image.Image:
        ...


def _extract_trailing_number(name: str) -> int:
    match = re.search(r"(\d+)(?!.*\d)", name)
    if not match:
        return -1
    try:
        return int(match.group(1))
    except Exception:
        return -1


def sort_paths_by_number(paths: List[Path]) -> List[Path]:
    """Stable sort: numeric suffix first; items without numbers go to the end."""

    def key(p: Path):
        stem = p.stem
        num_match = re.search(r"(\d+)(?!.*\d)", stem)
        if not num_match:
            return (3, stem.lower())
        num_str = num_match.group(1)
        num = int(num_str)
        has_leading_zero = len(num_str) > 1 and num_str.startswith("0")
        is_plain_numeric = stem.isdigit()
        if has_leading_zero:
            category = 0
        elif is_plain_numeric:
            category = 1
        else:
            category = 2
        return (category, num, stem.lower())

    return sorted(paths, key=key)


@dataclass
class PngSeriesProvider:
    """Slice provider for a PNG/JPEG stack."""

    id: str
    files: List[Path]

    def __post_init__(self) -> None:
        if not self.files:
            raise ValueError("PngSeriesProvider requires at least one file")
        self.files = sort_paths_by_number(self.files)
        if Image is None:
            raise RuntimeError("Pillow is required for PngSeriesProvider")
        first = Image.open(self.files[0]).convert("RGB")
        self.width, self.height = first.size
        first.close()
        self.depth = len(self.files)
        self.meta = {"isPseudo3D": True}

    def get_slice(self, index: int) -> Image.Image:
        if Image is None:
            raise RuntimeError("Pillow is required for PngSeriesProvider")
        if index < 0 or index >= self.depth:
            raise IndexError(f"slice {index} out of range")
        return Image.open(self.files[index]).convert("RGB")
