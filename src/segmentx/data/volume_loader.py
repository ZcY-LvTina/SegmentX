"""Volume loader utilities for 3D data (e.g., PNG stack -> pseudo 3D)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image


def _natural_key(name: str) -> List:
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", name)]


@dataclass
class VolumeMetadata:
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    spacing_unknown: bool = True
    extra: dict = field(default_factory=dict)


@dataclass
class Volume3D:
    data: np.ndarray  # shape: [z, h, w]
    metadata: VolumeMetadata
    case_id: Optional[str] = None


class VolumeLoader:
    """Load a folder of PNGs (or compatible images) into a 3D volume."""

    def __init__(self, default_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> None:
        self.default_spacing = default_spacing

    def load_png_stack(self, folder: Path | str) -> Volume3D:
        folder_path = Path(folder)
        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        files = sorted(
            [p for p in folder_path.iterdir() if p.is_file() and p.suffix.lower() in [".png", ".jpg", ".jpeg"]],
            key=lambda p: _natural_key(p.name),
        )
        if not files:
            raise ValueError(f"No PNG/JPEG files found in {folder_path}")

        slices: List[np.ndarray] = []
        shape: Optional[Tuple[int, int]] = None
        for path in files:
            img = Image.open(path).convert("L")
            arr = np.array(img)
            if shape is None:
                shape = arr.shape
            elif arr.shape != shape:
                raise ValueError(f"Inconsistent slice shape in {path.name}, expected {shape}, got {arr.shape}")
            slices.append(arr)

        volume = np.stack(slices, axis=0)  # [z, h, w]
        metadata = VolumeMetadata(spacing=self.default_spacing, spacing_unknown=True)
        return Volume3D(data=volume, metadata=metadata, case_id=folder_path.name)
