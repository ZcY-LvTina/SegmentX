import os
from pathlib import Path
from typing import Iterable, List

import numpy as np
from PIL import Image

from ..config import SUPPORTED_EXTENSIONS


def load_image(path: str) -> Image.Image:
    image = Image.open(path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def image_to_array(image: Image.Image) -> np.ndarray:
    return np.array(image)


def save_image(image: Image.Image, path: str, format_hint: str | None = None) -> str:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    format_to_use = format_hint or output_path.suffix.replace(".", "").upper() or "PNG"
    image.save(output_path, format=format_to_use)
    return str(output_path)


def list_images(directory: str | Path, extensions: Iterable[str] = SUPPORTED_EXTENSIONS) -> List[str]:
    directory = Path(directory)
    normalized_exts = {ext.lower() for ext in extensions}
    return sorted(
        [
            str(directory / name)
            for name in os.listdir(directory)
            if (directory / name).is_file() and (directory / name).suffix.lower() in normalized_exts
        ]
    )


__all__ = ["load_image", "image_to_array", "save_image", "list_images"]
