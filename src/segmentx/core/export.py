from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
from PIL import Image

from ..core.session import ImageState
from ..utils.paths import ensure_dir


def _compose_masked_image(image: Image.Image, mask: np.ndarray) -> tuple[Image.Image, Image.Image]:
    """Compose result using mask as alpha without drawing hint overlays."""
    base = image.convert("RGBA")
    mask_image = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    result = base.copy()
    result.putalpha(mask_image)
    return result, mask_image


def save_masked_result(
    image_state: ImageState,
    output_path: Path,
    file_format: Optional[str] = None,
) -> Path:
    """Save masked image to disk."""
    if image_state.mask is None:
        raise ValueError("No mask available to save.")

    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    # Build image with segmentation as alpha; drop hint layer on save.
    result, mask_image = _compose_masked_image(image_state.original_image, image_state.mask)

    suffix = output_path.suffix.lower()
    if not suffix and file_format:
        suffix = f".{file_format.lower()}"
        output_path = output_path.with_suffix(suffix)

    format_to_use = file_format or (suffix[1:].upper() if suffix else "PNG")
    if format_to_use.upper() in ["JPEG", "JPG", "BMP"]:
        # Flatten alpha onto white background for formats without transparency.
        flattened = Image.new("RGB", result.size, (255, 255, 255))
        flattened.paste(result, mask=mask_image)
        result = flattened

    result.save(output_path, format=format_to_use)
    return output_path


def bulk_export(
    states: Iterable[ImageState],
    output_dir: Path,
    file_format: str = "PNG",
    extension: Optional[str] = None,
) -> List[Path]:
    output_dir = ensure_dir(Path(output_dir))
    exported: List[Path] = []
    for state in states:
        if state.mask is None:
            continue
        base_name = Path(state.path).stem
        suffix = extension or f".{file_format.lower()}"
        output_path = output_dir / f"{base_name}_segmented{suffix}"
        exported.append(save_masked_result(state, output_path, file_format=file_format))
    return exported


__all__ = ["save_masked_result", "bulk_export"]
