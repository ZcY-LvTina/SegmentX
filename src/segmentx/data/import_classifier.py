from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Tuple

import numpy as np
from PIL import Image

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
VOLUME_EXTS = {".nii", ".nii.gz", ".nrrd", ".nhdr", ".mha", ".mhd"}

PngDecision = Literal["auto_volume", "auto_images", "ambiguous"]
ResourceType = Literal["image2d", "series3d", "volume3d", "dicom"]


def _stable_id(label: str, paths: Iterable[str]) -> str:
    joined = "|".join(sorted(str(p) for p in paths))
    digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()[:8]
    safe_label = re.sub(r"[^a-zA-Z0-9_-]", "_", label) or "item"
    return f"{safe_label}-{digest}"


@dataclass
class ResourceItem:
    id: str
    type: ResourceType
    paths: List[str]
    name: str
    meta: Dict


@dataclass
class PngSeriesGroup:
    parent: Path
    prefix: str
    ext: str
    paths: List[Path]
    numbers: List[int]
    size: Tuple[int, int]
    decision: PngDecision
    scores: Dict[str, float] = field(default_factory=dict)
    consistent_shape: bool = True

    def decision_key(self) -> str:
        return f"{self.parent.resolve()}::{self.prefix}"

    def as_series_resource(self) -> ResourceItem:
        rid = _stable_id(self.prefix or self.parent.name, [str(p) for p in self.paths])
        meta = {
            "sliceCount": len(self.paths),
            "prefix": self.prefix,
            "parent": str(self.parent),
            "ext": self.ext,
            "numbers": [min(self.numbers), max(self.numbers)],
            "size": self.size,
            "scores": self.scores,
            "source": "auto" if self.decision != "ambiguous" else "user",
        }
        return ResourceItem(
            id=rid,
            type="series3d",
            paths=[str(p) for p in sorted(self.paths)],
            name=f"{self.prefix or self.parent.name} ({len(self.paths)} slices)",
            meta=meta,
        )

    def as_image_resources(self) -> List[ResourceItem]:
        items: List[ResourceItem] = []
        for path in sorted(self.paths):
            meta = {
                "parent": str(self.parent),
                "prefix": self.prefix,
                "source": "from_series",
            }
            items.append(
                ResourceItem(
                    id=_stable_id(path.stem, [str(path)]),
                    type="image2d",
                    paths=[str(path)],
                    name=path.name,
                    meta=meta,
                )
            )
        return items


@dataclass
class ClassificationResult:
    images2d: List[ResourceItem] = field(default_factory=list)
    series3d: List[ResourceItem] = field(default_factory=list)
    volumes3d: List[ResourceItem] = field(default_factory=list)
    dicom: List[ResourceItem] = field(default_factory=list)
    ambiguous: List[PngSeriesGroup] = field(default_factory=list)

    def all_resources(self) -> List[ResourceItem]:
        return self.images2d + self.series3d + self.volumes3d + self.dicom


class ImportPreferences:
    """Persistent cache for user choices on ambiguous imports."""

    def __init__(self, path: Optional[Path]) -> None:
        self.path = path
        self._data: Dict[str, str] = {}
        self._loaded = False
        if self.path:
            self._load()

    def _load(self) -> None:
        if self._loaded:
            return
        if self.path and self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                self._data = {}
        self._loaded = True

    def get(self, key: str) -> Optional[str]:
        self._load()
        return self._data.get(key)

    def remember(self, key: str, choice: str) -> None:
        if not self.path:
            return
        self._load()
        self._data[key] = choice
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")


def _has_ext(path: Path, exts: Iterable[str]) -> bool:
    name = path.name.lower()
    return any(name.endswith(ext) for ext in exts)


def _matched_ext(path: Path, exts: Iterable[str]) -> str:
    name = path.name.lower()
    for ext in exts:
        if name.endswith(ext):
            return ext
    return path.suffix.lower()


def _group_by_prefix(paths: List[Path]) -> Dict[Tuple[Path, str, str], List[Tuple[int, Path]]]:
    pattern = re.compile(r"^(.*?)(\d{2,})(\.[^.]+)$")
    groups: Dict[Tuple[Path, str, str], List[Tuple[int, Path]]] = {}
    for path in paths:
        match = pattern.match(path.name)
        if not match:
            continue
        prefix, number, ext = match.group(1), match.group(2), match.group(3)
        key = (path.parent, prefix, ext.lower())
        groups.setdefault(key, []).append((int(number), path))
    return groups


def _compute_smoothness_score(paths: List[Path], limit: int = 24, thumb: int = 64) -> Optional[float]:
    if len(paths) < 2:
        return None
    samples = sorted(paths)[:limit]
    arrays: List[np.ndarray] = []
    resample = getattr(Image, "Resampling", Image).LANCZOS  # Pillow<9 fallback
    for p in samples:
        with Image.open(p) as img:
            img = img.convert("L")
            img = img.resize((thumb, thumb), resample)
            arrays.append(np.asarray(img, dtype=np.float32))
    diffs: List[float] = []
    for a, b in zip(arrays, arrays[1:]):
        diffs.append(float(np.mean(np.abs(a - b))))
    if not diffs:
        return None
    return float(np.median(diffs))


def detect_png_series(
    paths: List[Path],
    enable_smooth_check: bool = True,
) -> List[PngSeriesGroup]:
    """Detect PNG/JPEG stacks that look like 3D series."""
    groups: List[PngSeriesGroup] = []
    grouped = _group_by_prefix(paths)
    for (parent, prefix, ext), numbered in grouped.items():
        numbered = sorted(numbered, key=lambda x: x[0])
        if len(numbered) < 3:
            continue

        indices = [n for n, _ in numbered]
        sorted_paths = [p for _, p in numbered]
        sizes = []
        consistent_shape = True
        for p in sorted_paths:
            with Image.open(p) as img:
                sizes.append((img.width, img.height, img.mode))
        base_size = sizes[0]
        for size in sizes[1:]:
            if size != base_size:
                consistent_shape = False
                break

        contiguous = 0
        for a, b in zip(indices, indices[1:]):
            if b - a == 1:
                contiguous += 1
        contiguous_ratio = contiguous / max(len(indices) - 1, 1)
        slice_count = len(indices)

        smooth_score = None
        if enable_smooth_check:
            smooth_score = _compute_smoothness_score(sorted_paths)

        decision: PngDecision
        scores = {
            "contiguous_ratio": contiguous_ratio,
            "slice_count": slice_count,
            "smooth_median_mad": smooth_score if smooth_score is not None else -1,
        }
        if not consistent_shape:
            decision = "auto_images"
        else:
            score = 0.0
            if contiguous_ratio >= 0.8:
                score += 1.0
            if slice_count >= 16:
                score += 1.0
            if smooth_score is not None and smooth_score <= 10.0:
                score += 1.0

            if contiguous_ratio < 0.5 and slice_count < 8:
                decision = "auto_images"
            elif score >= 2.0:
                decision = "auto_volume"
            elif contiguous_ratio < 0.5:
                decision = "auto_images"
            else:
                decision = "ambiguous"

        groups.append(
            PngSeriesGroup(
                parent=parent,
                prefix=prefix,
                ext=ext,
                paths=sorted_paths,
                numbers=indices,
                size=(base_size[0], base_size[1]),
                decision=decision,
                scores=scores,
                consistent_shape=consistent_shape,
            )
        )
    return groups


def _is_dicom_file(path: Path) -> bool:
    if path.suffix.lower() == ".dcm":
        return True
    try:
        with path.open("rb") as fh:
            header = fh.read(132 + 4)
            return len(header) >= 132 + 4 and header[128:132] == b"DICM"
    except Exception:
        return False


def _looks_like_dicom_dir(directory: Path, max_scan: int = 64) -> bool:
    scanned = 0
    for file in directory.rglob("*"):
        if scanned >= max_scan:
            break
        if not file.is_file():
            continue
        scanned += 1
        if _is_dicom_file(file):
            return True
    return False


class ImportClassifier:
    """Classify import paths into images/series/volumes/dicom."""

    def __init__(
        self,
        preferences: Optional[ImportPreferences] = None,
        enable_smooth_check: bool = True,
    ) -> None:
        self.preferences = preferences or ImportPreferences(None)
        self.enable_smooth_check = enable_smooth_check

    def classify(self, paths: Iterable[str]) -> ClassificationResult:
        result = ClassificationResult()
        image_candidates: List[Path] = []
        processed: set[Path] = set()

        for raw in paths:
            path = Path(raw)
            if path in processed:
                continue
            processed.add(path)
            if path.is_dir():
                if _looks_like_dicom_dir(path):
                    rid = _stable_id(path.name or "dicom", [str(path)])
                    result.dicom.append(
                        ResourceItem(
                            id=rid,
                            type="dicom",
                            paths=[str(path)],
                            name=path.name or "dicom_series",
                            meta={"path": str(path), "is3d": True},
                        )
                    )
                    continue
                for file in sorted(path.iterdir()):
                    if file.is_file() and _has_ext(file, IMAGE_EXTS):
                        image_candidates.append(file)
                    elif file.is_file() and _has_ext(file, VOLUME_EXTS):
                        fmt = _matched_ext(file, VOLUME_EXTS)
                        result.volumes3d.append(
                            ResourceItem(
                                id=_stable_id(file.stem, [str(file)]),
                                type="volume3d",
                                paths=[str(file)],
                                name=file.name,
                                meta={"format": fmt, "is3d": True, "path": str(file)},
                            )
                        )
            elif path.is_file():
                suffix = path.suffix.lower()
                if _has_ext(path, IMAGE_EXTS):
                    image_candidates.append(path)
                elif _has_ext(path, VOLUME_EXTS):
                    fmt = _matched_ext(path, VOLUME_EXTS)
                    result.volumes3d.append(
                        ResourceItem(
                            id=_stable_id(path.stem, [str(path)]),
                            type="volume3d",
                            paths=[str(path)],
                            name=path.name,
                            meta={"format": fmt, "is3d": True, "path": str(path)},
                        )
                    )

        groups = detect_png_series(image_candidates, enable_smooth_check=self.enable_smooth_check)
        grouped_paths = {p for g in groups for p in g.paths}
        remaining_images = [p for p in image_candidates if p not in grouped_paths]
        for path in sorted(remaining_images):
            result.images2d.append(
                ResourceItem(
                    id=_stable_id(path.stem, [str(path)]),
                    type="image2d",
                    paths=[str(path)],
                    name=path.name,
                    meta={"path": str(path)},
                )
            )

        for group in groups:
            pref = self.preferences.get(group.decision_key())
            decision = group.decision
            if pref == "volume":
                decision = "auto_volume"
            elif pref == "images":
                decision = "auto_images"

            if decision == "auto_volume":
                result.series3d.append(group.as_series_resource())
            elif decision == "auto_images":
                result.images2d.extend(group.as_image_resources())
            else:
                result.ambiguous.append(group)

        return result
