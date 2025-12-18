"""Model registry for SegmentX.

This module keeps the models/ directory as a local cache, supports manifest
validation, installation from zip/directories, importing native nnU-Net
artifacts, and listing/removing models for the UI layer.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple
from zipfile import ZipFile

from ..config import MODELS_DIR, MODELS_DOWNLOADS_DIR, MODELS_SOURCES_FILE
from ..utils.paths import ensure_dir

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None


REQUIRED_FIELDS = ["id", "name", "type", "capabilities", "input_format", "version", "source", "entry"]


class RegistryError(Exception):
    """Raised for registry level failures."""


@dataclass
class ModelManifest:
    id: str
    name: str
    type: str
    capabilities: List[str]
    input_format: str
    version: str
    source: str
    entry: str
    labels: Optional[List[str]] = None
    extra: Dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict) -> "ModelManifest":
        missing = [field for field in REQUIRED_FIELDS if field not in data]
        if missing:
            raise RegistryError(f"manifest missing fields: {', '.join(missing)}")
        return cls(
            id=str(data["id"]),
            name=str(data["name"]),
            type=str(data["type"]),
            capabilities=list(data.get("capabilities", [])),
            input_format=str(data["input_format"]),
            version=str(data["version"]),
            source=str(data["source"]),
            entry=str(data["entry"]),
            labels=data.get("labels"),
            extra={k: v for k, v in data.items() if k not in REQUIRED_FIELDS + ["labels"]},
        )

    def to_dict(self) -> Dict:
        payload = {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "capabilities": self.capabilities,
            "input_format": self.input_format,
            "version": self.version,
            "source": self.source,
            "entry": self.entry,
        }
        if self.labels is not None:
            payload["labels"] = self.labels
        payload.update(self.extra)
        return payload


@dataclass
class ModelRecord:
    """Scan result for a single model folder."""

    manifest: Optional[ModelManifest]
    path: Path
    error: Optional[str] = None
    payload_path: Optional[Path] = None

    @property
    def id(self) -> Optional[str]:
        return self.manifest.id if self.manifest else None

    @property
    def is_valid(self) -> bool:
        return self.manifest is not None and self.error is None


class ModelRegistry:
    """Registry to manage installation, listing, and removal of models."""

    def __init__(self, models_dir: Path | str = MODELS_DIR) -> None:
        self.models_dir = ensure_dir(Path(models_dir))
        self._records: Dict[str, ModelRecord] = {}
        self.refresh()

    # Public API -------------------------------------------------------------
    def list_models(self) -> List[ModelRecord]:
        """Return current models with validation state."""
        return list(self._records.values())

    def get_model(self, model_id: str) -> Optional[ModelRecord]:
        return self._records.get(model_id)

    def refresh(self) -> None:
        """Rescan models directory and rebuild cache."""
        self._records = {}
        for record in self._scan_models():
            key = record.id or record.path.name
            self._records[key] = record

    def remove(self, model_id: str) -> bool:
        record = self._records.get(model_id)
        if not record:
            return False
        try:
            shutil.rmtree(record.path)
        except Exception as exc:
            raise RegistryError(f"failed to remove model {model_id}: {exc}") from exc
        self._records.pop(model_id, None)
        return True

    def install_from_zip(self, zip_path: Path | str) -> ModelManifest:
        """Install a SegmentX model package zip that contains manifest.yaml."""
        zip_path = Path(zip_path)
        if not zip_path.exists():
            raise RegistryError(f"zip not found: {zip_path}")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            with ZipFile(zip_path, "r") as zf:
                zf.extractall(tmpdir_path)
            manifest_path = self._find_manifest(tmpdir_path)
            if manifest_path is None:
                # treat as native nnU-Net archive
                raise RegistryError("manifest.yaml not found in zip; use import_nnunet_native() instead")
            manifest = self._load_manifest(manifest_path)
            self._materialize_model_dir(manifest_path.parent, manifest.id)
            self.refresh()
            return manifest

    def install_from_dir(self, directory: Path | str) -> ModelManifest:
        directory = Path(directory)
        manifest_path = self._find_manifest(directory)
        if manifest_path is None:
            raise RegistryError("manifest.yaml not found; use import_nnunet_native() for raw nnU-Net exports")
        manifest = self._load_manifest(manifest_path)
        self._materialize_model_dir(manifest_path.parent, manifest.id)
        self.refresh()
        return manifest

    def import_nnunet_native(
        self,
        path_or_zip: Path | str,
        model_id: Optional[str] = None,
        meta: Optional[Dict] = None,
    ) -> ModelManifest:
        """Import a raw nnU-Net result zip or directory and wrap with manifest."""
        source_meta = meta or {}
        path = Path(path_or_zip)
        if not path.exists():
            raise RegistryError(f"nnU-Net artifact not found: {path}")

        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            extracted_dir = workdir / "extracted"
            extracted_dir.mkdir(parents=True, exist_ok=True)
            payload_dir: Path
            if path.is_file() and path.suffix.lower() == ".zip":
                with ZipFile(path, "r") as zf:
                    zf.extractall(extracted_dir)
                payload_dir = self._detect_payload_root(extracted_dir)
            else:
                payload_dir = self._detect_payload_root(path)

            model_identifier = model_id or source_meta.get("id") or payload_dir.name
            target_dir = ensure_dir(self.models_dir / model_identifier)
            final_payload = ensure_dir(target_dir / "payload")
            self._copy_payload(payload_dir, final_payload)

            manifest = ModelManifest(
                id=model_identifier,
                name=source_meta.get("name", model_identifier),
                type="nnunet",
                capabilities=source_meta.get("capabilities", ["3d"]),
                input_format=source_meta.get("input_format", "nifti"),
                version=str(source_meta.get("version", "1.0.0")),
                source=source_meta.get("source", "imported:nnunet"),
                entry=str(Path("payload")),
                labels=source_meta.get("labels"),
                extra={"native": True},
            )
            self._write_manifest(manifest, target_dir / "manifest.yaml")
            self.refresh()
            return manifest

    # Internal helpers ------------------------------------------------------
    def _scan_models(self) -> Iterable[ModelRecord]:
        ignored = {"_downloads", "_cache"}
        for child in sorted(self.models_dir.iterdir()):
            if child.name in ignored:
                continue
            if not child.is_dir():
                continue
            manifest_path = self._find_manifest(child)
            if manifest_path is None:
                # Skip silently but keep placeholder error
                yield ModelRecord(manifest=None, path=child, error="manifest missing")
                continue
            try:
                manifest = self._load_manifest(manifest_path)
                payload_path = child / manifest.entry
                if not payload_path.exists():
                    raise RegistryError(f"payload missing at {payload_path}")
                yield ModelRecord(manifest=manifest, path=child, payload_path=payload_path)
            except Exception as exc:
                yield ModelRecord(manifest=None, path=child, error=str(exc))

    def _find_manifest(self, directory: Path) -> Optional[Path]:
        for name in ["manifest.yaml", "manifest.json"]:
            candidate = directory / name
            if candidate.exists():
                return candidate
        for name in ["manifest.yaml", "manifest.json"]:
            for path in directory.rglob(name):
                return path
        return None

    def _load_manifest(self, path: Path) -> ModelManifest:
        if not path.exists():
            raise RegistryError(f"manifest not found: {path}")
        data = self._load_yaml_or_json(path)
        return ModelManifest.from_dict(data)

    def _load_yaml_or_json(self, path: Path) -> Dict:
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() in [".json"]:
            return json.loads(text)
        if yaml is None:
            try:
                # Allow YAML files that are actually JSON content
                return json.loads(text)
            except Exception:
                raise RegistryError(
                    f"pyyaml is required to read YAML file {path}. Install pyyaml or provide JSON."
                )
        return yaml.safe_load(text)

    def _write_manifest(self, manifest: ModelManifest, path: Path) -> None:
        ensure_dir(path.parent)
        if yaml is None:
            # Fall back to JSON when yaml is unavailable
            path = path.with_suffix(".json")
            path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
            return
        path.write_text(yaml.safe_dump(manifest.to_dict(), allow_unicode=True, sort_keys=False), encoding="utf-8")

    def _materialize_model_dir(self, src_dir: Path, model_id: str) -> None:
        """Copy extracted model package into registry folder."""
        target_dir = self.models_dir / model_id
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.copytree(src_dir, target_dir, dirs_exist_ok=True)

    def _detect_payload_root(self, directory: Path) -> Path:
        """Guess payload root for native nnU-Net exports."""
        directory = directory.resolve()
        if (directory / "plans.json").exists() or (directory / "nnUNetPlans.json").exists():
            return directory
        subdirs = [p for p in directory.iterdir() if p.is_dir()]
        for child in subdirs:
            if (child / "plans.json").exists() or (child / "nnUNetPlans.json").exists():
                return child
        if subdirs:
            return subdirs[0]
        return directory

    def _copy_payload(self, src: Path, dst: Path) -> None:
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst, dirs_exist_ok=True)

    # Sources handling ------------------------------------------------------
    def list_default_sources(self) -> List[Dict]:
        """Load sources.yaml/json if present."""
        candidates = [
            MODELS_SOURCES_FILE,
            MODELS_SOURCES_FILE.with_suffix(".json"),
            MODELS_DIR / "sources.json",
        ]
        last_error: Optional[str] = None
        for path in candidates:
            if not path.exists():
                continue
            try:
                data = self._load_yaml_or_json(path)
                entries = data.get("models", []) if isinstance(data, dict) else data
                return [entry for entry in entries if isinstance(entry, dict)]
            except Exception as exc:
                last_error = str(exc)
                continue
        if last_error:
            raise RegistryError(
                "failed to read sources file (yaml/json). Please install pyyaml or provide model_store/sources.json. "
                f"Details: {last_error}"
            )
        return []

    def download_and_install_default(
        self,
        model_id: str,
        downloader: Callable[[str, Path], Path],
    ) -> Tuple[Optional[ModelManifest], Optional[str]]:
        """Download a default model zip from sources.yaml and install it.

        Returns (manifest, error_msg).
        """
        sources = {entry.get("model_id") or entry.get("id"): entry for entry in self.list_default_sources()}
        entry = sources.get(model_id)
        if not entry:
            return None, f"model_id {model_id} not found in sources list"

        url = entry.get("url")
        if not url:
            return None, f"model_id {model_id} missing url"
        dest = MODELS_DOWNLOADS_DIR / f"{model_id}.zip"

        try:
            downloader(url, dest)
        except Exception as exc:
            return None, f"download failed: {exc}"

        sha256_expected = entry.get("sha256")
        if sha256_expected:
            actual = self._calc_sha256(dest)
            if actual.lower() != str(sha256_expected).lower():
                return None, f"sha256 mismatch: expected {sha256_expected}, got {actual}"

        try:
            manifest = self.install_from_zip(dest)
            return manifest, None
        except Exception as exc:
            # fallback to nnU-Net import if needed
            if entry.get("type") == "nnunet":
                try:
                    manifest = self.import_nnunet_native(dest, model_id=model_id, meta=entry)
                    return manifest, None
                except Exception as inner:
                    return None, f"install failed: {inner}"
            return None, f"install failed: {exc}"

    def _calc_sha256(self, path: Path, chunk_size: int = 1024 * 1024) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            while True:
                chunk = fh.read(chunk_size)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()
