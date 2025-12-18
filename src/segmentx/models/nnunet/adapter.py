"""nnU-Net inference adapter: wraps model package + manifest into runnable predict call."""

from __future__ import annotations

import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from ...config import MODELS_CACHE_DIR
from ...data.exporters import NNUNetExporter
from ...data.volume_loader import Volume3D
from ...utils.paths import ensure_dir
from ..registry import ModelManifest, ModelRecord, ModelRegistry, RegistryError

try:
    import nibabel as nib  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    nib = None

LogCallback = Callable[[str], None]
ProgressCallback = Callable[[str], None]


@dataclass
class NNUNetPrediction:
    case_id: str
    output_file: Path
    mask_volume: Optional[np.ndarray] = None


class NNUNetAdapter:
    """Adapter to run nnU-Net predict command against a registered model."""

    def __init__(self, registry: ModelRegistry, predict_cmd: str = "nnUNetv2_predict") -> None:
        self.registry = registry
        self.predict_cmd = predict_cmd
        self.cache_dir = ensure_dir(Path(MODELS_CACHE_DIR) / "nnunet")

    def predict(
        self,
        model_id: str,
        volume: Volume3D,
        case_id: str,
        progress_cb: Optional[ProgressCallback] = None,
        log_cb: Optional[LogCallback] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> NNUNetPrediction:
        record = self._require_model(model_id)
        payload_path = self._resolve_payload(record)
        predict_cmd = shutil.which(self.predict_cmd) or self.predict_cmd
        if shutil.which(predict_cmd) is None:
            raise RegistryError(
                f"nnU-Net predict command '{self.predict_cmd}' not found. Configure the path in settings."
            )

        case_cache = ensure_dir(self.cache_dir / case_id)
        exporter = NNUNetExporter(case_cache)
        if progress_cb:
            progress_cb("正在导出输入数据为 NIfTI...")
        export_result = exporter.export_inference_volume(volume, case_id)

        output_dir = ensure_dir(case_cache / "predictions")
        cmd = [
            predict_cmd,
            "-i",
            str(export_result.image_path.parent),
            "-o",
            str(output_dir),
            "--model_path",
            str(payload_path),
        ]

        if progress_cb:
            progress_cb("正在运行 nnU-Net 推理...")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert process.stdout is not None
        for line in process.stdout:
            if cancel_event and cancel_event.is_set():
                process.terminate()
                raise RegistryError("nnU-Net 推理已被取消")
            if log_cb:
                log_cb(line.rstrip("\n"))
        code = process.wait()
        if code != 0:
            raise RegistryError(f"nnU-Net 推理失败，退出码 {code}")
        if progress_cb:
            progress_cb("推理完成，正在读取结果...")

        prediction_file = self._find_prediction_file(output_dir, case_id)
        mask_volume = self._load_mask(prediction_file)
        return NNUNetPrediction(case_id=case_id, output_file=prediction_file, mask_volume=mask_volume)

    def _require_model(self, model_id: str) -> ModelRecord:
        record = self.registry.get_model(model_id)
        if not record:
            raise RegistryError(f"model {model_id} not found in registry")
        if not record.is_valid:
            raise RegistryError(f"model {model_id} is invalid: {record.error}")
        assert record.manifest
        if record.manifest.type != "nnunet":
            raise RegistryError(f"model {model_id} is type {record.manifest.type}, expected nnunet")
        return record

    def _resolve_payload(self, record: ModelRecord) -> Path:
        if record.payload_path:
            return record.payload_path
        if record.manifest:
            payload = record.path / record.manifest.entry
            if payload.exists():
                return payload
        raise RegistryError(f"payload not found for model {record.id}")

    def _find_prediction_file(self, output_dir: Path, case_id: str) -> Path:
        candidates = list(output_dir.glob(f"{case_id}*.nii*"))
        if not candidates:
            raise RegistryError(f"prediction file for {case_id} not found in {output_dir}")
        return candidates[0]

    def _load_mask(self, file_path: Path) -> Optional[np.ndarray]:
        if nib is None:  # pragma: no cover - optional dependency
            return None
        img = nib.load(str(file_path))
        data = img.get_fdata()
        # Convert to integer labelmap
        return data.astype(np.int32)
