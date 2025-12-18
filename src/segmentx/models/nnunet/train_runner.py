"""nnU-Net training runner (non-blocking wrapper)."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional

from ...config import MODELS_DIR
from ...utils.paths import ensure_dir
from ..registry import ModelManifest, ModelRegistry, RegistryError

LogCallback = Callable[[str], None]


@dataclass
class TrainConfig:
    dataset_id: str
    configuration: str = "3d_fullres"
    folds: str = "all"
    trainer: str = "nnUNetTrainer"
    device: Optional[str] = None
    fast: bool = False
    results_dir: Optional[Path] = None
    target_model_id: Optional[str] = None
    meta: Dict = field(default_factory=dict)


class TrainRunner:
    """Kick off nnU-Net training without blocking the UI thread."""

    def __init__(self, registry: ModelRegistry, train_cmd: str = "nnUNetv2_train") -> None:
        self.registry = registry
        self.train_cmd = train_cmd

    def run(
        self,
        config: TrainConfig,
        log_cb: Optional[LogCallback] = None,
    ) -> Optional[ModelManifest]:
        """Start training and package results into registry."""
        train_exe = shutil.which(self.train_cmd) or self.train_cmd
        if shutil.which(train_exe) is None:
            raise RegistryError(f"训练命令 '{self.train_cmd}' 未找到，请在设置中配置正确路径。")

        cmd = [
            train_exe,
            config.dataset_id,
            config.configuration,
            config.folds,
            "-tr",
            config.trainer,
        ]
        if config.device:
            cmd.extend(["-d", config.device])
        if config.fast:
            cmd.append("--npz")

        env = os.environ.copy()
        if config.results_dir:
            env["nnUNet_results"] = str(config.results_dir)

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            if log_cb:
                log_cb(line.rstrip("\n"))
        code = process.wait()
        if code != 0:
            raise RegistryError(f"nnU-Net 训练失败，退出码 {code}")

        # Attempt to package training output
        results_root = Path(env.get("nnUNet_results", Path.home() / "nnUNet_results"))
        payload = self._find_result_dir(results_root, config.dataset_id, config.configuration, config.trainer)
        if not payload:
            if log_cb:
                log_cb("未找到训练输出目录，跳过模型打包。")
            return None

        model_id = config.target_model_id or f"{config.dataset_id}_{config.configuration}"
        target_dir = ensure_dir(MODELS_DIR / model_id)
        payload_dir = ensure_dir(target_dir / "payload")
        shutil.copytree(payload, payload_dir, dirs_exist_ok=True)

        manifest = ModelManifest(
            id=model_id,
            name=config.meta.get("name", model_id),
            type="nnunet",
            capabilities=config.meta.get("capabilities", ["3d"]),
            input_format=config.meta.get("input_format", "nifti"),
            version=str(config.meta.get("version", "1.0.0")),
            source=config.meta.get("source", "train:nnunet"),
            entry="payload",
            labels=config.meta.get("labels"),
        )
        manifest_path = target_dir / "manifest.yaml"
        self.registry._write_manifest(manifest, manifest_path)  # use registry to keep format consistent
        self.registry.refresh()
        return manifest

    def _find_result_dir(
        self, results_root: Path, dataset_id: str, configuration: str, trainer: str
    ) -> Optional[Path]:
        candidates = list(results_root.glob(f"**/{dataset_id}/{configuration}/{trainer}"))
        if candidates:
            # Prefer the deepest path (latest fold)
            candidates.sort(key=lambda p: len(p.parts), reverse=True)
            return candidates[0]
        return None
