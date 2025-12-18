"""Minimal self-checks for SegmentX model plumbing.

Run with: `python scripts/self_check.py`
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

try:
    import numpy as np
    from PIL import Image
except ImportError as exc:  # pragma: no cover - optional runtime deps
    print(f"[self-check] 缺少依赖，请先安装 numpy 和 pillow：{exc}")
    sys.exit(1)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from segmentx.data.volume_loader import VolumeLoader  # noqa: E402
from segmentx.models.registry import ModelRegistry  # noqa: E402


def _make_png_stack(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for name, value in [("1.png", 1), ("10.png", 10), ("2.png", 2)]:
        img = Image.fromarray(np.full((4, 4), value, dtype=np.uint8))
        img.save(folder / name)


def test_volume_loader(tmpdir: Path) -> None:
    stack_dir = tmpdir / "stack"
    _make_png_stack(stack_dir)
    loader = VolumeLoader()
    vol = loader.load_png_stack(stack_dir)
    assert list(vol.data[:, 0, 0]) == [1, 2, 10], "Natural sorting failed for PNG stack"


def test_registry_install_remove(tmpdir: Path) -> None:
    models_root = tmpdir / "models"
    registry = ModelRegistry(models_dir=models_root)
    model_dir = models_root / "demo_model"
    payload = model_dir / "payload"
    payload.mkdir(parents=True, exist_ok=True)
    (payload / "weights.txt").write_text("demo", encoding="utf-8")
    manifest = {
        "id": "demo_model",
        "name": "Demo",
        "type": "sam",
        "capabilities": ["2d"],
        "input_format": "image",
        "version": "0.1.0",
        "source": "self-check",
        "entry": "payload",
    }
    (model_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    registry.refresh()
    assert registry.get_model("demo_model"), "Model not discovered after refresh"
    registry.remove("demo_model")
    assert not (models_root / "demo_model").exists(), "Model folder not removed"


def test_import_nnunet(tmpdir: Path) -> None:
    models_root = tmpdir / "models"
    registry = ModelRegistry(models_dir=models_root)
    native_dir = tmpdir / "nnunet_result"
    (native_dir / "plans.json").parent.mkdir(parents=True, exist_ok=True)
    (native_dir / "plans.json").write_text("{}", encoding="utf-8")
    (native_dir / "model_final_checkpoint.model").write_text("ckpt", encoding="utf-8")
    manifest = registry.import_nnunet_native(native_dir, model_id="nnunet_check")
    assert manifest.id == "nnunet_check"
    assert (models_root / "nnunet_check" / "payload").exists()


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        test_volume_loader(tmpdir)
        test_registry_install_remove(tmpdir)
        test_import_nnunet(tmpdir)
        print("Self-check passed: registry + volume loader + nnU-Net import.")


if __name__ == "__main__":
    main()
