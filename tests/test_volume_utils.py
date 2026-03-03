import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from PIL import Image
except ImportError:  # pragma: no cover - optional dependency
    Image = None

from segmentx.volume.cache import SliceCache
from segmentx.volume.provider import PngSeriesProvider, sort_paths_by_number


def _make_image(path: Path, size=(8, 8), color=(255, 0, 0)) -> None:
    if Image is None:
        raise RuntimeError("Pillow required for this test")
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color)
    img.save(path)


class _DummyProvider:
    def __init__(self, depth: int) -> None:
        self.depth = depth
        self.width = 8
        self.height = 8
        self.id = "dummy"

    def get_slice(self, index: int):
        if Image is None:
            raise RuntimeError("Pillow required for this test")
        return Image.new("RGB", (self.width, self.height), (index % 255, 0, 0))


@unittest.skipUnless(Image, "Pillow not installed")
class VolumeUtilsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp(prefix="segx-vol-"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_sort_paths_by_number(self) -> None:
        names = ["slice_10.png", "slice_2.png", "a.png", "slice_001.png", "9.png"]
        paths = [self.tmpdir / name for name in names]
        for p in paths:
            _make_image(p)
        sorted_paths = sort_paths_by_number(paths)
        self.assertEqual(
            [p.name for p in sorted_paths], ["slice_001.png", "9.png", "slice_2.png", "slice_10.png", "a.png"]
        )

    def test_slice_cache_window(self) -> None:
        provider = _DummyProvider(depth=20)
        cache = SliceCache(provider, kmax=5)
        for idx in [0, 1, 2, 3]:
            cache.get(idx)
        self.assertLessEqual(len(cache.cache), 5)
        cache.get(10)
        self.assertLessEqual(len(cache.cache), 5)
        # Window should center near 10, so lower slices evicted
        self.assertFalse(any(idx < 5 for idx in cache.cache.keys()))

    def test_png_series_provider_meta(self) -> None:
        files = []
        for i in range(3):
            p = self.tmpdir / f"case_{i:04d}.png"
            _make_image(p)
            files.append(p)
        provider = PngSeriesProvider("case", files)
        self.assertEqual(provider.depth, 3)
        self.assertEqual(provider.width, 8)
        self.assertTrue(provider.meta.get("isPseudo3D"))
        img = provider.get_slice(1)
        self.assertEqual(img.size, (8, 8))


if __name__ == "__main__":
    unittest.main()
