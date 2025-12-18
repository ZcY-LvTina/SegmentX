import shutil
import tempfile
import unittest
from pathlib import Path

try:
    from PIL import Image
except ImportError:  # pragma: no cover - optional dependency guard
    Image = None

from src.segmentx.data.import_classifier import ImportClassifier, ImportPreferences


def _make_image(path: Path, size=(32, 32), color=(255, 0, 0)) -> None:
    if Image is None:
        raise RuntimeError("Pillow is required for this test")
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color)
    img.save(path)


@unittest.skipUnless(Image, "Pillow not installed")
class ImportClassifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp(prefix="import-classifier-"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_png_series_detected_as_volume(self) -> None:
        paths = []
        series_dir = self.tmpdir / "series"
        for i in range(1, 61):
            p = series_dir / f"case1_{i:04d}.png"
            _make_image(p, size=(64, 64))
            paths.append(str(p))

        clf = ImportClassifier(ImportPreferences(None), enable_smooth_check=False)
        result = clf.classify(paths)
        self.assertEqual(len(result.series3d), 1)
        self.assertEqual(len(result.images2d), 0)
        self.assertEqual(result.series3d[0].meta["sliceCount"], 60)

    def test_plain_images_remain_image2d(self) -> None:
        paths = []
        for name, size in [("a.png", (32, 32)), ("b.png", (48, 32)), ("c.png", (32, 48))]:
            p = self.tmpdir / name
            _make_image(p, size=size)
            paths.append(str(p))
        clf = ImportClassifier(ImportPreferences(None), enable_smooth_check=False)
        result = clf.classify(paths)
        self.assertEqual(len(result.images2d), 3)
        self.assertEqual(len(result.series3d), 0)

    def test_multiple_series_are_split(self) -> None:
        paths = []
        for i in range(1, 6):
            p = self.tmpdir / "g1" / f"caseA_{i:03d}.png"
            _make_image(p, size=(64, 64))
            paths.append(str(p))
        for i in range(10, 26):
            p = self.tmpdir / "g2" / f"caseB_{i:04d}.png"
            _make_image(p, size=(64, 64))
            paths.append(str(p))
        clf = ImportClassifier(ImportPreferences(None), enable_smooth_check=False)
        result = clf.classify(paths)
        self.assertEqual(len(result.series3d), 2)

    def test_nifti_goes_to_volume3d(self) -> None:
        nii = self.tmpdir / "brain.nii.gz"
        nii.write_bytes(b"nifti")
        clf = ImportClassifier(ImportPreferences(None), enable_smooth_check=False)
        result = clf.classify([str(nii)])
        self.assertEqual(len(result.volumes3d), 1)
        self.assertEqual(result.volumes3d[0].type, "volume3d")

    def test_dicom_directory_detected(self) -> None:
        dicom_dir = self.tmpdir / "dicom"
        dicom_dir.mkdir(parents=True, exist_ok=True)
        sample = dicom_dir / "IMG0001"
        sample.write_bytes(b"\x00" * 128 + b"DICM" + b"\x00" * 4)
        clf = ImportClassifier(ImportPreferences(None), enable_smooth_check=False)
        result = clf.classify([str(dicom_dir)])
        self.assertEqual(len(result.dicom), 1)
        self.assertEqual(result.dicom[0].type, "dicom")


if __name__ == "__main__":
    unittest.main()
