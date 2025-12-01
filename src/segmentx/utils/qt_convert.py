from typing import Optional

import numpy as np
from PIL import Image, ImageQt
from PySide6.QtGui import QImage, QPixmap


def pil_to_qpixmap(image: Image.Image) -> QPixmap:
    """Convert PIL image to QPixmap."""
    return QPixmap.fromImage(ImageQt.ImageQt(image))


def numpy_to_qimage(array: np.ndarray) -> QImage:
    """Convert numpy image array (H, W[, C]) to QImage."""
    if array.ndim == 2:
        h, w = array.shape
        return QImage(array.data, w, h, QImage.Format.Format_Grayscale8).copy()

    if array.shape[2] == 3:
        h, w, _ = array.shape
        return QImage(array.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()

    if array.shape[2] == 4:
        h, w, _ = array.shape
        return QImage(array.data, w, h, 4 * w, QImage.Format.Format_RGBA8888).copy()

    raise ValueError("Unsupported numpy array shape for QImage conversion")


def numpy_to_qpixmap(array: np.ndarray) -> QPixmap:
    return QPixmap.fromImage(numpy_to_qimage(array))


def qimage_to_pil(image: QImage) -> Image.Image:
    """Convert QImage to PIL Image."""
    return ImageQt.fromqimage(image)


__all__ = ["pil_to_qpixmap", "numpy_to_qimage", "numpy_to_qpixmap", "qimage_to_pil"]
