from typing import Optional

from PIL import Image, ImageDraw
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QPixmap

from ..core.session import ImageState
from ..utils import qt_convert
from .clickable_label import ClickableLabel


class ImageViewer(ClickableLabel):
    """Display image with overlay and convert click positions to image coords."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._state: Optional[ImageState] = None
        self._current_pixmap: Optional[QPixmap] = None

    def set_state(self, state: ImageState) -> None:
        self._state = state
        self._render()

    def _render(self) -> None:
        if not self._state:
            self.clear()
            return

        display_image: Image.Image = self._state.display_image.copy()
        overlay = Image.new("RGBA", display_image.size, (0, 0, 0, 0))

        # Layered overlay: show hint layer when enabled, otherwise fall back to result mask
        mask_to_draw = None
        if self._state.mask_layers.show_hint:
            if self._state.mask_layers.hint is not None:
                mask_to_draw = ("hint", self._state.mask_layers.hint)
            elif self._state.mask_layers.result is not None:
                mask_to_draw = ("result", self._state.mask_layers.result)

        if mask_to_draw is not None:
            layer_type, mask_data = mask_to_draw
            mask_image = Image.fromarray((mask_data * 255).astype("uint8"), mode="L")
            opacity = 100 if layer_type == "hint" else 140
            color_mask = Image.new("RGBA", display_image.size, (0, 255, 0, opacity))
            overlay.paste(color_mask, (0, 0), mask_image)

        # Draw click points
        # Click markers属于提示层，隐藏提示掩膜时一并隐藏
        if self._state.mask_layers.show_hint:
            for point, label in zip(self._state.click_points, self._state.labels):
                x, y = point
                color = (255, 0, 0, 255) if label == 1 else (0, 0, 255, 255)
                size = 10
                marker = Image.new("RGBA", (size * 2, size * 2), (0, 0, 0, 0))
                marker_draw = ImageDraw.Draw(marker)
                marker_draw.ellipse([(0, 0), (size * 2 - 1, size * 2 - 1)], fill=color)
                overlay.paste(marker, (x - size, y - size), marker)

        composed = Image.alpha_composite(display_image.convert("RGBA"), overlay).convert("RGB")
        pixmap = qt_convert.pil_to_qpixmap(composed)
        scaled_pixmap = pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._current_pixmap = scaled_pixmap
        self.setPixmap(scaled_pixmap)

    def map_to_image_coords(self, pos: QPoint) -> Optional[QPoint]:
        if not self._state or not self._current_pixmap:
            return None

        label_size = self.size()
        pixmap_size = self._current_pixmap.size()
        img_width, img_height = self._state.display_image.size

        x_offset = (label_size.width() - pixmap_size.width()) // 2
        y_offset = (label_size.height() - pixmap_size.height()) // 2

        click_x_in_content = pos.x() - x_offset
        click_y_in_content = pos.y() - y_offset

        scale = min(pixmap_size.width() / img_width, pixmap_size.height() / img_height)
        if scale == 0:
            return None

        original_x = int(click_x_in_content / scale)
        original_y = int(click_y_in_content / scale)

        original_x = max(0, min(original_x, img_width - 1))
        original_y = max(0, min(original_y, img_height - 1))

        return QPoint(original_x, original_y)

    def resizeEvent(self, event):  # type: ignore[override]
        super().resizeEvent(event)
        if self._state:
            self._render()


__all__ = ["ImageViewer"]
