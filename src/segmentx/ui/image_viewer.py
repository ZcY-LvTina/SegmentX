from typing import Optional, Tuple

from PIL import Image, ImageDraw
from PySide6.QtCore import QPoint, QRect, QSize, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QWidget

from ..core.session import ImageState
from ..utils import qt_convert
from .clickable_label import ClickableLabel


class AspectRatioContainer(QWidget):
    """Keep a child widget within a target aspect ratio."""

    def __init__(self, ratio: float = 4 / 3, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._ratio = ratio
        self._child: Optional[QWidget] = None

    def set_child(self, widget: QWidget) -> None:
        self._child = widget
        widget.setParent(self)
        widget.show()
        self._relayout_child()

    def resizeEvent(self, event):  # type: ignore[override]
        super().resizeEvent(event)
        self._relayout_child()

    def _relayout_child(self) -> None:
        if not self._child:
            return
        available = self.size()
        target_w = available.width()
        target_h = int(target_w / self._ratio)
        if target_h > available.height():
            target_h = available.height()
            target_w = int(target_h * self._ratio)
        x_offset = (available.width() - target_w) // 2
        y_offset = (available.height() - target_h) // 2
        self._child.setGeometry(QRect(x_offset, y_offset, target_w, target_h))


class ImageViewer(ClickableLabel):
    """Display image with overlay and convert view coords <-> image coords."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._state: Optional[ImageState] = None
        self._current_pixmap: Optional[QPixmap] = None
        self._current_scale: float = 1.0  # user-controlled scale multiplier
        self._fit_scale: float = 1.0  # scale to fit viewport (view coords)
        self._last_pixmap_size: Optional[QSize] = None
        self._last_offsets: Tuple[int, int] = (0, 0)

    def set_state(self, state: ImageState, reset_view: bool = False) -> None:
        self._state = state
        if reset_view:
            self.reset_view()
        else:
            self._render()

    def reset_view(self) -> None:
        """Reset zoom to fit view."""
        self._current_scale = 1.0
        self._render()

    def zoom(self, factor: float) -> None:
        # clamp scale to avoid too small/large values
        self._current_scale = max(0.2, min(self._current_scale * factor, 5.0))
        self._render()

    def _render(self) -> None:
        if not self._state:
            self.clear()
            self._current_pixmap = None
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

        viewport_size = self.size()
        if viewport_size.width() == 0 or viewport_size.height() == 0:
            return

        img_width, img_height = display_image.size
        self._fit_scale = min(
            viewport_size.width() / img_width,
            viewport_size.height() / img_height,
        )
        effective_scale = self._fit_scale * self._current_scale
        target_size = QSize(int(img_width * effective_scale), int(img_height * effective_scale))

        scaled_pixmap = pixmap.scaled(
            target_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        self._last_pixmap_size = scaled_pixmap.size()
        self._last_offsets = (
            (viewport_size.width() - scaled_pixmap.width()) // 2,
            (viewport_size.height() - scaled_pixmap.height()) // 2,
        )

        # Centered placement within the viewport
        composed_label = QPixmap(viewport_size)
        composed_label.fill(Qt.GlobalColor.transparent)
        painter = None
        try:
            from PySide6.QtGui import QPainter

            painter = QPainter(composed_label)
            painter.drawPixmap(self._last_offsets[0], self._last_offsets[1], scaled_pixmap)
        finally:
            if painter:
                painter.end()

        self._current_pixmap = scaled_pixmap
        self.setPixmap(composed_label)

    def map_to_image_coords(self, pos: QPoint) -> Optional[QPoint]:
        """Convert from view coords (widget space) to original image pixel coords."""
        if not self._state or not self._current_pixmap or not self._last_pixmap_size:
            return None

        click_x_in_content = pos.x() - self._last_offsets[0]
        click_y_in_content = pos.y() - self._last_offsets[1]
        if click_x_in_content < 0 or click_y_in_content < 0:
            return None

        img_width, img_height = self._state.display_image.size
        effective_scale = self._fit_scale * self._current_scale
        if effective_scale == 0:
            return None

        original_x = int(click_x_in_content / effective_scale)
        original_y = int(click_y_in_content / effective_scale)

        if original_x < 0 or original_y < 0 or original_x >= img_width or original_y >= img_height:
            return None

        return QPoint(original_x, original_y)

    def resizeEvent(self, event):  # type: ignore[override]
        super().resizeEvent(event)
        if self._state:
            self._render()


__all__ = ["ImageViewer", "AspectRatioContainer"]
