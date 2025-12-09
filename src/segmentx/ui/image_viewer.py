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
        self._offset_x: float = 0.0  # 视口内的图像偏移
        self._offset_y: float = 0.0
        self._dragging: bool = False
        self._last_mouse_pos: Optional[QPoint] = None
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
        self._offset_x = 0.0
        self._offset_y = 0.0
        self._render()

    def zoom(self, factor: float) -> None:
        """缩放：基于视口中心做锚点，保持对外接口不变。"""
        center_anchor = self.rect().center()
        self._apply_zoom(factor, center_anchor)

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
        self._fit_scale = min(viewport_size.width() / img_width, viewport_size.height() / img_height)
        effective_scale = self._fit_scale * self._current_scale
        target_size = QSize(int(img_width * effective_scale), int(img_height * effective_scale))

        scaled_pixmap = pixmap.scaled(
            target_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        scaled_size = scaled_pixmap.size()
        self._last_pixmap_size = scaled_size

        # 根据视图大小限制偏移：小于视口则居中，大于视口则可拖动但不能完全移出
        self._apply_offset_constraints(viewport_size, scaled_size)
        self._last_offsets = (int(self._offset_x), int(self._offset_y))

        # Centered placement within the viewport
        composed_label = QPixmap(viewport_size)
        composed_label.fill(Qt.GlobalColor.transparent)
        painter = None
        try:
            from PySide6.QtGui import QPainter

            painter = QPainter(composed_label)
            painter.drawPixmap(int(self._offset_x), int(self._offset_y), scaled_pixmap)
        finally:
            if painter:
                painter.end()

        self._current_pixmap = scaled_pixmap
        self.setPixmap(composed_label)

    def map_to_image_coords(self, pos: QPoint) -> Optional[QPoint]:
        """Convert from view coords (widget space) to original image pixel coords."""
        if not self._state or not self._current_pixmap or not self._last_pixmap_size:
            return None

        click_x_in_content = pos.x() - int(self._offset_x)
        click_y_in_content = pos.y() - int(self._offset_y)
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

    def wheelEvent(self, event):  # type: ignore[override]
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            if not self._state:
                return
            delta = event.angleDelta().y()
            factor = 1.1 if delta > 0 else 0.9
            anchor_pos = event.position().toPoint()
            self._apply_zoom(factor, anchor_pos)
            event.accept()
            return
        super().wheelEvent(event)

    def mousePressEvent(self, event):  # type: ignore[override]
        if event.button() in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton):
            self._dragging = True
            self._last_mouse_pos = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):  # type: ignore[override]
        if self._dragging and event.buttons() & (Qt.MouseButton.RightButton | Qt.MouseButton.MiddleButton):
            current_pos = event.position().toPoint()
            if self._last_mouse_pos:
                delta = current_pos - self._last_mouse_pos
                self._offset_x += delta.x()
                self._offset_y += delta.y()
                # 拖动时即时刷新，保持偏移受约束
                self._render()
            self._last_mouse_pos = current_pos
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):  # type: ignore[override]
        if event.button() in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton):
            self._dragging = False
            self._last_mouse_pos = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def resizeEvent(self, event):  # type: ignore[override]
        super().resizeEvent(event)
        if self._state:
            self._render()

    def _apply_zoom(self, factor: float, anchor_pos: Optional[QPoint]) -> None:
        """以锚点为中心缩放，锚点对应的图像坐标在缩放后尽量保持原位置。"""
        if not self._state:
            return
        old_scale = self._current_scale
        new_scale = max(0.2, min(old_scale * factor, 5.0))
        if abs(new_scale - old_scale) < 1e-6:
            return

        anchor_image_coords = None
        if anchor_pos is not None:
            anchor_image_coords = self._map_to_image_coords_float(anchor_pos)

        self._current_scale = new_scale

        if anchor_pos is not None and anchor_image_coords is not None:
            effective_scale = self._fit_scale * self._current_scale
            ax, ay = anchor_image_coords
            self._offset_x = anchor_pos.x() - ax * effective_scale
            self._offset_y = anchor_pos.y() - ay * effective_scale

        self._render()

    def _apply_offset_constraints(self, viewport_size: QSize, scaled_size: QSize) -> None:
        """偏移约束：小图居中，大图可拖动但始终有内容可见。"""
        if scaled_size.width() <= viewport_size.width():
            self._offset_x = (viewport_size.width() - scaled_size.width()) / 2
        else:
            min_x = viewport_size.width() - scaled_size.width()
            max_x = 0
            self._offset_x = max(min(self._offset_x, max_x), min_x)

        if scaled_size.height() <= viewport_size.height():
            self._offset_y = (viewport_size.height() - scaled_size.height()) / 2
        else:
            min_y = viewport_size.height() - scaled_size.height()
            max_y = 0
            self._offset_y = max(min(self._offset_y, max_y), min_y)

    def _map_to_image_coords_float(self, pos: QPoint) -> Optional[Tuple[float, float]]:
        """浮点版坐标反算，用于缩放锚点保持位置。"""
        if not self._state or not self._current_pixmap or not self._last_pixmap_size:
            return None
        effective_scale = self._fit_scale * self._current_scale
        if effective_scale == 0:
            return None

        img_x = (pos.x() - self._offset_x) / effective_scale
        img_y = (pos.y() - self._offset_y) / effective_scale
        return (img_x, img_y)


__all__ = ["ImageViewer", "AspectRatioContainer"]
