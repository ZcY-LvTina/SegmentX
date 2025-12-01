from PySide6.QtCore import QPoint, Signal, Qt
from PySide6.QtWidgets import QLabel


class ClickableLabel(QLabel):
    clicked = Signal(QPoint)

    def mousePressEvent(self, event):  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(event.position().toPoint())
        super().mousePressEvent(event)


__all__ = ["ClickableLabel"]
