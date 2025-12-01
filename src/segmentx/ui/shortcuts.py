from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import QMainWindow
from PySide6.QtCore import Qt


def setup_shortcuts(
    window: QMainWindow,
    undo_cb,
    redo_cb,
    prev_cb,
    next_cb,
) -> None:
    undo_action = QAction("撤销", window)
    undo_action.setShortcut(QKeySequence.StandardKey.Undo)
    undo_action.triggered.connect(undo_cb)
    window.addAction(undo_action)

    redo_action = QAction("重做", window)
    redo_action.setShortcut(QKeySequence.StandardKey.Redo)
    redo_action.triggered.connect(redo_cb)
    window.addAction(redo_action)

    prev_action = QAction("上一张", window)
    prev_action.setShortcut(QKeySequence(Qt.Key_Left))
    prev_action.triggered.connect(prev_cb)
    window.addAction(prev_action)

    next_action = QAction("下一张", window)
    next_action.setShortcut(QKeySequence(Qt.Key_Right))
    next_action.triggered.connect(next_cb)
    window.addAction(next_action)


__all__ = ["setup_shortcuts"]
