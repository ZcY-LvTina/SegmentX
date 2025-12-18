import sys

from PySide6.QtWidgets import QApplication

from .core.sam_engine import SamEngine
from .models.registry import ModelRegistry
from .ui.main_window import MainWindow


def main() -> None:
    """Application entry point: wire engine and main window, then start Qt loop."""
    app = QApplication(sys.argv)
    registry = ModelRegistry()
    try:
        engine = SamEngine()
    except Exception:
        engine = None
    window = MainWindow(registry, engine)
    window.show()
    sys.exit(app.exec())


__all__ = ["main"]
