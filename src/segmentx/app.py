import sys

from PySide6.QtWidgets import QApplication

from .core.sam_engine import SamEngine
from .ui.main_window import MainWindow


def main() -> None:
    """Application entry point: wire engine and main window, then start Qt loop."""
    app = QApplication(sys.argv)
    engine = SamEngine()
    window = MainWindow(engine)
    window.show()
    sys.exit(app.exec())


__all__ = ["main"]
