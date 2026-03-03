from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from PIL import Image
from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSlider, QSpinBox, QVBoxLayout, QWidget

from ..core.session import ImageState, Session
from ..volume.cache import SliceCache
from ..volume.provider import SliceProvider
from .image_viewer import ImageViewer


class VolumeViewer3D(QWidget):
    """Slice-by-slice viewer built on top of existing ImageViewer."""

    slice_changed = Signal(int)
    clicked = Signal(QPoint)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.provider: Optional[SliceProvider] = None
        self.cache: Optional[SliceCache] = None
        self._sessions: Dict[int, Session] = {}
        self._current_index: int = 0
        self._kmax: int = 64

        self.viewer = ImageViewer()
        self.viewer.clicked.connect(self.clicked.emit)

        self.info_label = QLabel("切片 0/0 | 0x0x0 | cache 0/0")
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.valueChanged.connect(self._on_slider_changed)
        self.spin = QSpinBox()
        self.spin.setMinimum(0)
        self.spin.valueChanged.connect(self._on_spin_changed)

        ctrl = QHBoxLayout()
        ctrl.setContentsMargins(4, 0, 4, 0)
        ctrl.setSpacing(8)
        ctrl.addWidget(self.info_label)
        ctrl.addStretch()
        ctrl.addWidget(self.spin)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.viewer, stretch=1)
        layout.addWidget(self.slider)
        layout.addLayout(ctrl)

    # Public API ----------------------------------------------------------
    def set_provider(self, provider: SliceProvider, cache: SliceCache, kmax: int = 64) -> None:
        self.provider = provider
        self.cache = cache
        self._kmax = kmax
        self._sessions.clear()
        self._current_index = 0
        self.slider.blockSignals(True)
        self.spin.blockSignals(True)
        self.slider.setMaximum(max(0, provider.depth - 1))
        self.spin.setMaximum(max(0, provider.depth - 1))
        self.slider.setValue(0)
        self.spin.setValue(0)
        self.slider.blockSignals(False)
        self.spin.blockSignals(False)
        self._show_slice(0, reset_view=True)

    def current_session(self) -> Optional[Session]:
        return self._sessions.get(self._current_index)

    def current_index(self) -> int:
        return self._current_index

    def total_slices(self) -> int:
        return self.provider.depth if self.provider else 0

    def set_mask_visibility(self, show: bool) -> None:
        session = self.current_session()
        if not session:
            return
        session.state.mask_layers.show_hint = show
        self.viewer.set_state(session.state, reset_view=False)

    # Event handlers ------------------------------------------------------
    def wheelEvent(self, event):  # type: ignore[override]
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            if not self.viewer:
                return
            delta = event.angleDelta().y()
            factor = 1.1 if delta > 0 else 0.9
            self.viewer._apply_zoom(factor, event.position().toPoint())
            event.accept()
            return
        if not self.provider:
            return
        delta = event.angleDelta().y()
        step = 1 if abs(delta) < 240 else 3
        if delta < 0:
            step = -step
        new_idx = max(0, min(self._current_index + step, self.provider.depth - 1))
        self.slider.setValue(new_idx)
        event.accept()

    # Internal helpers ----------------------------------------------------
    def _on_slider_changed(self, value: int) -> None:
        self._show_slice(value)

    def _on_spin_changed(self, value: int) -> None:
        self.slider.blockSignals(True)
        self.slider.setValue(value)
        self.slider.blockSignals(False)
        self._show_slice(value)

    def _make_session(self, image: Image.Image, path_hint: str) -> Session:
        state = ImageState(
            path=path_hint,
            original_image=image,
            display_image=image.copy(),
            click_points=[],
            labels=[],
        )
        return Session(state, max_history=20)

    def _show_slice(self, index: int, reset_view: bool = False) -> None:
        if not self.provider or not self.cache:
            return
        if index < 0 or index >= self.provider.depth:
            return
        try:
            img = self.cache.get(index)
        except Exception as exc:  # pragma: no cover - UI path
            self.info_label.setText(f"切片 {index} 加载失败: {exc}")
            return
        session = self._sessions.get(index)
        if not session:
            session = self._make_session(img, f"{self.provider.id}::{index}")
            self._sessions[index] = session
        else:
            session.state.original_image = img
            session.state.display_image = img.copy()
        self._current_index = index
        self.viewer.set_state(session.state, reset_view=reset_view)

        self.spin.blockSignals(True)
        self.spin.setValue(index)
        self.spin.blockSignals(False)

        stats = self.cache.stats()
        self.info_label.setText(
            f"切片 {index+1}/{self.provider.depth} | "
            f"{self.provider.width}x{self.provider.height}x{self.provider.depth} | cache {stats['cached']}/{stats['kmax']}"
        )
        self.slice_changed.emit(index)

