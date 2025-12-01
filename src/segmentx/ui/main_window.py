from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import Qt, QPoint
from PySide6.QtWidgets import (
    QLabel,
    QHBoxLayout,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ..config import MAX_HISTORY
from ..core.export import bulk_export, save_masked_result
from ..core.image_io import image_to_array, load_image
from ..core.sam_engine import SamEngine
from ..core.segmentation_pipeline import run_segmentation
from ..core.session import ImageState, Session
from .dialogs import (
    choose_directory,
    choose_format,
    choose_images,
    choose_save_path,
    show_error,
    show_info,
    show_warning,
)
from .image_viewer import ImageViewer
from .shortcuts import setup_shortcuts


class MainWindow(QMainWindow):
    def __init__(self, engine: SamEngine):
        super().__init__()
        self.engine = engine
        self.sessions: List[Session] = []
        self.current_index: int = -1
        self.current_mode: str = "foreground"

        self._init_ui()
        self.setWindowTitle("Segment")
        self.resize(1200, 800)
        setup_shortcuts(
            self,
            undo_cb=self.undo_action,
            redo_cb=self.redo_action,
            prev_cb=self.prev_image,
            next_cb=self.next_image,
        )

    def _init_ui(self) -> None:
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.load_button = QPushButton("加载图像")
        self.load_multiple_button = QPushButton("加载多个图像")
        self.prev_button = QPushButton("上一张 (←)")
        self.next_button = QPushButton("下一张 (→)")
        self.clear_button = QPushButton("清除标记")
        self.save_button = QPushButton("保存当前结果")
        self.save_all_button = QPushButton("保存所有结果")
        self.undo_button = QPushButton("撤销 (Ctrl+Z)")
        self.redo_button = QPushButton("重做 (Ctrl+Y)")
        self.foreground_button = QPushButton("前景标记")
        self.background_button = QPushButton("背景标记")
        self.status_label = QLabel("状态: 等待加载图像")
        self.image_info_label = QLabel("当前图片: 0/0")

        # Button wiring
        self.load_button.clicked.connect(lambda: self.load_images(single=True))
        self.load_multiple_button.clicked.connect(lambda: self.load_images(single=False))
        self.prev_button.clicked.connect(self.prev_image)
        self.next_button.clicked.connect(self.next_image)
        self.clear_button.clicked.connect(self.clear_markers)
        self.save_button.clicked.connect(self.save_current_result)
        self.save_all_button.clicked.connect(self.save_all_results)
        self.undo_button.clicked.connect(self.undo_action)
        self.redo_button.clicked.connect(self.redo_action)
        self.foreground_button.clicked.connect(self.set_foreground_mode)
        self.background_button.clicked.connect(self.set_background_mode)

        # Initial states
        self.undo_button.setEnabled(False)
        self.redo_button.setEnabled(False)
        self.prev_button.setEnabled(False)
        self.next_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.save_all_button.setEnabled(False)
        self.foreground_button.setCheckable(True)
        self.background_button.setCheckable(True)
        self.foreground_button.setChecked(True)

        # Add to layout
        for widget in [
            self.load_button,
            self.load_multiple_button,
            self.prev_button,
            self.next_button,
            self.clear_button,
            self.save_button,
            self.save_all_button,
            self.undo_button,
            self.redo_button,
            self.foreground_button,
            self.background_button,
            self.image_info_label,
            self.status_label,
        ]:
            control_layout.addWidget(widget)
        main_layout.addWidget(control_panel, stretch=1)

        # Image area
        image_panel = QWidget()
        image_layout = QVBoxLayout(image_panel)
        self.image_viewer = ImageViewer()
        self.image_viewer.setMinimumSize(800, 600)
        self.image_viewer.clicked.connect(self.on_image_clicked)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.image_viewer)
        image_layout.addWidget(scroll_area)
        main_layout.addWidget(image_panel, stretch=4)

    @property
    def current_session(self) -> Optional[Session]:
        if 0 <= self.current_index < len(self.sessions):
            return self.sessions[self.current_index]
        return None

    def load_images(self, single: bool = True) -> None:
        file_paths = choose_images(self, single)
        if not file_paths:
            return

        try:
            self.sessions = []
            self.current_index = -1
            for path in file_paths:
                img = load_image(path)
                state = ImageState(
                    path=path,
                    original_image=img,
                    display_image=img.copy(),
                    click_points=[],
                    labels=[],
                    mask=None,
                )
                self.sessions.append(Session(state, max_history=MAX_HISTORY))

            if self.sessions:
                self.current_index = 0
                self._show_current_image()
                self._set_engine_image(self.current_session.state.original_image)  # type: ignore[union-attr]
                self._update_navigation_buttons()
                self.save_button.setEnabled(True)
                self.save_all_button.setEnabled(True)
                self.status_label.setText(f"状态: 已加载 {len(self.sessions)} 张图片")
                self._update_image_info()
        except Exception as exc:
            show_error(self, "错误", f"加载图像失败: {exc}")
            self.status_label.setText("状态: 图像加载失败")

    def _set_engine_image(self, image) -> None:
        try:
            self.engine.set_image(image_to_array(image))
        except Exception as exc:
            show_error(self, "错误", f"设置图像到SAM模型失败: {exc}")

    def _show_current_image(self) -> None:
        session = self.current_session
        if not session:
            return
        session.undo_stack.clear()
        session.redo_stack.clear()
        self.image_viewer.set_state(session.state)
        self._update_history_buttons()

    def prev_image(self) -> None:
        if self.current_index > 0:
            self.current_index -= 1
            self._show_current_image()
            self._set_engine_image(self.current_session.state.original_image)  # type: ignore[union-attr]
            self._update_navigation_buttons()
            self._update_image_info()
            self.status_label.setText("状态: 切换到上一张图片")

    def next_image(self) -> None:
        if self.current_index < len(self.sessions) - 1:
            self.current_index += 1
            self._show_current_image()
            self._set_engine_image(self.current_session.state.original_image)  # type: ignore[union-attr]
            self._update_navigation_buttons()
            self._update_image_info()
            self.status_label.setText("状态: 切换到下一张图片")

    def _update_navigation_buttons(self) -> None:
        self.prev_button.setEnabled(self.current_index > 0)
        self.next_button.setEnabled(self.current_index < len(self.sessions) - 1)

    def _update_image_info(self) -> None:
        if self.sessions:
            self.image_info_label.setText(f"当前图片: {self.current_index + 1}/{len(self.sessions)}")
        else:
            self.image_info_label.setText("当前图片: 0/0")

    def save_current_result(self) -> None:
        session = self.current_session
        if not session or session.state.mask is None:
            show_warning(self, "警告", "没有可保存的分割结果")
            return

        original_name = Path(session.state.path).stem
        default_filename = f"{original_name}_segmented.png"
        file_path, selected_filter = choose_save_path(self, default_filename)
        if not file_path:
            return

        format_map = {
            "PNG图像 (*.png)": ("PNG", ".png"),
            "JPEG图像 (*.jpg *.jpeg)": ("JPEG", ".jpg"),
            "BMP图像 (*.bmp)": ("BMP", ".bmp"),
            "TIFF图像 (*.tif *.tiff)": ("TIFF", ".tif"),
        }

        if selected_filter in format_map:
            file_format, default_ext = format_map[selected_filter]
            if not any(file_path.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]):
                file_path += default_ext
        else:
            ext = Path(file_path).suffix.lower()
            if ext == ".png":
                file_format = "PNG"
            elif ext in [".jpg", ".jpeg"]:
                file_format = "JPEG"
            elif ext == ".bmp":
                file_format = "BMP"
            elif ext in [".tif", ".tiff"]:
                file_format = "TIFF"
            else:
                file_format = "PNG"
                file_path += ".png"

        try:
            saved_path = save_masked_result(session.state, Path(file_path), file_format)
            self.status_label.setText(f"状态: 结果已保存到 {saved_path.name}")
            show_info(self, "成功", f"图片已成功保存到:\n{saved_path}")
        except Exception as exc:
            show_error(self, "错误", f"保存结果失败: {exc}")
            self.status_label.setText("状态: 保存失败")

    def save_all_results(self) -> None:
        if not self.sessions:
            show_warning(self, "警告", "没有可保存的分割结果")
            return

        selected_format = choose_format(self)
        if not selected_format:
            return

        save_dir = choose_directory(self)
        if not save_dir:
            return

        ext_map = {"PNG": ".png", "JPEG": ".jpg", "BMP": ".bmp", "TIFF": ".tif"}
        try:
            exported = bulk_export(
                [session.state for session in self.sessions if session.state.mask is not None],
                Path(save_dir),
                file_format=selected_format,
                extension=ext_map.get(selected_format),
            )
            self.status_label.setText(
                f"状态: 已保存 {len(exported)}/{len(self.sessions)} 张图片的结果到 {save_dir}"
            )
            show_info(
                self,
                "完成",
                f"成功保存 {len(exported)} 张图片的分割结果\n格式: {selected_format}\n保存目录: {save_dir}",
            )
        except Exception as exc:
            show_error(self, "错误", f"批量保存结果失败: {exc}")
            self.status_label.setText("状态: 批量保存失败")

    def set_foreground_mode(self) -> None:
        self.current_mode = "foreground"
        self.foreground_button.setChecked(True)
        self.background_button.setChecked(False)
        self.status_label.setText("状态: 当前模式 - 前景标记")

    def set_background_mode(self) -> None:
        self.current_mode = "background"
        self.foreground_button.setChecked(False)
        self.background_button.setChecked(True)
        self.status_label.setText("状态: 当前模式 - 背景标记")

    def on_image_clicked(self, pos: QPoint) -> None:
        session = self.current_session
        if not session:
            return
        image_pos = self.image_viewer.map_to_image_coords(pos)
        if not image_pos:
            return

        session.save()
        session.state.click_points.append([image_pos.x(), image_pos.y()])
        session.state.labels.append(1 if self.current_mode == "foreground" else 0)

        self._run_segmentation(session)
        self.image_viewer.set_state(session.state)
        self._update_history_buttons()

    def _run_segmentation(self, session: Session) -> None:
        try:
            self.status_label.setText("状态: 正在执行分割...")
            mask = run_segmentation(session, self.engine)
            if mask is None:
                self.status_label.setText("状态: 没有足够的点击点执行分割")
                return
            self.status_label.setText("状态: 分割完成")
        except Exception as exc:
            show_error(self, "错误", f"分割失败: {exc}")
            self.status_label.setText("状态: 分割失败")

    def update_image_display(self) -> None:
        session = self.current_session
        if not session:
            return
        self.image_viewer.set_state(session.state)

    def clear_markers(self) -> None:
        session = self.current_session
        if not session:
            return

        session.save()
        session.state.click_points = []
        session.state.labels = []
        session.state.mask = None
        session.state.display_image = session.state.original_image.copy()

        self.image_viewer.set_state(session.state)
        self.status_label.setText("状态: 已清除所有标记")
        self._update_history_buttons()

    def undo_action(self) -> None:
        session = self.current_session
        if not session:
            return
        if session.undo() is None:
            return
        self.image_viewer.set_state(session.state)
        self.status_label.setText("状态: 已撤销上一步操作")
        self._update_history_buttons()

    def redo_action(self) -> None:
        session = self.current_session
        if not session:
            return
        if session.redo() is None:
            return
        self.image_viewer.set_state(session.state)
        self.status_label.setText("状态: 已重做操作")
        self._update_history_buttons()

    def _update_history_buttons(self) -> None:
        session = self.current_session
        if not session:
            self.undo_button.setEnabled(False)
            self.redo_button.setEnabled(False)
            return
        self.undo_button.setEnabled(session.can_undo())
        self.redo_button.setEnabled(session.can_redo())

    def resizeEvent(self, event):  # type: ignore[override]
        super().resizeEvent(event)
        self.update_image_display()


__all__ = ["MainWindow"]
