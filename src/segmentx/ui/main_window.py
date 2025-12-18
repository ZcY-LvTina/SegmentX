import json
import threading
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image, ImageDraw
from PySide6.QtCore import QPoint, Qt, QSize, QThread, Signal, QObject, QUrl, QMetaObject, Q_ARG, Slot
from PySide6.QtGui import QDesktopServices, QIcon
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QCheckBox,
    QComboBox,
    QInputDialog,
    QSizePolicy,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..config import (
    MAX_HISTORY,
    MODEL_TYPE,
    RESOURCES_DIR,
    MODELS_DIR,
    IMPORT_PREFERENCES_FILE,
    NNUNET_PREDICT_CMD,
    NNUNET_TRAIN_CMD,
    NNUNET_SETTINGS_FILE,
)
from ..data.import_classifier import (
    ClassificationResult,
    ImportClassifier,
    ImportPreferences,
    PngSeriesGroup,
    ResourceItem,
)
from ..data.volume_loader import VolumeLoader, Volume3D
from ..core.export import bulk_export, save_masked_result
from ..core.image_io import image_to_array, load_image
from ..core.sam_engine import SamEngine
from ..core.segmentation_pipeline import run_segmentation
from ..core.session import ImageState, Session
from ..models import ModelRegistry, ModelRecord
from ..models.downloader import DownloadCancelled, download_file
from ..models.nnunet import NNUNetAdapter, TrainConfig, TrainRunner
from .dialogs import (
    choose_directory,
    choose_format,
    choose_import_paths,
    choose_folder,
    choose_save_path,
    choose_zip_file,
    show_error,
    show_info,
    show_warning,
)
from .image_viewer import AspectRatioContainer, ImageViewer
from .shortcuts import setup_shortcuts


class TaskWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)
    status = Signal(str)
    progress = Signal(int)
    log = Signal(str)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def run(self) -> None:
        try:
            result = self.fn(self)
            self.finished.emit(result)
        except Exception as exc:  # pragma: no cover - UI path
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self, registry: ModelRegistry, engine: Optional[SamEngine] = None):
        super().__init__()
        self.registry = registry
        self.engine = engine
        self.nnunet_predict_cmd = NNUNET_PREDICT_CMD
        self.nnunet_train_cmd = NNUNET_TRAIN_CMD
        self._load_nnunet_settings()
        self.nnunet_adapter = NNUNetAdapter(self.registry, predict_cmd=self.nnunet_predict_cmd)
        self.train_runner = TrainRunner(self.registry, train_cmd=self.nnunet_train_cmd)
        self.current_model_id: Optional[str] = None
        self.current_volume: Optional[Volume3D] = None
        self.model_records: Dict[str, ModelRecord] = {}
        self._background_threads: List[QThread] = []
        self.volume_loader = VolumeLoader()
        self.import_preferences = ImportPreferences(IMPORT_PREFERENCES_FILE)
        self.resources: List[ResourceItem] = []
        self.volume_resources: List[ResourceItem] = []
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
        self._refresh_model_list()

    def _init_ui(self) -> None:
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主布局：顶部工具栏 + 下方内容区域
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Image viewer created early so control buttons can bind zoom/reset handlers
        self.image_viewer = ImageViewer()
        self.image_viewer.setMinimumSize(800, 600)
        self.image_viewer.clicked.connect(self.on_image_clicked)

        # 左侧侧边栏按钮改为图标+文字 QToolButton
        self.load_button = QToolButton()
        self.prev_button = QToolButton()
        self.next_button = QToolButton()
        self.clear_button = QToolButton()
        self.save_button = QToolButton()
        self.save_all_button = QToolButton()
        self.undo_button = QToolButton()
        self.redo_button = QToolButton()
        self.toggle_hint_button = QToolButton()
        self.toggle_hint_button.setCheckable(True)
        self.toggle_hint_button.setChecked(True)
        self.toggle_hint_button.setEnabled(False)
        self.zoom_in_button = QToolButton()
        self.zoom_out_button = QToolButton()
        self.reset_zoom_button = QToolButton()
        self.foreground_button = QToolButton()
        self.background_button = QToolButton()
        self.manual_mode_button = QToolButton()
        self.manual_add_button = QToolButton()
        self.manual_erase_button = QToolButton()
        self.cancel_polygon_button = QToolButton()
        self.status_label = QLabel("状态: 等待加载图像")
        self.model_install_button = QToolButton()
        self.model_import_zip_button = QToolButton()
        self.model_import_nnunet_button = QToolButton()
        self.model_remove_button = QToolButton()
        self.model_open_dir_button = QToolButton()
        self.run_nnunet_button = QToolButton()
        self.nnunet_settings_button = QToolButton()
        self.run_nnunet_train_button = QToolButton()
        self.model_menu_button = QToolButton()

        # Button wiring
        self.load_button.clicked.connect(self.load_images)
        self.prev_button.clicked.connect(self.prev_image)
        self.next_button.clicked.connect(self.next_image)
        self.clear_button.clicked.connect(self.clear_markers)
        self.save_button.clicked.connect(self.save_current_result)
        self.save_all_button.clicked.connect(self.save_all_results)
        self.undo_button.clicked.connect(self.undo_action)
        self.redo_button.clicked.connect(self.redo_action)
        self.toggle_hint_button.clicked.connect(self.toggle_hint_mask)
        self.zoom_in_button.clicked.connect(lambda: self.image_viewer.zoom(1.2))
        self.zoom_out_button.clicked.connect(lambda: self.image_viewer.zoom(1 / 1.2))
        self.reset_zoom_button.clicked.connect(self.image_viewer.reset_view)
        self.foreground_button.clicked.connect(self.set_foreground_mode)
        self.background_button.clicked.connect(self.set_background_mode)
        self.manual_mode_button.clicked.connect(self.toggle_manual_mode)
        self.manual_add_button.clicked.connect(lambda: self.set_manual_brush_mode("add"))
        self.manual_erase_button.clicked.connect(lambda: self.set_manual_brush_mode("erase"))
        self.cancel_polygon_button.clicked.connect(self.cancel_polygon)
        self._build_model_menu()

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
        self.manual_mode_button.setEnabled(False)
        self.manual_mode_button.setCheckable(True)
        self.manual_add_button.setCheckable(True)
        self.manual_erase_button.setCheckable(True)
        self._enable_manual_controls(False)

        # 顶部工具栏：左侧信息区 + 右侧分割相关按钮
        header_bar = QFrame()
        header_bar.setObjectName("HeaderBar")
        header_bar.setFixedHeight(82)
        header_layout = QHBoxLayout(header_bar)
        header_layout.setContentsMargins(16, 10, 16, 10)
        header_layout.setSpacing(12)

        info_layout = QVBoxLayout()
        info_layout.setSpacing(2)
        self.model_combo = QComboBox()
        self.model_combo.setPlaceholderText("选择模型")
        self.model_combo.setMinimumContentsLength(14)
        self.model_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.model_combo.setMaximumWidth(260)
        view = getattr(self.model_combo, "view", None)
        if view:
            combo_view = view()
            if hasattr(combo_view, "setTextElideMode"):
                combo_view.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.model_combo.currentIndexChanged.connect(self._on_model_selected)
        model_row = QHBoxLayout()
        model_row.setSpacing(6)
        self.refresh_models_button = QPushButton("刷新模型列表")
        self.refresh_models_button.clicked.connect(self._refresh_model_list)
        model_row.addWidget(self.model_combo, stretch=1)
        model_row.addWidget(self.refresh_models_button, stretch=0)
        info_layout.addLayout(model_row)
        header_layout.addLayout(info_layout)
        header_layout.addStretch()

        toolbar_layout = QHBoxLayout()
        toolbar_layout.setSpacing(8)
        for btn, text in [
            (self.clear_button, "清除标记"),
            (self.toggle_hint_button, "隐藏/显示掩膜"),
            (self.foreground_button, "前景标记"),
            (self.background_button, "背景标记"),
            (self.manual_mode_button, "手动精修"),
            (self.manual_add_button, "多边形添加"),
            (self.manual_erase_button, "多边形擦除"),
            (self.cancel_polygon_button, "取消多边形"),
        ]:

            icon_path = None
            if btn is self.clear_button:
                icon_path = RESOURCES_DIR / "icons" / "clear.png"
            elif btn is self.toggle_hint_button:
                icon_path = RESOURCES_DIR / "icons" / "xianshi-yincang.png"
            elif btn is self.foreground_button:
                icon_path = RESOURCES_DIR / "icons" / "qianjing.png"
            elif btn is self.background_button:
                icon_path = RESOURCES_DIR / "icons" / "houjing.png"
            elif btn is self.manual_mode_button:
                icon_path = RESOURCES_DIR / "icons" / "shoudong.png"
            elif btn is self.manual_add_button:
                icon_path = RESOURCES_DIR / "icons" / "poly_add.png"
            elif btn is self.manual_erase_button:
                icon_path = RESOURCES_DIR / "icons" / "poly_erase.png"
            elif btn is self.cancel_polygon_button:
                icon_path = RESOURCES_DIR / "icons" / "cancel.png"
            self._setup_top_button(btn, text, icon_path)
            toolbar_layout.addWidget(btn)
        header_layout.addLayout(toolbar_layout)
        main_layout.addWidget(header_bar)

        # 中间内容区域：左侧侧边栏 + 图片区域
        content_widget = QWidget()
        content_layout = QHBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        sidebar = QFrame()
        sidebar.setObjectName("SideBar")
        sidebar.setFixedWidth(140)
        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(8, 12, 8, 12)
        side_layout.setSpacing(6)

        icon_size = QSize(28, 28)

        self._setup_side_button(self.load_button, "加载图片", RESOURCES_DIR / "icons" / "open.png", icon_size)
        self._setup_side_button(self.prev_button, "上一张", RESOURCES_DIR / "icons" / "prev.png", icon_size)
        self._setup_side_button(self.next_button, "下一张", RESOURCES_DIR / "icons" / "next.png", icon_size)
        self._setup_side_button(self.save_button, "保存结果", RESOURCES_DIR / "icons" / "save.png", icon_size)
        self._setup_side_button(self.save_all_button, "批量保存", RESOURCES_DIR / "icons" / "save_all.png", icon_size)
        self._setup_side_button(self.undo_button, "撤销", RESOURCES_DIR / "icons" / "undo.png", icon_size)
        self._setup_side_button(self.redo_button, "重做", RESOURCES_DIR / "icons" / "redo.png", icon_size)

        for btn in [
            self.load_button,
            self.prev_button,
            self.next_button,
            self.save_button,
            self.save_all_button,
            self.undo_button,
            self.redo_button,
        ]:
            side_layout.addWidget(btn)

        side_layout.addSpacing(6)
        for btn, label, icon_file in [
            (self.zoom_in_button, "放大", "fangda.png"),
            (self.zoom_out_button, "缩小", "suoxiao.png"),
            (self.reset_zoom_button, "适配", "fit.png"),
        ]:

            self._setup_side_button(btn, label, RESOURCES_DIR / "icons" / icon_file, icon_size)
            side_layout.addWidget(btn)

        # 模型管理区域（下拉菜单）
        side_layout.addSpacing(8)
        self._setup_side_button(self.model_menu_button, "设置", RESOURCES_DIR / "icons" / "setting.png", icon_size)
        self.model_menu_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        side_layout.addWidget(self.model_menu_button)

        side_layout.addStretch()
        content_layout.addWidget(sidebar, stretch=0)

        # Image area
        image_panel = QWidget()
        image_layout = QVBoxLayout(image_panel)
        ratio_container = AspectRatioContainer(ratio=4 / 3)
        ratio_container.set_child(self.image_viewer)
        ratio_container.setMinimumSize(800, 600)
        ratio_container.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        image_layout.addWidget(ratio_container)
        content_layout.addWidget(image_panel, stretch=1)
        main_layout.addWidget(content_widget, stretch=1)

        # Bottom status bar
        bottom_bar = QFrame()
        bottom_bar.setObjectName("BottomBar")
        bottom_layout = QHBoxLayout(bottom_bar)
        bottom_layout.setContentsMargins(12, 6, 12, 6)
        bottom_layout.setSpacing(16)
        self.model_status_label = QLabel("当前模型：--")
        self.model_status_label.setStyleSheet("font-size: 12px; color: #374151;")
        self.model_status_label.setMaximumWidth(320)
        self.image_count_label = QLabel("图片：0/0")
        self.image_count_label.setStyleSheet("font-size: 12px; color: #374151;")
        self.volume_count_label = QLabel("体数据：0")
        self.volume_count_label.setStyleSheet("font-size: 12px; color: #374151;")
        self.status_label.setStyleSheet("font-size: 12px; color: #111827;")

        bottom_layout.addWidget(self.model_status_label)
        bottom_layout.addWidget(self.image_count_label)
        bottom_layout.addWidget(self.volume_count_label)
        bottom_layout.addStretch()
        bottom_layout.addWidget(self.status_label)
        main_layout.addWidget(bottom_bar, stretch=0)

        self._apply_styles(sidebar, header_bar)
        self.update_info_bar()

    @property
    def current_session(self) -> Optional[Session]:
        if 0 <= self.current_index < len(self.sessions):
            return self.sessions[self.current_index]
        return None

    def load_images(self) -> None:
        import_paths = choose_import_paths(self)
        if not import_paths:
            return

        try:
            if self.engine:
                self.engine.clear_cache()

            classifier = ImportClassifier(self.import_preferences)
            result = classifier.classify(import_paths)
            if result.ambiguous:
                resolved_series: List[ResourceItem] = []
                resolved_images: List[ResourceItem] = []
                for group in result.ambiguous:
                    decision, remember = self._ask_series_resolution(group)
                    if decision is None:
                        continue
                    if remember:
                        self.import_preferences.remember(
                            group.decision_key(), "volume" if decision == "series" else "images"
                        )
                    if decision == "series":
                        resolved_series.append(group.as_series_resource())
                    else:
                        resolved_images.extend(group.as_image_resources())
                result.series3d.extend(resolved_series)
                result.images2d.extend(resolved_images)
                result.ambiguous = []

            self._apply_import_result(result)
        except Exception as exc:
            show_error(self, "错误", f"加载图像失败: {exc}")
            self.status_label.setText("状态: 图像加载失败")

    def _apply_import_result(self, result: ClassificationResult) -> None:
        """Apply classified resources to UI state."""
        self.resources = result.all_resources()
        self.volume_resources = result.series3d + result.volumes3d + result.dicom
        self.sessions = []
        self.current_index = -1
        self.manual_mode_button.setEnabled(False)
        self.manual_mode_button.setChecked(False)
        self._enable_manual_controls(False)
        self.toggle_hint_button.setEnabled(False)

        for item in result.images2d:
            if not item.paths:
                continue
            path = item.paths[0]
            img = load_image(path)
            state = ImageState(
                path=path,
                original_image=img,
                display_image=img.copy(),
                click_points=[],
                labels=[],
            )
            self.sessions.append(Session(state, max_history=MAX_HISTORY))

        if self.sessions:
            self.current_index = 0
            self._show_current_image(reset_view=True)
            if self.engine:
                self._set_engine_image(self.current_session.state.original_image, self.current_session.state.path)  # type: ignore[union-attr]
            else:
                self.status_label.setText("状态: 已加载图像，但未选择SAM模型")
        else:
            self.image_viewer.reset_view()

        self._update_navigation_buttons()
        self._update_history_buttons()
        self.save_button.setEnabled(bool(self.sessions))
        self.save_all_button.setEnabled(bool(self.sessions))
        self.manual_mode_button.setEnabled(bool(self.sessions))
        self.toggle_hint_button.setEnabled(bool(self.sessions))
        self._update_image_info()
        hint_state = self.current_session.state.mask_layers.show_hint if self.current_session else False  # type: ignore[union-attr]
        self._sync_hint_button(hint_state)

        image_count = len(self.sessions)
        volume_count = len(self.volume_resources)
        if image_count or volume_count:
            if image_count:
                self.status_label.setText(f"状态: 已加载 {image_count} 张图片，体数据 {volume_count} 个")
            else:
                self.status_label.setText(f"状态: 已加载体数据 {volume_count} 个，当前不支持预览")
        else:
            self.status_label.setText("状态: 未找到可导入的文件")

    def _ask_series_resolution(self, group: PngSeriesGroup) -> tuple[Optional[str], bool]:
        text = (
            f"检测到可能的 3D 序列：{group.parent.name}/{group.prefix} "
            f"共 {len(group.paths)} 张{group.ext}，连续率 {group.scores.get('contiguous_ratio', 0):.2f}，"
            f"尺寸 {group.size[0]}x{group.size[1]}"
        )
        dialog = QMessageBox(self)
        dialog.setWindowTitle("处理 PNG/JPEG 序列")
        dialog.setText(text)
        dialog.setInformativeText("请选择将其作为 3D 序列还是 2D 图片集导入。")
        dialog.setIcon(QMessageBox.Icon.Question)
        as_volume = dialog.addButton("作为 3D 序列", QMessageBox.ButtonRole.AcceptRole)
        as_images = dialog.addButton("作为 2D 图片集", QMessageBox.ButtonRole.DestructiveRole)
        dialog.addButton("跳过", QMessageBox.ButtonRole.RejectRole)
        remember_box = QCheckBox("记住该目录/前缀的选择")
        dialog.setCheckBox(remember_box)
        dialog.exec()
        clicked = dialog.clickedButton()
        if clicked is as_volume:
            return "series", remember_box.isChecked()
        if clicked is as_images:
            return "images", remember_box.isChecked()
        return None, False

    def _set_engine_image(self, image, image_id: str) -> bool:
        # 为多图切换复用缓存的SAM embedding，减少重复编码开销
        if not self.engine:
            show_warning(self, "提示", "当前未加载 SAM 模型，无法执行交互分割。请在顶部选择 SAM 模型。")
            return False
        try:
            self.status_label.setText("状态: 正在准备图像特征")
            from_cache = self.engine.set_image(image_to_array(image), image_id=image_id)
            status = "状态: 使用缓存的图像特征" if from_cache else "状态: 已准备图像特征"
            self.status_label.setText(status)
            return from_cache
        except Exception as exc:
            show_error(self, "错误", f"设置图像到SAM模型失败: {exc}")
            return False

    def _show_current_image(self, reset_view: bool = False) -> None:
        session = self.current_session
        if not session:
            return
        session.undo_stack.clear()
        session.redo_stack.clear()
        self.image_viewer.set_state(session.state, reset_view=reset_view)
        self._update_history_buttons()
        self._sync_hint_button(session.state.mask_layers.show_hint)
        self._sync_manual_controls(session.state)

    def prev_image(self) -> None:
        if self.current_index > 0:
            self.current_index -= 1
            self._show_current_image(reset_view=False)
            from_cache = False
            if self.engine:
                from_cache = self._set_engine_image(
                    self.current_session.state.original_image, self.current_session.state.path
                )  # type: ignore[union-attr]
            self._update_navigation_buttons()
            self._update_image_info()
            status = "状态: 切换到上一张图片 (缓存特征)" if from_cache else "状态: 切换到上一张图片"
            self.status_label.setText(status)

    def next_image(self) -> None:
        if self.current_index < len(self.sessions) - 1:
            self.current_index += 1
            self._show_current_image(reset_view=False)
            from_cache = False
            if self.engine:
                from_cache = self._set_engine_image(
                    self.current_session.state.original_image, self.current_session.state.path
                )  # type: ignore[union-attr]
            self._update_navigation_buttons()
            self._update_image_info()
            status = "状态: 切换到下一张图片 (缓存特征)" if from_cache else "状态: 切换到下一张图片"
            self.status_label.setText(status)

    def _update_navigation_buttons(self) -> None:
        self.prev_button.setEnabled(self.current_index > 0)
        self.next_button.setEnabled(self.current_index < len(self.sessions) - 1)

    def _update_image_info(self) -> None:
        self.update_info_bar()

    def _current_model_display(self) -> str:
        if self.current_model_id == "__legacy_sam__":
            return "SAM (config)"
        record = self.model_records.get(self.current_model_id or "")
        if record and record.manifest:
            return f"{record.manifest.name} ({record.manifest.type})"
        if self.engine:
            name = getattr(self.engine, "model_name", None) or getattr(self.engine, "model_type", None)
            if name:
                return str(name)
        return ""

    def update_info_bar(self) -> None:
        """更新顶部信息栏：模型名称 + 图片进度。"""
        model_name = self._current_model_display() or MODEL_TYPE
        self._set_elided_label(self.model_status_label, f"当前模型：{model_name}")
        if self.sessions and 0 <= self.current_index < len(self.sessions):
            self.image_count_label.setText(f"图片：{self.current_index + 1}/{len(self.sessions)}")
        else:
            self.image_count_label.setText("图片：0/0")
        self.volume_count_label.setText(f"体数据：{len(self.volume_resources)}")

    def _set_elided_label(self, label: QLabel, text: str, fallback_width: int = 280) -> None:
        width = label.width() or label.maximumWidth() or fallback_width
        elided = label.fontMetrics().elidedText(text, Qt.TextElideMode.ElideRight, width)
        label.setText(elided)
        label.setToolTip(text)

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

    def toggle_manual_mode(self) -> None:
        session = self.current_session
        if not session:
            self.manual_mode_button.setChecked(False)
            return

        enable = self.manual_mode_button.isChecked()
        session.save()
        if enable:
            # 进入手动模式：优先复制模型掩膜，否则从空白开始
            if session.state.manual_mask is None:
                base_mask = session.state.auto_mask
                if base_mask is not None:
                    session.state.manual_mask = base_mask.astype(bool).copy()
                else:
                    width, height = session.state.original_image.size
                    session.state.manual_mask = np.zeros((height, width), dtype=bool)
            session.state.manual_edit_enabled = True
            session.state.current_polygon_points = []
            if session.state.manual_brush_mode not in ["add", "erase"]:
                session.state.manual_brush_mode = "add"
            self._enable_manual_controls(True)
            self.set_manual_brush_mode(session.state.manual_brush_mode, update_status=False)
            self.status_label.setText("状态: 手动精修模式开启")
        else:
            session.state.manual_edit_enabled = False
            session.state.current_polygon_points = []
            self._enable_manual_controls(False)
            self.manual_mode_button.setChecked(False)
            self.status_label.setText("状态: 手动精修模式关闭")

        self.image_viewer.set_state(session.state)
        self._update_history_buttons()

    def set_manual_brush_mode(self, mode: str, update_status: bool = True) -> None:
        session = self.current_session
        if not session:
            return
        if mode not in ["add", "erase"]:
            return
        session.state.manual_brush_mode = mode
        self.manual_add_button.blockSignals(True)
        self.manual_erase_button.blockSignals(True)
        self.manual_add_button.setChecked(mode == "add")
        self.manual_erase_button.setChecked(mode == "erase")
        self.manual_add_button.blockSignals(False)
        self.manual_erase_button.blockSignals(False)
        if update_status:
            action = "加入前景" if mode == "add" else "从前景抠除"
            self.status_label.setText(f"状态: 多边形模式 - {action}")

    def cancel_polygon(self) -> None:
        session = self.current_session
        if not session:
            return
        if not session.state.current_polygon_points:
            return
        session.state.current_polygon_points = []
        self.image_viewer.set_state(session.state)
        self.status_label.setText("状态: 已取消当前多边形")

    def toggle_hint_mask(self) -> None:
        session = self.current_session
        if not session:
            return
        session.state.mask_layers.show_hint = not session.state.mask_layers.show_hint
        self._sync_hint_button(session.state.mask_layers.show_hint)
        # Render without mutating underlying image; hint layer remains data-only
        self.image_viewer.set_state(session.state)

    def on_image_clicked(self, pos: QPoint) -> None:
        session = self.current_session
        if not session:
            return
        if session.state.manual_edit_enabled:
            self._handle_manual_polygon_click(session, pos)
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
            if not self.engine:
                show_warning(self, "提示", "当前未加载 SAM 模型，无法执行交互分割。")
                return
            self.status_label.setText("状态: 正在执行分割...")
            mask = run_segmentation(session, self.engine)
            if mask is None:
                self.status_label.setText("状态: 没有足够的点击点执行分割")
                return
            # 分割结果作为 auto_mask，关闭手动模式时清空手动掩膜
            session.state.auto_mask = mask.astype(bool)
            if not session.state.manual_edit_enabled:
                session.state.manual_mask = None
                session.state.current_polygon_points = []
            self.status_label.setText("状态: 分割完成")
        except Exception as exc:
            show_error(self, "错误", f"分割失败: {exc}")
            self.status_label.setText("状态: 分割失败")

    def _handle_manual_polygon_click(self, session: Session, pos: QPoint) -> None:
        """手动模式下左键添加多边形顶点，闭合后生成掩膜。"""
        image_pos = self.image_viewer.map_to_image_coords(pos)
        if not image_pos:
            return
        point = (int(image_pos.x()), int(image_pos.y()))
        points = session.state.current_polygon_points

        # 距离第一个点足够近则视为闭合，直接落在第一点位置
        if points and len(points) >= 3:
            first = points[0]
            dx = point[0] - first[0]
            dy = point[1] - first[1]
            if dx * dx + dy * dy <= 100:  # 10 像素阈值
                session.save()
                self._commit_polygon(session)
                return

        session.save()
        session.state.current_polygon_points.append([point[0], point[1]])
        self.image_viewer.set_state(session.state)
        self.status_label.setText("状态: 已添加多边形顶点")
        self._update_history_buttons()

    def _commit_polygon(self, session: Session) -> None:
        """闭合多边形生成掩膜，按 add/erase 逻辑与当前手动掩膜合并。"""
        poly_points = [tuple(pt) for pt in session.state.current_polygon_points]
        if len(poly_points) < 3:
            return

        width, height = session.state.original_image.size
        mask_img = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask_img)
        draw.polygon(poly_points, outline=1, fill=1)
        poly_mask = np.array(mask_img, dtype=bool)

        if session.state.manual_mask is None:
            session.state.manual_mask = np.zeros((height, width), dtype=bool)
        if session.state.manual_brush_mode == "erase":
            session.state.manual_mask = np.logical_and(session.state.manual_mask, ~poly_mask)
        else:
            session.state.manual_mask = np.logical_or(session.state.manual_mask, poly_mask)

        # 手动掩膜优先作为最终结果显示/保存
        session.state.current_polygon_points = []
        session.state.mask_layers.result = session.state.manual_mask
        if session.state.mask_layers.show_hint:
            session.state.mask_layers.hint = session.state.manual_mask

        self.image_viewer.set_state(session.state)
        action = "加入前景" if session.state.manual_brush_mode == "add" else "从前景抠除"
        self.status_label.setText(f"状态: 多边形已闭合，{action}")
        self._update_history_buttons()

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
        session.state.auto_mask = None
        session.state.manual_mask = None
        session.state.manual_edit_enabled = False
        session.state.current_polygon_points = []
        session.state.manual_brush_mode = "add"
        session.state.mask_layers.hint = None
        session.state.display_image = session.state.original_image.copy()

        self.image_viewer.set_state(session.state)
        self._sync_manual_controls(session.state)
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
            self._sync_hint_button(False)
            self._enable_manual_controls(False)
            self.manual_mode_button.setEnabled(False)
            self.manual_mode_button.setChecked(False)
            return
        self.undo_button.setEnabled(session.can_undo())
        self.redo_button.setEnabled(session.can_redo())
        self.manual_mode_button.setEnabled(True)

    def _sync_hint_button(self, show: bool) -> None:
        self.toggle_hint_button.blockSignals(True)
        self.toggle_hint_button.setChecked(show)
        self.toggle_hint_button.setText("隐藏提示掩膜" if show else "显示提示掩膜")
        self.toggle_hint_button.setEnabled(bool(self.sessions))
        self.toggle_hint_button.blockSignals(False)

    def _enable_manual_controls(self, enabled: bool) -> None:
        self.manual_add_button.setEnabled(enabled)
        self.manual_erase_button.setEnabled(enabled)
        self.cancel_polygon_button.setEnabled(enabled)

    def _sync_manual_controls(self, state: ImageState) -> None:
        self.manual_mode_button.blockSignals(True)
        self.manual_mode_button.setEnabled(bool(self.sessions))
        self.manual_mode_button.setChecked(state.manual_edit_enabled)
        self.manual_mode_button.blockSignals(False)
        self._enable_manual_controls(state.manual_edit_enabled)
        if state.manual_edit_enabled:
            self.set_manual_brush_mode(state.manual_brush_mode, update_status=False)
        else:
            self.manual_add_button.setChecked(False)
            self.manual_erase_button.setChecked(False)

    # ---------------- Model registry + tasks ----------------
    def _build_model_menu(self) -> None:
        menu = QMenu(self)
        nnunet_menu = menu.addMenu("nnU-Net 模型管理")
        actions = {
            "install_default": nnunet_menu.addAction("安装默认模型", self.install_default_model),
            "install_zip": nnunet_menu.addAction("安装模型包", self.install_model_zip),
            "import_nnunet": nnunet_menu.addAction("导入原生nnU-Net", self.import_nnunet_native),
            "run_nnunet": nnunet_menu.addAction("nnU-Net推理", self.run_nnunet_inference),
            "train_nnunet": nnunet_menu.addAction("nnU-Net训练", self.start_nnunet_training),
            "nnunet_settings": nnunet_menu.addAction("nnU-Net设置", self.configure_nnunet_commands),
            "open_dir": nnunet_menu.addAction("打开模型目录", self.open_selected_model_dir),
            "remove": nnunet_menu.addAction("移除模型", self.remove_selected_model),
        }
        actions["run_nnunet"].setEnabled(False)
        actions["remove"].setEnabled(False)
        self.menu_actions = actions
        self.model_menu_button.setMenu(menu)

    def _load_nnunet_settings(self) -> None:
        if NNUNET_SETTINGS_FILE.exists():
            try:
                data = json.loads(NNUNET_SETTINGS_FILE.read_text(encoding="utf-8"))
                self.nnunet_predict_cmd = data.get("predict_cmd", self.nnunet_predict_cmd)
                self.nnunet_train_cmd = data.get("train_cmd", self.nnunet_train_cmd)
            except Exception:
                # ignore parse error, keep defaults
                pass

    def _save_nnunet_settings(self) -> None:
        payload = {"predict_cmd": self.nnunet_predict_cmd, "train_cmd": self.nnunet_train_cmd}
        NNUNET_SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        NNUNET_SETTINGS_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def configure_nnunet_commands(self) -> None:
        predict_cmd, ok = QInputDialog.getText(
            self, "配置 nnU-Net 命令", "预测命令 (nnUNetv2_predict):", text=self.nnunet_predict_cmd
        )
        if not ok:
            return
        train_cmd, ok = QInputDialog.getText(
            self, "配置 nnU-Net 命令", "训练命令 (nnUNetv2_train):", text=self.nnunet_train_cmd
        )
        if not ok:
            return
        if predict_cmd:
            self.nnunet_predict_cmd = predict_cmd
        if train_cmd:
            self.nnunet_train_cmd = train_cmd
        self._save_nnunet_settings()
        self.nnunet_adapter = NNUNetAdapter(self.registry, predict_cmd=self.nnunet_predict_cmd)
        self.train_runner = TrainRunner(self.registry, train_cmd=self.nnunet_train_cmd)
        show_info(self, "完成", "nnU-Net 命令已更新")

    def _refresh_model_list(self) -> None:
        self.registry.refresh()
        records = self.registry.list_models()
        self.model_records = {}
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        for rec in records:
            model_id = rec.id or rec.path.name
            self.model_records[model_id] = rec
            if rec.manifest:
                label = f"{rec.manifest.name} ({rec.manifest.type})"
            else:
                label = f"{model_id} [损坏]"
            self.model_combo.addItem(label, userData=model_id)
            tooltip = rec.error or label
            self.model_combo.setItemData(self.model_combo.count() - 1, tooltip, Qt.ItemDataRole.ToolTipRole)
        self.model_combo.addItem("SAM (config)", userData="__legacy_sam__")
        self.model_combo.blockSignals(False)
        if self.model_combo.count() > 0:
            target_index = 0
            if self.current_model_id:
                for idx in range(self.model_combo.count()):
                    if self.model_combo.itemData(idx) == self.current_model_id:
                        target_index = idx
                        break
            self.model_combo.setCurrentIndex(target_index)
            self._on_model_selected(target_index)
        else:
            self.current_model_id = None
            self.update_info_bar()

    def _on_model_selected(self, index: int) -> None:
        if index < 0:
            return
        model_id = self.model_combo.itemData(index)
        self.model_combo.setToolTip(self.model_combo.currentText())
        self.current_model_id = model_id
        # 更新菜单可用性
        if hasattr(self, "menu_actions"):
            self.menu_actions["remove"].setEnabled(bool(model_id and model_id != "__legacy_sam__"))
            self.menu_actions["open_dir"].setEnabled(True)
            self.menu_actions["run_nnunet"].setEnabled(False)
        if model_id == "__legacy_sam__":
            if self.engine is None:
                try:
                    self.engine = SamEngine()
                except Exception as exc:
                    show_error(self, "错误", f"加载默认 SAM 模型失败: {exc}")
            self.status_label.setText("状态: 使用配置中的 SAM 模型")
            self.update_info_bar()
            return
        record = self.model_records.get(model_id, None)
        if not record:
            return
        if record.error:
            self.status_label.setText(f"状态: 模型损坏 - {record.error}")
            self.update_info_bar()
            return
        manifest = record.manifest
        if not manifest:
            return
        if manifest.type.lower() == "sam":
            try:
                checkpoint_path = record.path / manifest.entry
                model_type = manifest.extra.get("model_type", MODEL_TYPE)
                self.engine = SamEngine(checkpoint=checkpoint_path, model_type=model_type)
                self.status_label.setText(f"状态: 已加载 SAM 模型 {manifest.name}")
            except Exception as exc:
                show_error(self, "错误", f"加载 SAM 模型失败: {exc}")
        elif manifest.type.lower() == "nnunet":
            self.engine = None
            if hasattr(self, "menu_actions"):
                self.menu_actions["run_nnunet"].setEnabled(True)
            self.status_label.setText(f"状态: 已选择 nnU-Net 模型 {manifest.name}")
        else:
            self.status_label.setText(f"状态: 已选择模型 {manifest.name}")
        self.update_info_bar()

    def _start_task(
        self, fn, on_success=None, error_title: str = "错误", progress_dialog: Optional[QProgressDialog] = None
    ):
        worker = TaskWorker(fn)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.status.connect(lambda msg: self._update_status(msg), Qt.ConnectionType.QueuedConnection)
        if progress_dialog:
            worker.progress.connect(progress_dialog.setValue, Qt.ConnectionType.QueuedConnection)
        worker.finished.connect(
            lambda result: self._task_finished(thread, worker, on_success, progress_dialog, result),
            Qt.ConnectionType.QueuedConnection,
        )
        worker.failed.connect(
            lambda msg: self._task_failed(thread, worker, error_title, progress_dialog, msg),
            Qt.ConnectionType.QueuedConnection,
        )
        thread.start()
        self._background_threads.append(thread)
        # Keep worker alive
        thread.worker = worker  # type: ignore[attr-defined]
        if progress_dialog:
            progress_dialog.setValue(0)
        return worker, thread

    def _task_finished(
        self,
        thread: QThread,
        worker: TaskWorker,
        on_success,
        progress_dialog: Optional[QProgressDialog],
        result,
    ) -> None:
        if progress_dialog:
            self._close_dialog_async(progress_dialog)
        thread.quit()
        if thread is not QThread.currentThread():
            thread.wait()
        if thread in self._background_threads:
            self._background_threads.remove(thread)
        if on_success:
            on_success(result)

    def _task_failed(
        self,
        thread: QThread,
        worker: TaskWorker,
        title: str,
        progress_dialog: Optional[QProgressDialog],
        message: str,
    ) -> None:
        if progress_dialog:
            self._close_dialog_async(progress_dialog)
        thread.quit()
        if thread is not QThread.currentThread():
            thread.wait()
        if thread in self._background_threads:
            self._background_threads.remove(thread)
        self._show_dialog_async("error", title, message)
        self.status_label.setText(f"状态: {message}")

    def _close_dialog_async(self, dialog: QProgressDialog) -> None:
        if dialog is None:
            return
        QMetaObject.invokeMethod(dialog, "close", Qt.ConnectionType.QueuedConnection)

    @Slot(str, str, str)
    def _show_dialog(self, kind: str, title: str, message: str) -> None:
        if kind == "error":
            show_error(self, title, message)
        elif kind == "warning":
            show_warning(self, title, message)
        else:
            show_info(self, title, message)

    def _show_dialog_async(self, kind: str, title: str, message: str) -> None:
        QMetaObject.invokeMethod(
            self,
            "_show_dialog",
            Qt.ConnectionType.QueuedConnection,
            Q_ARG(str, kind),
            Q_ARG(str, title),
            Q_ARG(str, message),
        )

    def _update_status(self, text: str) -> None:
        self.status_label.setText(f"状态: {text}")

    def _on_install_success(self, manifest) -> None:
        if manifest:
            self._show_dialog_async("info", "成功", f"模型 {manifest.name} 已安装")
        self._refresh_model_list()

    def install_default_model(self) -> None:
        try:
            sources = self.registry.list_default_sources()
        except Exception as exc:
            self._show_dialog_async("error", "错误", f"读取 sources 列表失败: {exc}")
            return
        if not sources:
            self._show_dialog_async("warning", "提示", "未找到 model_store/sources.yaml/json，无法安装默认模型。")
            return
        model_ids = [entry.get("model_id") or entry.get("id") for entry in sources]
        model_ids = [mid for mid in model_ids if mid]
        if not model_ids:
            self._show_dialog_async("warning", "提示", "sources 中没有可用的 model_id。")
            return
        from PySide6.QtWidgets import QInputDialog

        model_id, ok = QInputDialog.getItem(self, "安装默认模型", "选择模型:", model_ids, 0, False)
        if not ok or not model_id:
            return

        cancel_event = threading.Event()

        def task(worker: TaskWorker):
            def progress_cb(downloaded: int, total: Optional[int]) -> None:
                percent = int(downloaded * 100 / total) if total else 0
                worker.progress.emit(percent)
                worker.status.emit(f"下载中 {downloaded}/{total or '?'} bytes")

            try:
                manifest, error = self.registry.download_and_install_default(
                    model_id,
                    downloader=lambda url, dest: download_file(
                        url, dest, progress_cb=progress_cb, cancel_event=cancel_event, timeout=8.0
                    ),
                )
            except DownloadCancelled:
                raise RuntimeError("下载已取消")
            if error:
                raise RuntimeError(error)
            return manifest

        progress = QProgressDialog("正在下载模型...", "取消", 0, 100, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.canceled.connect(lambda: cancel_event.set(), Qt.ConnectionType.QueuedConnection)
        progress.show()
        worker, _ = self._start_task(task, on_success=self._on_install_success, progress_dialog=progress)

    def install_model_zip(self) -> None:
        path = choose_zip_file(self, "选择模型包 zip")
        if not path:
            return

        def task(worker: TaskWorker):
            worker.status.emit("正在安装模型包...")
            return self.registry.install_from_zip(Path(path))

        self._start_task(task, on_success=self._on_install_success)

    def import_nnunet_native(self) -> None:
        choice = QMessageBox(self)
        choice.setWindowTitle("导入 nnU-Net")
        choice.setText("选择 nnU-Net 结果来源类型")
        choice.setIcon(QMessageBox.Icon.Question)
        btn_zip = choice.addButton("zip 文件", QMessageBox.ButtonRole.ActionRole)
        btn_dir = choice.addButton("目录", QMessageBox.ButtonRole.ActionRole)
        choice.addButton("取消", QMessageBox.ButtonRole.RejectRole)
        choice.exec()

        clicked = choice.clickedButton()
        if clicked is btn_zip:
            path = choose_zip_file(self, "选择 nnU-Net zip")
        elif clicked is btn_dir:
            path = choose_folder(self, "选择 nnU-Net 目录")
        else:
            path = ""
        if not path:
            return

        def task(worker: TaskWorker):
            worker.status.emit("正在导入 nnU-Net 结果...")
            return self.registry.import_nnunet_native(Path(path))

        self._start_task(task, on_success=self._on_install_success)

    def remove_selected_model(self) -> None:
        if not self.current_model_id or self.current_model_id == "__legacy_sam__":
            return
        try:
            self.registry.remove(self.current_model_id)
            self.status_label.setText(f"状态: 已移除模型 {self.current_model_id}")
            self._refresh_model_list()
        except Exception as exc:
            show_error(self, "错误", f"移除模型失败: {exc}")

    def open_selected_model_dir(self) -> None:
        if not self.current_model_id or self.current_model_id == "__legacy_sam__":
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(MODELS_DIR)))
            return
        record = self.model_records.get(self.current_model_id)
        if not record:
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(record.path)))

    def run_nnunet_inference(self) -> None:
        if not self.current_model_id:
            show_warning(self, "提示", "请先选择 nnU-Net 模型")
            return
        record = self.model_records.get(self.current_model_id)
        if not record or not record.manifest or record.manifest.type.lower() != "nnunet":
            show_warning(self, "提示", "当前选择的模型不是 nnU-Net 类型")
            return
        folder = choose_folder(self, "选择 PNG 序列文件夹")
        if not folder:
            return
        try:
            volume = self.volume_loader.load_png_stack(folder)
        except Exception as exc:
            show_error(self, "错误", f"加载 PNG 序列失败: {exc}")
            return

        def task(worker: TaskWorker):
            return self.nnunet_adapter.predict(
                self.current_model_id,
                volume,
                case_id=volume.case_id or "case",
                progress_cb=lambda msg: worker.status.emit(msg),
                log_cb=lambda line: worker.log.emit(line),
            )

        self._start_task(task, on_success=self._handle_nnunet_result)

    def start_nnunet_training(self) -> None:
        dataset_id, ok = QInputDialog.getText(self, "启动 nnU-Net 训练", "Dataset ID (数字或名称):", text="")
        if not ok or not dataset_id:
            return
        configuration, ok = QInputDialog.getItem(
            self, "网络配置", "选择 2D/3D 配置:", ["3d_fullres", "2d"], 0, False
        )
        if not ok:
            return
        device, _ = QInputDialog.getText(self, "设备 (可选)", "如 cuda:0 / cpu，留空自动选择:", text="")
        fast_choice, ok = QInputDialog.getItem(self, "快速/开发模式", "是否启用 fast/dev (--npz):", ["否", "是"], 0, False)
        fast_mode = ok and fast_choice == "是"

        cfg = TrainConfig(
            dataset_id=dataset_id,
            configuration=configuration,
            device=device or None,
            fast=fast_mode,
            meta={
                "name": f"{dataset_id}_{configuration}",
                "capabilities": ["3d"] if "3d" in configuration else ["2d"],
                "input_format": "nifti",
            },
        )

        def task(worker: TaskWorker):
            return self.train_runner.run(cfg, log_cb=lambda line: worker.status.emit(line))

        self._start_task(task, on_success=self._on_install_success)

    def _handle_nnunet_result(self, prediction) -> None:
        if not prediction:
            return
        if prediction.mask_volume is not None and self.current_session:
            mask_vol = prediction.mask_volume
            slice_idx = mask_vol.shape[0] // 2
            mask2d = mask_vol[slice_idx]
            # 维度检查
            state = self.current_session.state
            if mask2d.shape != (state.original_image.height, state.original_image.width):
                show_warning(self, "提示", "nnU-Net 输出尺寸与当前图片不一致，跳过自动覆盖。")
            else:
                state.auto_mask = mask2d.astype(bool)
                state.mask_layers.result = state.auto_mask
                self.image_viewer.set_state(state)
        show_info(self, "完成", f"nnU-Net 推理完成，输出文件: {prediction.output_file}")
        self.status_label.setText("状态: nnU-Net 推理完成")

    def _apply_styles(self, sidebar: QWidget, header: QWidget) -> None:
        """统一顶部工具栏与左侧侧边栏的简单美化样式。"""
        header.setStyleSheet(
            """
            #HeaderBar {
                background-color: #f5f5f7;
                border-bottom: 1px solid #ddd;
            }
            QToolButton#TopToolButton {
                border: 1px solid transparent;
                border-radius: 8px;
                padding: 6px 10px;
                background: transparent;
            }
            QToolButton#TopToolButton:hover {
                border: 1px solid #9ae6b4;
            }
            QToolButton#TopToolButton:pressed,
            QToolButton#TopToolButton:checked {
                border: 1px solid #047857;
            }
            """
        )
        sidebar.setStyleSheet(
            """
            #SideBar {
                background-color: #f8f9fb;
                border-right: 1px solid #e0e0e0;
            }
            QToolButton#SideToolButton {
                border: 1px solid transparent;
                border-radius: 8px;
                padding: 8px 10px;
                margin: 4px 2px;
                background: transparent;
            }
            QToolButton#SideToolButton:hover {
                border: 1px solid #9ae6b4;
            }
            QToolButton#SideToolButton:pressed,
            QToolButton#SideToolButton:checked {
                border: 1px solid #047857;
            }
            """
        )

    def _setup_top_button(self, button: QToolButton, text: str, icon_path: Optional[Path]) -> None:
        """顶部工具栏按钮：图标+文字下方，矩形边框。"""
        button.setObjectName("TopToolButton")
        button.setText(text)
        button.setIcon(QIcon(str(icon_path)) if icon_path else QIcon())
        button.setIconSize(QSize(22, 22))
        button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        button.setFixedHeight(60)

    def _setup_side_button(
        self,
        button: QToolButton,
        text: str,
        icon_path: Optional[Path],
        icon_size: QSize,
    ) -> None:
        """侧边栏按钮：图标+文字右侧，矩形边框。"""
        button.setObjectName("SideToolButton")
        button.setText(text)
        button.setIcon(QIcon(str(icon_path)) if icon_path else QIcon())
        button.setIconSize(icon_size)
        button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        button.setMinimumHeight(48)

    def resizeEvent(self, event):  # type: ignore[override]
        super().resizeEvent(event)
        self.update_image_display()

    def keyPressEvent(self, event):  # type: ignore[override]
        if event.key() == Qt.Key_Escape:
            # Esc 取消当前多边形，但不影响已有掩膜
            self.cancel_polygon()
            event.accept()
            return
        super().keyPressEvent(event)


__all__ = ["MainWindow"]
