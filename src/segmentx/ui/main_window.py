from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image, ImageDraw
from PySide6.QtCore import QPoint, Qt, QSize
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QSizePolicy,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..config import MAX_HISTORY, MODEL_TYPE, RESOURCES_DIR
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
from .image_viewer import AspectRatioContainer, ImageViewer
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
        self.image_info_label = QLabel("当前图片: 0/0")

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
        self.title_label = QLabel("SegmentX")
        self.title_label.setStyleSheet("font-size: 18px; font-weight: 700; color: #1f2937;")
        self.model_label = QLabel("当前模型：--")
        self.model_label.setStyleSheet("font-size: 13px; color: #4b5563;")
        self.image_count_label = QLabel("图片：0/0")
        self.image_count_label.setStyleSheet("font-size: 13px; color: #4b5563;")
        info_layout.addWidget(self.title_label)
        info_layout.addWidget(self.model_label)
        info_layout.addWidget(self.image_count_label)
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

        side_layout.addStretch()
        side_layout.addWidget(self.image_info_label)
        side_layout.addWidget(self.status_label)
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

        self._apply_styles(sidebar, header_bar)
        self.update_info_bar()

    @property
    def current_session(self) -> Optional[Session]:
        if 0 <= self.current_index < len(self.sessions):
            return self.sessions[self.current_index]
        return None

    def load_images(self) -> None:
        file_paths = choose_images(self)
        if not file_paths:
            return

        try:
            self.engine.clear_cache()
            self.sessions = []
            self.current_index = -1
            self.manual_mode_button.setEnabled(False)
            self.manual_mode_button.setChecked(False)
            self._enable_manual_controls(False)
            for path in file_paths:
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
                self._set_engine_image(self.current_session.state.original_image, self.current_session.state.path)  # type: ignore[union-attr]
                self._update_navigation_buttons()
                self.save_button.setEnabled(True)
                self.save_all_button.setEnabled(True)
                self.manual_mode_button.setEnabled(True)
                self.status_label.setText(f"状态: 已加载 {len(self.sessions)} 张图片")
                self._update_image_info()
        except Exception as exc:
            show_error(self, "错误", f"加载图像失败: {exc}")
            self.status_label.setText("状态: 图像加载失败")

    def _set_engine_image(self, image, image_id: str) -> bool:
        # 为多图切换复用缓存的SAM embedding，减少重复编码开销
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
        if self.sessions:
            self.image_info_label.setText(f"当前图片: {self.current_index + 1}/{len(self.sessions)}")
        else:
            self.image_info_label.setText("当前图片: 0/0")

    def update_info_bar(self) -> None:
        """更新顶部信息栏：模型名称 + 图片进度。"""
        model_name = getattr(self.engine, "model_type", None) or getattr(self.engine, "model_name", None) or MODEL_TYPE
        self.model_label.setText(f"当前模型：{model_name}")
        if self.sessions and 0 <= self.current_index < len(self.sessions):
            self.image_count_label.setText(f"图片：{self.current_index + 1}/{len(self.sessions)}")
        else:
            self.image_count_label.setText("图片：0/0")

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
