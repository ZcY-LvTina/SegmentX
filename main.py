# This Python file uses the following encoding: utf-8
import os
import sys
import numpy as np
from PIL import Image, ImageQt, ImageDraw
import torch
import cv2

from segment_anything import sam_model_registry, SamPredictor

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QLabel, QVBoxLayout,
    QHBoxLayout, QWidget, QPushButton, QScrollArea, QMessageBox
)
from PySide6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QMouseEvent, QKeySequence, QAction
from PySide6.QtCore import Qt, QPoint, QSize, Signal


class ClickableLabel(QLabel):
    clicked = Signal(QPoint)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(event.position().toPoint())


class MedicalImageSegmentationApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # 初始化UI
        self.init_ui()

        # 初始化SAM模型
        self.init_sam_model()

        # 初始化变量
        self.image_paths = []  # 所有图片路径
        self.current_image_index = -1  # 当前图片索引
        self.image_data = {}  # 存储每张图片的数据

        # 添加撤销/重做功能
        self.undo_stack = []  # 撤销栈
        self.redo_stack = []  # 重做栈
        self.max_history = 20  # 最大历史记录数

        # 设置窗口标题和大小
        self.setWindowTitle("Segment")
        self.resize(1200, 800)

    def init_ui(self):
        """初始化用户界面"""
        # 主窗口部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主布局
        main_layout = QHBoxLayout(central_widget)

        # 左侧控制面板
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # 初始化所有按钮和控件
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

        # 设置按钮属性
        self.undo_button.setEnabled(False)
        self.redo_button.setEnabled(False)
        self.prev_button.setEnabled(False)
        self.next_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.save_all_button.setEnabled(False)

        self.foreground_button.setCheckable(True)
        self.foreground_button.setChecked(True)
        self.background_button.setCheckable(True)

        # 连接信号
        self.load_button.clicked.connect(lambda: self.load_image(single=True))
        self.load_multiple_button.clicked.connect(lambda: self.load_image(single=False))
        self.prev_button.clicked.connect(self.prev_image)
        self.next_button.clicked.connect(self.next_image)
        self.clear_button.clicked.connect(self.clear_markers)
        self.save_button.clicked.connect(self.save_current_result)
        self.save_all_button.clicked.connect(self.save_all_results)
        self.undo_button.clicked.connect(self.undo_action)  # 修改为 undo_action
        self.redo_button.clicked.connect(self.redo_action)  # 修改为 redo_action
        self.foreground_button.clicked.connect(self.set_foreground_mode)
        self.background_button.clicked.connect(self.set_background_mode)

        # 添加到控制面板布局
        control_layout.addWidget(self.load_button)
        control_layout.addWidget(self.load_multiple_button)
        control_layout.addWidget(self.prev_button)
        control_layout.addWidget(self.next_button)
        control_layout.addWidget(self.clear_button)
        control_layout.addWidget(self.save_button)
        control_layout.addWidget(self.save_all_button)
        control_layout.addWidget(self.undo_button)
        control_layout.addWidget(self.redo_button)
        control_layout.addWidget(self.foreground_button)
        control_layout.addWidget(self.background_button)
        control_layout.addWidget(self.image_info_label)
        control_layout.addWidget(self.status_label)

        # 添加到主布局
        main_layout.addWidget(control_panel, stretch=1)

        # 右侧图像显示区域
        image_panel = QWidget()
        image_layout = QVBoxLayout(image_panel)

        # 图像显示标签
        self.image_label = ClickableLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(800, 600)
        self.image_label.clicked.connect(self.on_image_clicked)

        # 添加滚动区域以便处理大图像
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.image_label)
        image_layout.addWidget(scroll_area)

        # 添加到主布局
        main_layout.addWidget(image_panel, stretch=4)

        # 设置当前标记模式
        self.current_mode = "foreground"

        # 添加快捷键
        undo_action = QAction("撤销", self)
        undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        undo_action.triggered.connect(self.undo_action)  # 修改为 undo_action
        self.addAction(undo_action)

        redo_action = QAction("重做", self)
        redo_action.setShortcut(QKeySequence.StandardKey.Redo)
        redo_action.triggered.connect(self.redo_action)  # 修改为 redo_action
        self.addAction(redo_action)

        prev_action = QAction("上一张", self)
        prev_action.setShortcut(QKeySequence(Qt.Key_Left))
        prev_action.triggered.connect(self.prev_image)
        self.addAction(prev_action)

        next_action = QAction("下一张", self)
        next_action.setShortcut(QKeySequence(Qt.Key_Right))
        next_action.triggered.connect(self.next_image)
        self.addAction(next_action)

    def load_image(self, single=True):
        """加载图像文件"""
        options = QFileDialog.Option.ReadOnly
        if single:
            file_paths, _ = QFileDialog.getOpenFileName(
                self, "选择医学影像", "",
                "图像文件 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;所有文件 (*)",
                options=options
            )
            file_paths = [file_paths] if file_paths else []
        else:
            file_paths, _ = QFileDialog.getOpenFileNames(
                self, "选择多个医学影像", "",
                "图像文件 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;所有文件 (*)",
                options=options
            )

        if file_paths:
            try:
                # 清空当前数据
                self.image_paths = []
                self.image_data = {}
                self.current_image_index = -1

                # 加载所有图片
                for file_path in file_paths:
                    # 使用PIL加载图像，确保无损画质
                    original_image = Image.open(file_path)

                    # 转换为RGB模式（如果是灰度图像）
                    if original_image.mode != "RGB":
                        original_image = original_image.convert("RGB")

                    # 存储图片数据
                    self.image_paths.append(file_path)
                    self.image_data[file_path] = {
                        'original_image': original_image,
                        'display_image': original_image.copy(),
                        'click_points': [],
                        'labels': [],
                        'mask': None
                    }

                # 显示第一张图片
                if self.image_paths:
                    self.current_image_index = 0
                    self.show_current_image()

                    # 为SAM模型设置图像
                    if self.set_image_for_sam(np.array(self.current_data['original_image'])):
                        self.status_label.setText(f"状态: 已加载 {len(self.image_paths)} 张图片")
                    else:
                        self.status_label.setText("状态: 图像加载完成但SAM模型设置失败")

                    # 更新按钮状态
                    self.update_navigation_buttons()
                    self.save_button.setEnabled(True)
                    self.save_all_button.setEnabled(True)

                    # 更新状态
                    self.status_label.setText(f"状态: 已加载 {len(self.image_paths)} 张图片")
                    self.update_image_info()

            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载图像失败: {str(e)}")
                self.status_label.setText("状态: 图像加载失败")

    @property
    def current_data(self):
        """获取当前图片的数据"""
        if self.current_image_index >= 0 and self.current_image_index < len(self.image_paths):
            return self.image_data[self.image_paths[self.current_image_index]]
        return None

    def show_current_image(self):
        """显示当前图片"""
        if not self.current_data:
            return

        # 更新显示
        self.update_image_display()

        # 更新撤销/重做栈
        self.undo_stack = []
        self.redo_stack = []
        self.undo_button.setEnabled(False)
        self.redo_button.setEnabled(False)

    def prev_image(self):
        """切换到上一张图片"""
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.show_current_image()

            # 为SAM模型设置图像
            if not self.set_image_for_sam(np.array(self.current_data['original_image'])):
                self.status_label.setText("状态: 切换到图片但SAM模型设置失败")

            # 更新按钮状态
            self.update_navigation_buttons()
            self.update_image_info()

            self.status_label.setText(f"状态: 切换到上一张图片")

    def next_image(self):
        """切换到下一张图片"""
        if self.current_image_index < len(self.image_paths) - 1:
            self.current_image_index += 1
            self.show_current_image()

            # 为SAM模型设置图像
            if not self.set_image_for_sam(np.array(self.current_data['original_image'])):
                self.status_label.setText("状态: 切换到图片但SAM模型设置失败")

            # 更新按钮状态
            self.update_navigation_buttons()
            self.update_image_info()

            self.status_label.setText(f"状态: 切换到下一张图片")

    def update_navigation_buttons(self):
        """更新导航按钮状态"""
        self.prev_button.setEnabled(self.current_image_index > 0)
        self.next_button.setEnabled(self.current_image_index < len(self.image_paths) - 1)

    def update_image_info(self):
        """更新图片信息标签"""
        if self.image_paths:
            self.image_info_label.setText(f"当前图片: {self.current_image_index + 1}/{len(self.image_paths)}")
        else:
            self.image_info_label.setText("当前图片: 0/0")

    def save_current_result(self):
        """保存当前图片的分割结果"""
        if not self.current_data or self.current_data['mask'] is None:
            QMessageBox.warning(self, "警告", "没有可保存的分割结果")
            return

        options = QFileDialog.Option.DontUseNativeDialog
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self, "保存结果", "",
            "PNG图像 (*.png);;JPEG图像 (*.jpg *.jpeg);;BMP图像 (*.bmp);;TIFF图像 (*.tif *.tiff);;所有文件 (*)",
            options=options
        )

        if file_path:
            try:
                # 根据选择的过滤器确定文件格式
                if selected_filter:
                    if "PNG" in selected_filter:
                        format = "PNG"
                        if not file_path.lower().endswith('.png'):
                            file_path += '.png'
                    elif "JPEG" in selected_filter:
                        format = "JPEG"
                        if not file_path.lower().endswith(('.jpg', '.jpeg')):
                            file_path += '.jpg'
                    elif "BMP" in selected_filter:
                        format = "BMP"
                        if not file_path.lower().endswith('.bmp'):
                            file_path += '.bmp'
                    elif "TIFF" in selected_filter:
                        format = "TIFF"
                        if not file_path.lower().endswith(('.tif', '.tiff')):
                            file_path += '.tif'
                    else:
                        # 默认使用PNG格式
                        format = "PNG"
                        if not any(file_path.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']):
                            file_path += '.png'
                else:
                    # 如果没有选择过滤器，根据文件扩展名确定格式
                    if file_path.lower().endswith('.png'):
                        format = "PNG"
                    elif file_path.lower().endswith(('.jpg', '.jpeg')):
                        format = "JPEG"
                    elif file_path.lower().endswith('.bmp'):
                        format = "BMP"
                    elif file_path.lower().endswith(('.tif', '.tiff')):
                        format = "TIFF"
                    else:
                        # 默认使用PNG格式
                        format = "PNG"
                        file_path += '.png'

                # 创建结果图像
                result_image = self.current_data['original_image'].copy()

                # 应用mask
                mask_image = Image.fromarray((self.current_data['mask'] * 255).astype(np.uint8), mode="L")
                color_mask = Image.new("RGBA", result_image.size, (0, 255, 0, 128))

                # 如果原始图像不是RGBA，转换为RGBA以支持透明度
                if result_image.mode != 'RGBA':
                    result_image = result_image.convert('RGBA')

                result_image.paste(color_mask, (0, 0), mask_image)

                # 保存图像
                result_image.save(file_path, format=format)
                self.status_label.setText(f"状态: 结果已保存到 {os.path.basename(file_path)}")

            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存结果失败: {str(e)}")
                self.status_label.setText("状态: 保存失败")

    def undo_action(self):
        """撤销上一步操作"""
        if len(self.undo_stack) == 0 or not self.current_data:
            return

        # 获取当前状态并保存到重做栈
        current_state = {
            'click_points': self.current_data['click_points'].copy(),
            'labels': self.current_data['labels'].copy(),
            'mask': self.current_data['mask'].copy() if self.current_data['mask'] is not None else None,
            'display_image': self.current_data['display_image'].copy() if self.current_data['display_image'] else None
        }
        self.redo_stack.append(current_state)

        # 恢复上一个状态
        last_state = self.undo_stack.pop()
        self.current_data['click_points'] = last_state['click_points']
        self.current_data['labels'] = last_state['labels']
        self.current_data['mask'] = last_state['mask']
        self.current_data['display_image'] = last_state['display_image']

        # 更新显示
        self.update_image_display()

        # 更新按钮状态
        self.undo_button.setEnabled(len(self.undo_stack) > 0)
        self.redo_button.setEnabled(len(self.redo_stack) > 0)

        self.status_label.setText("状态: 已撤销上一步操作")

    def redo_action(self):
        """重做被撤销的操作"""
        if len(self.redo_stack) == 0 or not self.current_data:
            return

        # 保存当前状态到撤销栈
        current_state = {
            'click_points': self.current_data['click_points'].copy(),
            'labels': self.current_data['labels'].copy(),
            'mask': self.current_data['mask'].copy() if self.current_data['mask'] is not None else None,
            'display_image': self.current_data['display_image'].copy() if self.current_data['display_image'] else None
        }
        self.undo_stack.append(current_state)

        # 恢复重做状态
        redo_state = self.redo_stack.pop()
        self.current_data['click_points'] = redo_state['click_points']
        self.current_data['labels'] = redo_state['labels']
        self.current_data['mask'] = redo_state['mask']
        self.current_data['display_image'] = redo_state['display_image']

        # 更新显示
        self.update_image_display()

        # 更新按钮状态
        self.undo_button.setEnabled(len(self.undo_stack) > 0)
        self.redo_button.setEnabled(len(self.redo_stack) > 0)

        self.status_label.setText("状态: 已重做操作")

    def init_sam_model(self):
        """初始化SAM模型"""
        try:
            # 检查模型文件是否存在
            model_path = os.path.join("models", "sam_vit_h_4b8939.pth")
            if not os.path.exists(model_path):
                QMessageBox.critical(self, "错误", f"模型文件未找到: {model_path}")
                return

            # 设置设备
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.status_label.setText(f"状态: 正在加载SAM模型 (设备: {self.device})")
            QApplication.processEvents()  # 更新UI

            # 加载模型
            sam = sam_model_registry["vit_h"](checkpoint=model_path)
            sam.to(device=self.device)
            self.predictor = SamPredictor(sam)

            self.status_label.setText(f"状态: SAM模型加载完成 (设备: {self.device})")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载SAM模型失败: {str(e)}")
            self.status_label.setText("状态: 模型加载失败")

    def set_image_for_sam(self, image_array):
        """为SAM模型设置当前图像"""
        try:
            if hasattr(self, 'predictor') and self.predictor is not None:
                # 将图像转换为RGB格式（如果必要）
                if len(image_array.shape) == 2:  # 灰度图像
                    image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
                elif image_array.shape[2] == 4:  # RGBA图像
                    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)

                # 设置图像到SAM模型
                self.predictor.set_image(image_array)
                return True
            else:
                self.status_label.setText("状态: SAM模型未初始化")
                return False
        except Exception as e:
            QMessageBox.critical(self, "错误", f"设置图像到SAM模型失败: {str(e)}")
            self.status_label.setText("状态: 设置图像到SAM模型失败")
            return False

    def save_state_to_history(self):
        """保存当前状态到历史记录"""
        if not self.current_data:
            return

        # 如果撤销栈已满，移除最旧的状态
        if len(self.undo_stack) >= self.max_history:
            self.undo_stack.pop(0)

        # 保存当前状态到撤销栈
        state = {
            'click_points': self.current_data['click_points'].copy(),
            'labels': self.current_data['labels'].copy(),
            'mask': self.current_data['mask'].copy() if self.current_data['mask'] is not None else None,
            'display_image': self.current_data['display_image'].copy() if self.current_data['display_image'] else None
        }
        self.undo_stack.append(state)

        # 清空重做栈
        self.redo_stack.clear()

        # 更新按钮状态
        self.undo_button.setEnabled(len(self.undo_stack) > 0)
        self.redo_button.setEnabled(False)

    def on_image_clicked(self, pos):
        """处理图像点击事件"""
        if not self.current_data:
            return

        # 获取点击位置（相对于图像）
        image_pos = self.get_image_position(pos)

        if image_pos:
            x, y = image_pos.x(), image_pos.y()

            # 保存当前状态到历史记录
            self.save_state_to_history()

            # 添加点击点到列表
            self.current_data['click_points'].append([x, y])
            self.current_data['labels'].append(1 if self.current_mode == "foreground" else 0)

            # 执行分割
            self.perform_segmentation()

            # 更新显示
            self.update_image_display()

    def get_image_position(self, pos):
        """精确处理居中缩放图像的坐标转换"""
        if not self.current_data or not self.image_label.pixmap():
            return None

        # 获取关键尺寸
        label_size = self.image_label.size()
        pixmap = self.image_label.pixmap()
        pixmap_size = pixmap.size()
        img_width, img_height = self.current_data['display_image'].width, self.current_data['display_image'].height

        # 计算实际显示区域（保持宽高比缩放后的尺寸）
        scaled_width = min(pixmap_size.width(), label_size.width())
        scaled_height = min(pixmap_size.height(), label_size.height())

        # 计算图像在QLabel中的绘制起点（居中补偿）
        x_offset = (label_size.width() - scaled_width) // 2
        y_offset = (label_size.height() - scaled_height) // 2

        # 转换为相对于图像内容的坐标
        click_x_in_content = pos.x() - x_offset
        click_y_in_content = pos.y() - y_offset

        # 计算缩放比例
        scale = min(
            pixmap_size.width() / img_width,
            pixmap_size.height() / img_height
        )

        # 转换为原始图像坐标
        original_x = int(click_x_in_content / scale)
        original_y = int(click_y_in_content / scale)

        # 边界检查
        original_x = max(0, min(original_x, img_width - 1))
        original_y = max(0, min(original_y, img_height - 1))

        return QPoint(original_x, original_y)

    def perform_segmentation(self):
        """使用SAM模型执行分割"""
        if not self.current_data or not self.current_data['click_points']:
            return

        try:
            self.status_label.setText("状态: 正在执行分割...")
            QApplication.processEvents()  # 更新UI

            # 准备输入点
            input_points = np.array(self.current_data['click_points'])
            input_labels = np.array(self.current_data['labels'])

            # 执行预测
            masks, scores, _ = self.predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=True,
            )

            # 选择得分最高的mask
            best_mask_idx = np.argmax(scores)
            self.current_data['mask'] = masks[best_mask_idx]

            self.status_label.setText("状态: 分割完成")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"分割失败: {str(e)}")
            self.status_label.setText("状态: 分割失败")

    def update_image_display(self):
        """更新图像显示"""
        if not self.current_data:
            return

        # 创建显示用的图像副本
        display_image = self.current_data['display_image'].copy()
        draw = Image.new("RGBA", display_image.size, (0, 0, 0, 0))

        # 如果有mask，绘制mask
        if self.current_data['mask'] is not None:
            # 创建mask图像
            mask_image = Image.fromarray((self.current_data['mask'] * 255).astype(np.uint8), mode="L")

            # 创建彩色遮罩（半透明绿色）
            color_mask = Image.new("RGBA", display_image.size, (0, 255, 0, 128))

            # 应用mask
            draw.paste(color_mask, (0, 0), mask_image)

        # 绘制点击点
        for i, (point, label) in enumerate(zip(self.current_data['click_points'], self.current_data['labels'])):
            x, y = point
            color = (255, 0, 0) if label == 1 else (0, 0, 255)  # 红色前景，蓝色背景
            size = 10

            # 绘制圆形标记
            marker = Image.new("RGBA", (size*2, size*2), (0, 0, 0, 0))
            marker_draw = ImageDraw.Draw(marker)
            marker_draw.ellipse([(0, 0), (size*2-1, size*2-1)], fill=color)

            # 将标记粘贴到图像上
            draw.paste(marker, (x-size, y-size), marker)

        # 合并原始图像和绘制内容
        result = Image.alpha_composite(
            display_image.convert("RGBA"),
            draw
        ).convert("RGB")

        # 转换为QPixmap并显示
        qimage = ImageQt.ImageQt(result)
        pixmap = QPixmap.fromImage(qimage)

        # 保持图像比例
        scaled_pixmap = pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        self.image_label.setPixmap(scaled_pixmap)

    def clear_markers(self):
        """清除所有标记和分割结果"""
        if not self.current_data:
            return

        # 保存当前状态到历史记录
        self.save_state_to_history()

        self.current_data['click_points'] = []
        self.current_data['labels'] = []
        self.current_data['mask'] = None
        self.current_data['display_image'] = self.current_data['original_image'].copy()

        self.update_image_display()
        self.status_label.setText("状态: 已清除所有标记")

    def save_current_result(self):
        """保存当前图片的分割结果"""
        if not self.current_data or self.current_data['mask'] is None:
            QMessageBox.warning(self, "警告", "没有可保存的分割结果")
            return

        # 获取原始文件名作为默认文件名
        original_path = self.image_paths[self.current_image_index]
        original_name = os.path.splitext(os.path.basename(original_path))[0]
        default_filename = f"{original_name}_segmented.png"

        options = QFileDialog.Option.DontUseNativeDialog
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self, "保存结果", default_filename,
            "PNG图像 (*.png);;JPEG图像 (*.jpg *.jpeg);;BMP图像 (*.bmp);;TIFF图像 (*.tif *.tiff);;所有文件 (*)",
            options=options
        )

        if file_path:
            try:
                # 确保目录存在
                os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

                # 根据选择的过滤器确定文件格式和扩展名
                format_map = {
                    "PNG图像 (*.png)": ("PNG", ".png"),
                    "JPEG图像 (*.jpg *.jpeg)": ("JPEG", ".jpg"),
                    "BMP图像 (*.bmp)": ("BMP", ".bmp"),
                    "TIFF图像 (*.tif *.tiff)": ("TIFF", ".tif")
                }

                if selected_filter in format_map:
                    format, default_ext = format_map[selected_filter]
                    # 确保文件有正确的扩展名
                    if not any(file_path.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']):
                        file_path += default_ext
                else:
                    # 根据现有扩展名确定格式，如果没有扩展名则使用PNG
                    ext = os.path.splitext(file_path)[1].lower()
                    if ext == '.png':
                        format = "PNG"
                    elif ext in ['.jpg', '.jpeg']:
                        format = "JPEG"
                    elif ext == '.bmp':
                        format = "BMP"
                    elif ext in ['.tif', '.tiff']:
                        format = "TIFF"
                    else:
                        # 默认使用PNG格式
                        format = "PNG"
                        file_path += '.png'

                # 创建结果图像
                if self.current_data['original_image'].mode == 'RGBA':
                    result_image = self.current_data['original_image'].copy()
                else:
                    result_image = self.current_data['original_image'].convert('RGBA')

                # 应用mask
                mask_array = (self.current_data['mask'] * 255).astype(np.uint8)
                mask_image = Image.fromarray(mask_array, mode="L")

                # 创建半透明绿色遮罩
                color_mask = Image.new("RGBA", result_image.size, (0, 255, 0, 128))

                # 合并图像和遮罩
                result_image = Image.alpha_composite(result_image, color_mask)

                # 对于不支持透明度的格式，转换为RGB
                if format in ["JPEG", "BMP"]:
                    result_image = result_image.convert('RGB')

                # 保存图像
                result_image.save(file_path, format=format)
                self.status_label.setText(f"状态: 结果已保存到 {os.path.basename(file_path)}")
                QMessageBox.information(self, "成功", f"图片已成功保存到:\n{file_path}")

            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存结果失败: {str(e)}")
                self.status_label.setText("状态: 保存失败")

    def save_all_results(self):
        """保存所有图片的分割结果"""
        if not self.image_paths:
            QMessageBox.warning(self, "警告", "没有可保存的分割结果")
            return

        # 首先让用户选择保存格式
        format_options = {
            "PNG": ("PNG", ".png"),
            "JPEG": ("JPEG", ".jpg"),
            "BMP": ("BMP", ".bmp"),
            "TIFF": ("TIFF", ".tif")
        }

        format_dialog = QMessageBox(self)
        format_dialog.setWindowTitle("选择保存格式")
        format_dialog.setText("请选择批量保存的图片格式:")
        format_dialog.setIcon(QMessageBox.Icon.Question)

        # 添加格式选择按钮
        png_button = format_dialog.addButton("PNG", QMessageBox.ButtonRole.ActionRole)
        jpeg_button = format_dialog.addButton("JPEG", QMessageBox.ButtonRole.ActionRole)
        bmp_button = format_dialog.addButton("BMP", QMessageBox.ButtonRole.ActionRole)
        tiff_button = format_dialog.addButton("TIFF", QMessageBox.ButtonRole.ActionRole)
        cancel_button = format_dialog.addButton("取消", QMessageBox.ButtonRole.RejectRole)

        format_dialog.exec()

        clicked_button = format_dialog.clickedButton()
        if clicked_button == cancel_button:
            return

        # 确定选择的格式
        selected_format = None
        if clicked_button == png_button:
            selected_format = "PNG"
        elif clicked_button == jpeg_button:
            selected_format = "JPEG"
        elif clicked_button == bmp_button:
            selected_format = "BMP"
        elif clicked_button == tiff_button:
            selected_format = "TIFF"
        else:
            return

        # 然后选择保存目录
        options = QFileDialog.Option.DontUseNativeDialog | QFileDialog.Option.ShowDirsOnly
        save_dir = QFileDialog.getExistingDirectory(
            self, "选择保存目录", "", options=options
        )

        if not save_dir:
            return

        try:
            saved_count = 0
            format_info = format_options[selected_format]
            file_format, file_extension = format_info

            for i, file_path in enumerate(self.image_paths):
                data = self.image_data[file_path]
                if data['mask'] is not None:
                    # 创建结果图像
                    if data['original_image'].mode == 'RGBA':
                        result_image = data['original_image'].copy()
                    else:
                        result_image = data['original_image'].convert('RGBA')

                    # 应用mask
                    mask_array = (data['mask'] * 255).astype(np.uint8)
                    mask_image = Image.fromarray(mask_array, mode="L")

                    # 创建半透明绿色遮罩
                    color_mask = Image.new("RGBA", result_image.size, (0, 255, 0, 128))

                    # 合并图像和遮罩
                    result_image = Image.alpha_composite(result_image, color_mask)

                    # 对于不支持透明度的格式，转换为RGB
                    if file_format in ["JPEG", "BMP"]:
                        result_image = result_image.convert('RGB')

                    # 构建保存路径
                    base_name = os.path.splitext(os.path.basename(file_path))[0]
                    save_path = os.path.join(save_dir, f"{base_name}_segmented{file_extension}")

                    # 保存图像
                    result_image.save(save_path, format=file_format)
                    saved_count += 1

                    # 更新状态（每处理10张图片更新一次）
                    if saved_count % 10 == 0:
                        self.status_label.setText(f"状态: 正在保存... 已处理 {saved_count} 张图片")
                        QApplication.processEvents()

            self.status_label.setText(f"状态: 已保存 {saved_count}/{len(self.image_paths)} 张图片的结果到 {save_dir}")
            QMessageBox.information(self, "完成", f"成功保存 {saved_count} 张图片的分割结果\n格式: {selected_format}\n保存目录: {save_dir}")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"批量保存结果失败: {str(e)}")
            self.status_label.setText("状态: 批量保存失败")

    def set_foreground_mode(self):
        """设置前景标记模式"""
        self.current_mode = "foreground"
        self.foreground_button.setChecked(True)
        self.background_button.setChecked(False)
        self.status_label.setText("状态: 当前模式 - 前景标记")

    def set_background_mode(self):
        """设置背景标记模式"""
        self.current_mode = "background"
        self.foreground_button.setChecked(False)
        self.background_button.setChecked(True)
        self.status_label.setText("状态: 当前模式 - 背景标记")

    def resizeEvent(self, event):
        """处理窗口大小变化事件"""
        super().resizeEvent(event)
        self.update_image_display()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MedicalImageSegmentationApp()
    window.show()
    sys.exit(app.exec())
