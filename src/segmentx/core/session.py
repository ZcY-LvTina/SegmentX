from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from PIL import Image


@dataclass
class MaskLayers:
    """Mask layers to separate final result and auxiliary tool overlays."""

    result: Optional[np.ndarray] = None  # final segmentation mask
    hint: Optional[np.ndarray] = None  # tool/hint layer (e.g., green overlay)
    show_hint: bool = True

    def clone(self) -> "MaskLayers":
        return MaskLayers(
            result=self.result.copy() if self.result is not None else None,
            hint=self.hint.copy() if self.hint is not None else None,
            show_hint=self.show_hint,
        )


@dataclass
class ImageState:
    path: str
    original_image: Image.Image
    display_image: Image.Image
    click_points: List[List[int]]
    labels: List[int]
    auto_mask: Optional[np.ndarray] = None  # 模型分割掩膜
    manual_mask: Optional[np.ndarray] = None  # 手动精修掩膜，优先作为最终结果
    manual_edit_enabled: bool = False  # 是否处于手动精修模式
    current_polygon_points: List[List[int]] = field(default_factory=list)  # 当前绘制的多边形（图像坐标）
    manual_brush_mode: str = "add"  # add / erase
    mask_layers: MaskLayers = field(default_factory=MaskLayers)

    @property
    def mask(self) -> Optional[np.ndarray]:
        # 优先返回手动精修后的掩膜；无手动掩膜时回退到模型结果
        if self.manual_mask is not None:
            return self.manual_mask
        if self.auto_mask is not None:
            return self.auto_mask
        return self.mask_layers.result

    @mask.setter
    def mask(self, value: Optional[np.ndarray]) -> None:
        # setter 主要给模型分割结果使用，作为 auto_mask 存储
        self.auto_mask = value
        self.mask_layers.result = value

    def clone(self) -> "ImageState":
        return ImageState(
            path=self.path,
            original_image=self.original_image.copy(),
            display_image=self.display_image.copy(),
            click_points=[pt.copy() for pt in self.click_points],
            labels=self.labels.copy(),
            auto_mask=self.auto_mask.copy() if self.auto_mask is not None else None,
            manual_mask=self.manual_mask.copy() if self.manual_mask is not None else None,
            manual_edit_enabled=self.manual_edit_enabled,
            current_polygon_points=[pt.copy() for pt in self.current_polygon_points],
            manual_brush_mode=self.manual_brush_mode,
            mask_layers=self.mask_layers.clone(),
        )


class Session:
    """Manage image state with undo/redo stacks."""

    def __init__(self, state: ImageState, max_history: int = 20) -> None:
        self.state = state
        self.max_history = max_history
        self.undo_stack: List[ImageState] = []
        self.redo_stack: List[ImageState] = []

    def save(self) -> None:
        """Save current state to undo stack."""
        if len(self.undo_stack) >= self.max_history:
            self.undo_stack.pop(0)
        self.undo_stack.append(self.state.clone())
        self.redo_stack.clear()

    def can_undo(self) -> bool:
        return len(self.undo_stack) > 0

    def can_redo(self) -> bool:
        return len(self.redo_stack) > 0

    def undo(self) -> Optional[ImageState]:
        if not self.can_undo():
            return None
        self.redo_stack.append(self.state.clone())
        self.state = self.undo_stack.pop()
        return self.state

    def redo(self) -> Optional[ImageState]:
        if not self.can_redo():
            return None
        self.undo_stack.append(self.state.clone())
        self.state = self.redo_stack.pop()
        return self.state
