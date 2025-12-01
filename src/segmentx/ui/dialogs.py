from typing import Optional

from PySide6.QtWidgets import QFileDialog, QMessageBox, QWidget

from ..config import FILE_DIALOG_FILTER


def show_error(parent: QWidget, title: str, message: str) -> None:
    QMessageBox.critical(parent, title, message)


def show_warning(parent: QWidget, title: str, message: str) -> None:
    QMessageBox.warning(parent, title, message)


def show_info(parent: QWidget, title: str, message: str) -> None:
    QMessageBox.information(parent, title, message)


def choose_images(parent: QWidget, single: bool) -> list[str]:
    options = QFileDialog.Option.ReadOnly
    if single:
        path, _ = QFileDialog.getOpenFileName(parent, "选择医学影像", "", FILE_DIALOG_FILTER, options=options)
        return [path] if path else []
    paths, _ = QFileDialog.getOpenFileNames(parent, "选择多个医学影像", "", FILE_DIALOG_FILTER, options=options)
    return paths


def choose_save_path(parent: QWidget, default_name: str = "") -> tuple[str, str]:
    options = QFileDialog.Option.DontUseNativeDialog
    return QFileDialog.getSaveFileName(
        parent,
        "保存结果",
        default_name,
        "PNG图像 (*.png);;JPEG图像 (*.jpg *.jpeg);;BMP图像 (*.bmp);;TIFF图像 (*.tif *.tiff);;所有文件 (*)",
        options=options,
    )


def choose_directory(parent: QWidget) -> str:
    options = QFileDialog.Option.DontUseNativeDialog | QFileDialog.Option.ShowDirsOnly
    return QFileDialog.getExistingDirectory(parent, "选择保存目录", "", options=options)


def choose_format(parent: QWidget) -> Optional[str]:
    dialog = QMessageBox(parent)
    dialog.setWindowTitle("选择保存格式")
    dialog.setText("请选择批量保存的图片格式:")
    dialog.setIcon(QMessageBox.Icon.Question)

    buttons = {
        dialog.addButton("PNG", QMessageBox.ButtonRole.ActionRole): "PNG",
        dialog.addButton("JPEG", QMessageBox.ButtonRole.ActionRole): "JPEG",
        dialog.addButton("BMP", QMessageBox.ButtonRole.ActionRole): "BMP",
        dialog.addButton("TIFF", QMessageBox.ButtonRole.ActionRole): "TIFF",
        dialog.addButton("取消", QMessageBox.ButtonRole.RejectRole): None,
    }

    dialog.exec()
    return buttons.get(dialog.clickedButton())


__all__ = [
    "show_error",
    "show_warning",
    "show_info",
    "choose_images",
    "choose_save_path",
    "choose_directory",
    "choose_format",
]
