from typing import Optional

from PySide6.QtWidgets import QFileDialog, QMessageBox, QWidget

from ..config import FILE_DIALOG_FILTER


def show_error(parent: QWidget, title: str, message: str) -> None:
    QMessageBox.critical(parent, title, message)


def show_warning(parent: QWidget, title: str, message: str) -> None:
    QMessageBox.warning(parent, title, message)


def show_info(parent: QWidget, title: str, message: str) -> None:
    QMessageBox.information(parent, title, message)


def choose_images(parent: QWidget) -> list[str]:
    """统一的图片选择入口，支持单选和多选。"""
    options = QFileDialog.Option.ReadOnly
    paths, _ = QFileDialog.getOpenFileNames(parent, "选择医学影像", "", FILE_DIALOG_FILTER, options=options)
    return paths


def choose_import_paths(parent: QWidget) -> list[str]:
    """选择图片/体数据文件，并可追加目录用于 3D 序列或 DICOM。"""
    options = QFileDialog.Option.ReadOnly | QFileDialog.Option.DontUseNativeDialog
    filter_str = (
        "图像/体数据 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.nii *.nii.gz *.nrrd *.nhdr *.mha *.mhd);;所有文件 (*)"
    )
    files, _ = QFileDialog.getOpenFileNames(parent, "选择文件", "", filter_str, options=options)
    if not files:
        return []

    paths = list(files)
    add_dir = QMessageBox.question(
        parent,
        "添加文件夹？",
        "是否额外选择包含 DICOM/PNG 序列的文件夹？",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
    )
    if add_dir == QMessageBox.StandardButton.Yes:
        folder = QFileDialog.getExistingDirectory(
            parent, "选择文件夹 (可选，DICOM 或 PNG 序列)", "", QFileDialog.Option.ShowDirsOnly
        )
        if folder:
            paths.append(folder)
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


def choose_zip_file(parent: QWidget, title: str = "选择模型 zip") -> str:
    options = QFileDialog.Option.ReadOnly
    path, _ = QFileDialog.getOpenFileName(parent, title, "", "Zip 文件 (*.zip);;所有文件 (*)", options=options)
    return path


def choose_folder(parent: QWidget, title: str = "选择文件夹") -> str:
    options = QFileDialog.Option.ShowDirsOnly | QFileDialog.Option.DontResolveSymlinks
    return QFileDialog.getExistingDirectory(parent, title, "")


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
    "choose_zip_file",
    "choose_folder",
    "choose_import_paths",
]
