from pathlib import Path

from .utils.paths import get_project_root

# Project paths
PROJECT_ROOT = get_project_root()
RESOURCES_DIR = PROJECT_ROOT / "resources"
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Model configuration
MODEL_CHECKPOINT = MODELS_DIR / "sam_vit_h_4b8939.pth"
MODEL_TYPE = "vit_h"

# Image handling
SUPPORTED_EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
FILE_DIALOG_FILTER = (
    "图像文件 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;所有文件 (*)"
)
MAX_HISTORY = 20
