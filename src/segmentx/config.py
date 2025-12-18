from pathlib import Path

from .utils.paths import ensure_dir, get_project_root

# Project paths
PROJECT_ROOT = get_project_root()
RESOURCES_DIR = PROJECT_ROOT / "resources"
MODELS_DIR = ensure_dir(PROJECT_ROOT / "models")
MODEL_STORE_DIR = ensure_dir(PROJECT_ROOT / "model_store")
IMPORT_PREFERENCES_FILE = MODEL_STORE_DIR / "import_preferences.json"
MODELS_DOWNLOADS_DIR = ensure_dir(MODEL_STORE_DIR / "downloads")
MODELS_CACHE_DIR = ensure_dir(MODEL_STORE_DIR / "cache")
MODELS_SOURCES_FILE = MODEL_STORE_DIR / "sources.yaml"
NNUNET_SETTINGS_FILE = MODEL_STORE_DIR / "nnunet_settings.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Model configuration
MODEL_CHECKPOINT = MODELS_DIR / "sam_vit_h_4b8939.pth"
MODEL_TYPE = "vit_h"

# nnU-Net binaries
NNUNET_PREDICT_CMD = "nnUNetv2_predict"
NNUNET_TRAIN_CMD = "nnUNetv2_train"

# Image handling
SUPPORTED_EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
FILE_DIALOG_FILTER = (
    "图像文件 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;所有文件 (*)"
)
MAX_HISTORY = 20
