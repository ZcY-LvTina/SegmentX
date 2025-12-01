from pathlib import Path


def get_project_root() -> Path:
    """Return repository root assuming src/segmentx/... layout."""
    return Path(__file__).resolve().parents[3]


def get_resource_dir() -> Path:
    return get_project_root() / "resources"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


__all__ = ["get_project_root", "get_resource_dir", "ensure_dir"]
