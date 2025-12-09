import numpy as np

from .sam_engine import SamEngine
from .session import Session


def run_segmentation(session: Session, engine: SamEngine) -> np.ndarray | None:
    """Run SAM on current session points and update mask."""
    if not session.state.click_points:
        return None
    points = np.array(session.state.click_points)
    labels = np.array(session.state.labels)
    mask = engine.segment(points, labels)
    if mask is None:
        return None
    mask = mask.astype(bool)
    # mask setter会同步 auto_mask，保持模型结果与 hint 对齐
    session.state.mask = mask
    session.state.auto_mask = mask
    # Keep interactive hint layer aligned with latest result; can be hidden via UI toggle
    session.state.mask_layers.hint = mask
    return mask


__all__ = ["run_segmentation"]
