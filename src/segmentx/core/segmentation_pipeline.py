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
    session.state.mask = mask
    return mask


__all__ = ["run_segmentation"]
