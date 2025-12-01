from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry

from ..config import MODEL_CHECKPOINT, MODEL_TYPE


class SamEngine:
    """Thin wrapper around SAM predictor without Qt dependency."""

    def __init__(
        self,
        checkpoint: Path = MODEL_CHECKPOINT,
        model_type: str = MODEL_TYPE,
    ) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint_path}")

        sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
        sam.to(device=self.device)
        self.predictor = SamPredictor(sam)

    def set_image(self, image: np.ndarray) -> None:
        """Set current image for predictor."""
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        self.predictor.set_image(image)

    def segment(
        self,
        points: np.ndarray,
        labels: np.ndarray,
        multimask_output: bool = True,
    ) -> Optional[np.ndarray]:
        if points.size == 0:
            return None

        masks, scores, _ = self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=multimask_output,
        )
        best_mask_idx = int(np.argmax(scores))
        return masks[best_mask_idx]
