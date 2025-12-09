from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry

from ..config import MODEL_CHECKPOINT, MODEL_TYPE


@dataclass
class ImageEmbedding:
    embedding: torch.Tensor
    original_size: tuple[int, int]
    input_size: tuple[int, int]
    image_format: Optional[str] = None


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
        # Cache per-image embeddings to avoid recomputing encoder output when切换多图
        self.embedding_cache: Dict[str, ImageEmbedding] = {}

    def clear_cache(self) -> None:
        """Clear cached embeddings (e.g., when loading a new image set)."""
        self.embedding_cache.clear()

    def set_image(self, image: np.ndarray, image_id: Optional[str] = None) -> bool:
        """Set current image for predictor, reusing cached embedding when available.

        Returns True when loaded from cache, False when a new embedding was computed.
        """
        if image_id and image_id in self.embedding_cache:
            self._set_image_from_cache(self.embedding_cache[image_id])
            return True

        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        self.predictor.set_image(image)
        if image_id:
            self._cache_current_embedding(image_id)
        return False

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

    def _cache_current_embedding(self, image_id: str) -> None:
        embedding = self._get_embedding_from_predictor()
        if embedding is None:
            return
        # TODO: add LRU cap if embedding数量过多以控制显存/内存占用
        self.embedding_cache[image_id] = ImageEmbedding(
            embedding=embedding.detach() if hasattr(embedding, "detach") else embedding,
            original_size=tuple(self.predictor.original_size),
            input_size=tuple(self.predictor.input_size),
            image_format=getattr(self.predictor, "image_format", None),
        )

    def _set_image_from_cache(self, cached: ImageEmbedding) -> None:
        """Reuse already-computed encoder output for faster switching."""
        self.predictor.reset_image()
        if cached.image_format is not None and hasattr(self.predictor, "image_format"):
            self.predictor.image_format = cached.image_format
        self.predictor.original_size = cached.original_size
        self.predictor.input_size = cached.input_size
        self.predictor.features = cached.embedding
        self.predictor.is_image_set = True

    def _get_embedding_from_predictor(self) -> Optional[torch.Tensor]:
        # Prefer public getter if available; fallback to predictor.features for compatibility.
        if hasattr(self.predictor, "get_image_embedding"):
            return self.predictor.get_image_embedding()
        return getattr(self.predictor, "features", None)
