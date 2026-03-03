from __future__ import annotations

from collections import OrderedDict
from typing import Dict

from .provider import SliceProvider


class SliceCache:
    """Windowed slice cache to cap memory while scrolling through a volume."""

    def __init__(self, provider: SliceProvider, kmax: int = 64) -> None:
        self.provider = provider
        self.kmax = max(1, kmax)
        self.cache: OrderedDict[int, object] = OrderedDict()

    def get(self, index: int):
        if index in self.cache:
            img = self.cache.pop(index)
            self.cache[index] = img
            return img
        img = self.provider.get_slice(index)
        self.cache[index] = img
        self._enforce_window(index)
        return img

    def _enforce_window(self, center: int) -> None:
        radius = max(0, self.kmax // 2)
        min_idx = max(0, center - radius)
        max_idx = min(self.provider.depth - 1, center + radius)
        allowed = set(range(min_idx, max_idx + 1))
        evict = [idx for idx in self.cache.keys() if idx not in allowed]
        for idx in evict:
            self.cache.pop(idx, None)

    def stats(self) -> Dict[str, int]:
        return {"cached": len(self.cache), "kmax": self.kmax}
