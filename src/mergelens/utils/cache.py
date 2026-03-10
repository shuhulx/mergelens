"""Caching for expensive metric computations."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

try:
    import diskcache

    HAS_DISKCACHE = True
except ImportError:
    HAS_DISKCACHE = False

_CACHE_VERSION = "v2"


def _tensor_hash(tensor) -> str:
    """Create a stable hash for a tensor based on shape and sample values."""
    import torch

    shape_str = str(tuple(tensor.shape))
    flat = tensor.flatten()
    n = min(1000, len(flat)) if len(flat) >= 1_000_000 else len(flat)
    indices = torch.linspace(0, len(flat) - 1, n).long()
    sample = flat[indices].float().cpu().numpy().tobytes()
    return hashlib.sha256(shape_str.encode() + sample).hexdigest()[:16]


class MetricCache:
    """Cache for metric results keyed by tensor hashes."""

    def __init__(self, cache_dir: str | None = None, enabled: bool = True):
        self.enabled = enabled and HAS_DISKCACHE
        self._cache = None
        if self.enabled:
            cache_path = cache_dir or str(Path.home() / ".cache" / "mergelens")
            self._cache = diskcache.Cache(cache_path, size_limit=2 * 2**30)  # 2GB

    def get(self, key: str) -> Any | None:
        """Get a cached value."""
        if not self.enabled or self._cache is None:
            return None
        return self._cache.get(key)

    def set(self, key: str, value: Any, expire: int | None = None) -> None:
        """Set a cached value with optional expiration in seconds."""
        if not self.enabled or self._cache is None:
            return
        self._cache.set(key, value, expire=expire)

    def make_key(self, metric_name: str, *tensors) -> str:
        """Create a cache key from metric name and tensor hashes."""
        parts = [_CACHE_VERSION, metric_name]
        for t in tensors:
            parts.append(_tensor_hash(t))
        return ":".join(parts)

    def clear(self) -> None:
        """Clear all cached values."""
        if self._cache is not None:
            self._cache.clear()

    def close(self) -> None:
        """Close the cache."""
        if self._cache is not None:
            self._cache.close()
