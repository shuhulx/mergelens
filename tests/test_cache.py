"""Tests for metric caching."""

import torch

from mergelens.utils.cache import MetricCache, _tensor_hash


def test_tensor_hash_deterministic():
    t = torch.randn(32, 32)
    assert _tensor_hash(t) == _tensor_hash(t)


def test_tensor_hash_different():
    a = torch.randn(32, 32)
    b = torch.randn(32, 32)
    assert _tensor_hash(a) != _tensor_hash(b)


def test_cache_set_get(tmp_path):
    cache = MetricCache(cache_dir=str(tmp_path / "cache"))
    cache.set("test_key", {"value": 42})
    assert cache.get("test_key") == {"value": 42}


def test_cache_miss(tmp_path):
    cache = MetricCache(cache_dir=str(tmp_path / "cache"))
    assert cache.get("nonexistent") is None


def test_cache_disabled():
    cache = MetricCache(enabled=False)
    cache.set("key", "value")
    assert cache.get("key") is None


def test_make_key():
    cache = MetricCache(enabled=False)
    t1 = torch.randn(16, 16)
    t2 = torch.randn(16, 16)
    key = cache.make_key("cosine", t1, t2)
    assert key.startswith("v2:cosine:")
