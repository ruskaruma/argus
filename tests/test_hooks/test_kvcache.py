from __future__ import annotations

import sys

from argus.core.tracer import Tracer
from argus.hooks.kvcache import KVCacheTracker, _compute_kv_cache_bytes


class FakeTensor:
    """Mock tensor with nelement() and element_size() for testing without torch."""

    def __init__(self, numel: int, elem_size: int = 2) -> None:
        self._numel = numel
        self._elem_size = elem_size

    def nelement(self) -> int:
        return self._numel

    def element_size(self) -> int:
        return self._elem_size


def _make_kv(num_layers: int = 2, numel: int = 1024, elem_size: int = 2):
    """Build fake past_key_values: tuple of (key, value) tuples per layer."""
    return tuple(
        (FakeTensor(numel, elem_size), FakeTensor(numel, elem_size)) for _ in range(num_layers)
    )


def test_kvcache_tracker_creates_event():
    t = Tracer()
    tracker = KVCacheTracker(t)
    tracker.record(_make_kv(), token_index=0)
    assert len(t._events) == 1


def test_kvcache_event_category():
    t = Tracer()
    tracker = KVCacheTracker(t)
    tracker.record(_make_kv(), token_index=0)
    assert t._events[0].category == "memory"


def test_kvcache_event_name():
    t = Tracer()
    tracker = KVCacheTracker(t)
    tracker.record(_make_kv(), token_index=0)
    assert t._events[0].name == "kv_cache_grow"


def test_kvcache_event_scope():
    t = Tracer()
    tracker = KVCacheTracker(t)
    tracker.record(_make_kv(), token_index=5)
    assert t._events[0].scope == "decode.token.5"


def test_kvcache_event_metadata():
    t = Tracer()
    tracker = KVCacheTracker(t)
    tracker.record(_make_kv(num_layers=3, numel=512, elem_size=4), token_index=2)
    meta = t._events[0].metadata
    assert meta["cache_size_bytes"] == 3 * 2 * 512 * 4  # 3 layers, 2 tensors each
    assert meta["num_layers"] == 3
    assert meta["token_index"] == 2


def test_kvcache_multiple_tokens():
    t = Tracer()
    tracker = KVCacheTracker(t)
    for i in range(10):
        tracker.record(_make_kv(), token_index=i)
    assert len(t._events) == 10
    for i, event in enumerate(t._events):
        assert event.token_index == i


def test_kvcache_monotonic_growth():
    t = Tracer()
    tracker = KVCacheTracker(t)
    for i in range(5):
        # each step adds more elements to simulate growing cache
        kv = _make_kv(num_layers=2, numel=1024 * (i + 1))
        tracker.record(kv, token_index=i)
    sizes = [e.metadata["cache_size_bytes"] for e in t._events]
    for i in range(1, len(sizes)):
        assert sizes[i] >= sizes[i - 1]


def test_compute_kv_cache_bytes():
    kv = _make_kv(num_layers=4, numel=256, elem_size=2)
    total, num_layers = _compute_kv_cache_bytes(kv)
    assert total == 4 * 2 * 256 * 2
    assert num_layers == 4


def test_no_torch_import_on_module_load():
    assert "argus.hooks.kvcache" in sys.modules
    assert "torch" not in sys.modules
