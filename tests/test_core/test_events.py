from __future__ import annotations

import dataclasses

from argus.core.events import VALID_CATEGORIES, TraceEvent


def _make_event(**overrides) -> TraceEvent:
    defaults = {
        "event_id": "0",
        "name": "test",
        "start_ns": 1000,
        "end_ns": 2000,
        "category": "compute",
        "scope": "test",
    }
    defaults.update(overrides)
    return TraceEvent(**defaults)


def test_basic_construction():
    e = _make_event()
    assert e.event_id == "0"
    assert e.name == "test"
    assert e.start_ns == 1000
    assert e.end_ns == 2000
    assert e.category == "compute"
    assert e.scope == "test"


def test_zero_duration():
    e = _make_event(start_ns=5000, end_ns=5000)
    assert e.duration_ns == 0
    assert e.duration_us == 0.0
    assert e.duration_ms == 0.0


def test_large_timestamps():
    start = 10**18
    end = 10**18 + 10**9
    e = _make_event(start_ns=start, end_ns=end)
    assert e.duration_ns == 10**9
    assert e.duration_us == 10**9 / 1_000
    assert e.duration_ms == 10**9 / 1_000_000


def test_empty_metadata():
    e = _make_event()
    assert e.metadata == {}
    assert isinstance(e.metadata, dict)


def test_none_optional_fields():
    e = _make_event(parent_id=None, token_index=None)
    assert e.parent_id is None
    assert e.token_index is None


def test_all_valid_categories():
    for cat in VALID_CATEGORIES:
        e = _make_event(category=cat)
        assert e.category == cat


def test_frozen_immutability():
    e = _make_event()
    with __import__("pytest").raises(dataclasses.FrozenInstanceError):
        e.name = "x"  # type: ignore[misc]


def test_slots_enforcement():
    e = _make_event()
    # frozen + slots: assigning unknown attr raises TypeError or AttributeError
    # depending on Python version (frozen __setattr__ fires before slots check)
    with __import__("pytest").raises((AttributeError, TypeError)):
        e.new_field = "x"  # type: ignore[attr-defined]


def test_duration_ns():
    e = _make_event(start_ns=100, end_ns=350)
    assert e.duration_ns == 250


def test_duration_us():
    e = _make_event(start_ns=0, end_ns=5000)
    assert e.duration_us == 5.0


def test_duration_ms():
    e = _make_event(start_ns=0, end_ns=2_000_000)
    assert e.duration_ms == 2.0


def test_metadata_is_shallow():
    meta = {"layer": "attn", "nested": {"a": 1}, "items": [1, 2]}
    e = _make_event(metadata=meta)
    assert e.metadata["nested"] == {"a": 1}
    assert e.metadata["items"] == [1, 2]


def test_scope_dot_notation():
    e = _make_event(scope="decode.token.42.layer.12")
    assert e.scope == "decode.token.42.layer.12"


def test_event_id_is_string():
    e = _make_event(event_id="42")
    assert isinstance(e.event_id, str)
