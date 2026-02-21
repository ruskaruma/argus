from __future__ import annotations

import json
from io import StringIO

from argus.core.events import TraceEvent
from argus.exporters.chrome import events_to_chrome, export_chrome_trace


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


def test_empty_events():
    assert events_to_chrome([]) == []
    sio = StringIO()
    export_chrome_trace([], sio)
    data = json.loads(sio.getvalue())
    assert data["traceEvents"] == []


def test_single_event_fields():
    chrome = events_to_chrome([_make_event()])[0]
    for key in ("ph", "name", "cat", "ts", "dur", "pid", "tid", "args"):
        assert key in chrome


def test_ph_x_for_duration_events():
    events = [_make_event(event_id=str(i)) for i in range(5)]
    for ce in events_to_chrome(events):
        assert ce["ph"] == "X"


def test_timestamp_in_microseconds():
    e = _make_event(start_ns=5_000_000)
    chrome = events_to_chrome([e])[0]
    assert chrome["ts"] == 5000.0


def test_duration_in_microseconds():
    e = _make_event(start_ns=1000, end_ns=3000)
    chrome = events_to_chrome([e])[0]
    assert chrome["dur"] == 2.0


def test_dur_non_negative():
    e = _make_event(start_ns=5000, end_ns=5000)
    chrome = events_to_chrome([e])[0]
    assert chrome["dur"] == 0.0


def test_category_to_tid_mapping():
    for cat, expected_tid in [("compute", 3), ("phase", 1), ("memory", 4)]:
        e = _make_event(category=cat)
        chrome = events_to_chrome([e])[0]
        assert chrome["tid"] == expected_tid


def test_unknown_category_tid():
    e = _make_event(category="unknown")
    chrome = events_to_chrome([e])[0]
    assert chrome["tid"] == 0


def test_args_contains_event_id():
    e = _make_event(event_id="abc")
    chrome = events_to_chrome([e])[0]
    assert chrome["args"]["event_id"] == "abc"


def test_args_contains_scope():
    e = _make_event(scope="decode.token.5")
    chrome = events_to_chrome([e])[0]
    assert chrome["args"]["scope"] == "decode.token.5"


def test_args_parent_id_when_present():
    e = _make_event(parent_id="5")
    chrome = events_to_chrome([e])[0]
    assert chrome["args"]["parent_id"] == "5"


def test_args_no_parent_id_when_none():
    e = _make_event(parent_id=None)
    chrome = events_to_chrome([e])[0]
    assert "parent_id" not in chrome["args"]


def test_args_token_index_when_present():
    e = _make_event(token_index=42)
    chrome = events_to_chrome([e])[0]
    assert chrome["args"]["token_index"] == 42


def test_args_no_token_index_when_none():
    e = _make_event(token_index=None)
    chrome = events_to_chrome([e])[0]
    assert "token_index" not in chrome["args"]


def test_metadata_flattened_into_args():
    e = _make_event(metadata={"layer": "attn"})
    chrome = events_to_chrome([e])[0]
    assert chrome["args"]["layer"] == "attn"


def test_args_is_flat():
    e = _make_event(metadata={"key": "val", "nested": {"a": 1}, "items": [1, 2]})
    chrome = events_to_chrome([e])[0]
    assert chrome["args"]["key"] == "val"
    assert "nested" not in chrome["args"]
    assert "items" not in chrome["args"]


def test_round_trip_event_count():
    events = [_make_event(event_id=str(i)) for i in range(50)]
    sio = StringIO()
    export_chrome_trace(events, sio)
    data = json.loads(sio.getvalue())
    assert len(data["traceEvents"]) == 50


def test_large_trace_10k_events():
    events = [
        _make_event(event_id=str(i), start_ns=i * 1000, end_ns=i * 1000 + 500)
        for i in range(10_000)
    ]
    sio = StringIO()
    export_chrome_trace(events, sio)
    data = json.loads(sio.getvalue())
    assert len(data["traceEvents"]) == 10_000


def test_export_to_file(tmp_path):
    events = [_make_event()]
    path = tmp_path / "test.json"
    export_chrome_trace(events, str(path))
    data = json.loads(path.read_text())
    assert len(data["traceEvents"]) == 1


def test_export_to_stringio():
    sio = StringIO()
    export_chrome_trace([_make_event()], sio)
    data = json.loads(sio.getvalue())
    assert "traceEvents" in data


def test_wrapper_json_structure():
    sio = StringIO()
    export_chrome_trace([_make_event()], sio)
    data = json.loads(sio.getvalue())
    assert set(data.keys()) == {"traceEvents", "displayTimeUnit", "metadata"}


def test_display_time_unit():
    sio = StringIO()
    export_chrome_trace([], sio)
    data = json.loads(sio.getvalue())
    assert data["displayTimeUnit"] == "ns"


def test_no_pretty_printing():
    sio = StringIO()
    export_chrome_trace([_make_event()], sio)
    raw = sio.getvalue()
    assert "\n  " not in raw


def test_metadata_key_collision():
    e = _make_event(event_id="real_id", metadata={"event_id": "fake_id", "scope": "fake"})
    chrome = events_to_chrome([e])[0]
    assert chrome["args"]["event_id"] == "real_id"
    assert chrome["args"]["scope"] == "test"


def test_zero_duration_memory_event_is_counter():
    e = _make_event(category="memory", start_ns=5000, end_ns=5000)
    chrome = events_to_chrome([e])[0]
    assert chrome["ph"] == "C"
    assert "dur" not in chrome


def test_negative_duration_clamped():
    e = _make_event(start_ns=5000, end_ns=4000)
    chrome = events_to_chrome([e])[0]
    assert chrome["dur"] == 0.0


def test_metadata_cannot_inject_fake_parent_id():
    e = _make_event(parent_id=None, metadata={"parent_id": "FAKE"})
    chrome = events_to_chrome([e])[0]
    assert "parent_id" not in chrome["args"]


def test_metadata_cannot_inject_fake_token_index():
    e = _make_event(token_index=None, metadata={"token_index": 999})
    chrome = events_to_chrome([e])[0]
    assert "token_index" not in chrome["args"]


def test_unknown_category_warns():
    import warnings as w

    e = _make_event(category="bogus")
    with w.catch_warnings(record=True) as caught:
        w.simplefilter("always")
        events_to_chrome([e])
    assert len(caught) == 1
    assert "bogus" in str(caught[0].message)
