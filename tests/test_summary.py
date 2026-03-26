from __future__ import annotations

from argus.core.events import TraceEvent
from argus.summary import summarize


def _make_event(**overrides) -> TraceEvent:
    defaults = {
        "event_id": "0",
        "name": "test",
        "start_ns": 1_000_000,
        "end_ns": 2_000_000,
        "category": "compute",
        "scope": "test",
    }
    defaults.update(overrides)
    return TraceEvent(**defaults)


def test_empty_events():
    result = summarize([])
    assert result.total_events == 0
    assert result.total_duration_ms == 0.0
    assert result.categories == {}
    assert result.token_count == 0
    assert result.token_latency is None
    assert result.tokens_per_second is None


def test_single_event():
    result = summarize([_make_event()])
    assert result.total_events == 1
    assert result.total_duration_ms == 1.0


def test_total_duration_spans_all_events():
    events = [
        _make_event(event_id="0", start_ns=0, end_ns=1_000_000),
        _make_event(event_id="1", start_ns=5_000_000, end_ns=10_000_000),
    ]
    result = summarize(events)
    assert result.total_duration_ms == 10.0


def test_category_counts():
    events = [
        _make_event(event_id="0", category="compute"),
        _make_event(event_id="1", category="compute"),
        _make_event(event_id="2", category="memory", start_ns=0, end_ns=0),
    ]
    result = summarize(events)
    assert result.categories["compute"].count == 2
    assert result.categories["memory"].count == 1


def test_category_total_duration():
    events = [
        _make_event(event_id="0", category="compute", start_ns=0, end_ns=3_000_000),
        _make_event(event_id="1", category="compute", start_ns=0, end_ns=2_000_000),
    ]
    result = summarize(events)
    assert result.categories["compute"].total_duration_ms == 5.0


def test_token_count():
    events = [
        _make_event(event_id="0", category="token", token_index=0),
        _make_event(event_id="1", category="token", token_index=1),
        _make_event(event_id="2", category="token", token_index=2),
    ]
    result = summarize(events)
    assert result.token_count == 3


def test_token_latency_stats():
    events = [
        _make_event(event_id=str(i), category="token", start_ns=0, end_ns=(i + 1) * 1_000_000)
        for i in range(10)
    ]
    result = summarize(events)
    assert result.token_latency is not None
    assert result.token_latency.count == 10
    assert result.token_latency.min_ms == 1.0
    assert result.token_latency.max_ms == 10.0


def test_token_latency_single_token():
    events = [_make_event(category="token", start_ns=0, end_ns=5_000_000)]
    result = summarize(events)
    assert result.token_latency is not None
    assert result.token_latency.count == 1
    assert result.token_latency.p50_ms == 5.0
    assert result.token_latency.p99_ms == 5.0


def test_no_token_latency_without_tokens():
    events = [_make_event(category="compute")]
    result = summarize(events)
    assert result.token_latency is None


def test_tokens_per_second():
    decode = _make_event(
        event_id="0",
        category="phase",
        name="decode",
        start_ns=0,
        end_ns=1_000_000_000,
    )
    t0 = _make_event(
        event_id="1",
        category="token",
        token_index=0,
        start_ns=0,
        end_ns=100_000_000,
    )
    t1 = _make_event(
        event_id="2",
        category="token",
        token_index=1,
        start_ns=100_000_000,
        end_ns=200_000_000,
    )
    result = summarize([decode, t0, t1])
    assert result.tokens_per_second == 2.0


def test_tokens_per_second_none_without_decode():
    events = [
        _make_event(event_id="0", category="token", token_index=0),
    ]
    result = summarize(events)
    assert result.tokens_per_second is None


def test_prefill_duration():
    events = [
        _make_event(category="phase", name="prefill", start_ns=0, end_ns=5_000_000),
    ]
    result = summarize(events)
    assert result.prefill_duration_ms == 5.0


def test_decode_duration():
    events = [
        _make_event(category="phase", name="decode", start_ns=0, end_ns=10_000_000),
    ]
    result = summarize(events)
    assert result.decode_duration_ms == 10.0


def test_no_phase_durations():
    events = [_make_event(category="compute")]
    result = summarize(events)
    assert result.prefill_duration_ms is None
    assert result.decode_duration_ms is None


def test_peak_memory_bytes():
    events = [
        _make_event(
            event_id="0",
            category="memory",
            name="kv_cache_grow",
            start_ns=0,
            end_ns=0,
            metadata={"cache_size_bytes": 1000},
        ),
        _make_event(
            event_id="1",
            category="memory",
            name="kv_cache_grow",
            start_ns=0,
            end_ns=0,
            metadata={"cache_size_bytes": 5000},
        ),
        _make_event(
            event_id="2",
            category="memory",
            name="kv_cache_grow",
            start_ns=0,
            end_ns=0,
            metadata={"cache_size_bytes": 3000},
        ),
    ]
    result = summarize(events)
    assert result.peak_memory_bytes == 5000


def test_peak_memory_none_without_memory_events():
    events = [_make_event(category="compute")]
    result = summarize(events)
    assert result.peak_memory_bytes is None


def test_summary_dataclass_is_frozen():
    result = summarize([])
    try:
        result.total_events = 5  # type: ignore[misc]
        raise AssertionError("Expected FrozenInstanceError")
    except AttributeError:
        pass


def test_latency_stats_is_frozen():
    events = [_make_event(category="token")]
    result = summarize(events)
    assert result.token_latency is not None
    try:
        result.token_latency.count = 999  # type: ignore[misc]
        raise AssertionError("Expected FrozenInstanceError")
    except AttributeError:
        pass


def test_category_stats_is_frozen():
    events = [_make_event(category="compute")]
    result = summarize(events)
    try:
        result.categories["compute"].count = 999  # type: ignore[misc]
        raise AssertionError("Expected FrozenInstanceError")
    except AttributeError:
        pass


def test_full_trace_summary():
    prefill = _make_event(
        event_id="0",
        category="phase",
        name="prefill",
        start_ns=0,
        end_ns=10_000_000,
    )
    fwd = _make_event(
        event_id="1",
        category="compute",
        name="forward_pass",
        start_ns=0,
        end_ns=8_000_000,
    )
    decode = _make_event(
        event_id="2",
        category="phase",
        name="decode",
        start_ns=10_000_000,
        end_ns=110_000_000,
    )
    t0 = _make_event(
        event_id="3",
        category="token",
        token_index=0,
        start_ns=10_000_000,
        end_ns=20_000_000,
    )
    t1 = _make_event(
        event_id="4",
        category="token",
        token_index=1,
        start_ns=20_000_000,
        end_ns=30_000_000,
    )
    mem = _make_event(
        event_id="5",
        category="memory",
        name="kv_cache_grow",
        start_ns=20_000_000,
        end_ns=20_000_000,
        metadata={"cache_size_bytes": 4096},
    )
    result = summarize([prefill, fwd, decode, t0, t1, mem])
    assert result.total_events == 6
    assert result.total_duration_ms == 110.0
    assert result.token_count == 2
    assert result.prefill_duration_ms == 10.0
    assert result.decode_duration_ms == 100.0
    assert result.peak_memory_bytes == 4096
    assert result.tokens_per_second == 20.0
    assert result.token_latency is not None
    assert result.token_latency.count == 2


def test_summarize_available_from_package():
    from argus import TraceSummary
    from argus import summarize as s

    assert callable(s)
    assert TraceSummary is not None
