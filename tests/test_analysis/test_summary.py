from __future__ import annotations

import math

import pytest

from argus.analysis.summary import (
    SpanStats,
    TraceSummary,
    _fmt_duration,
    _percentile,
    format_summary,
    summarize,
)
from argus.core.events import TraceEvent
from argus.core.tracer import Tracer


def _make_event(**overrides: object) -> TraceEvent:
    defaults: dict[str, object] = {
        "event_id": "0",
        "name": "test",
        "start_ns": 1000,
        "end_ns": 2000,
        "category": "compute",
        "scope": "test",
    }
    defaults.update(overrides)
    return TraceEvent(**defaults)  # type: ignore[arg-type]


# ── Percentile helper ──


def test_percentile_empty():
    assert _percentile([], 50) == 0.0


def test_percentile_single():
    assert _percentile([100], 50) == 100.0
    assert _percentile([100], 99) == 100.0


def test_percentile_two_values():
    assert _percentile([100, 200], 0) == 100.0
    assert _percentile([100, 200], 100) == 200.0
    assert _percentile([100, 200], 50) == 150.0


def test_percentile_known_values():
    vals = list(range(1, 101))  # 1..100
    assert _percentile(vals, 50) == pytest.approx(50.5)
    assert _percentile(vals, 95) == pytest.approx(95.05)


# ── SpanStats properties ──


def test_span_stats_unit_conversions():
    stats = SpanStats(
        group="g",
        count=1,
        total_ns=5_000_000,
        mean_ns=5_000_000.0,
        median_ns=5_000_000.0,
        min_ns=5_000_000,
        max_ns=5_000_000,
        stddev_ns=0.0,
        p50_ns=5_000_000.0,
        p95_ns=5_000_000.0,
        p99_ns=5_000_000.0,
    )
    assert stats.mean_us == 5000.0
    assert stats.mean_ms == 5.0
    assert stats.median_us == 5000.0
    assert stats.p95_us == 5000.0
    assert stats.p99_us == 5000.0


def test_span_stats_frozen():
    stats = SpanStats(
        group="g",
        count=1,
        total_ns=0,
        mean_ns=0.0,
        median_ns=0.0,
        min_ns=0,
        max_ns=0,
        stddev_ns=0.0,
        p50_ns=0.0,
        p95_ns=0.0,
        p99_ns=0.0,
    )
    with pytest.raises(AttributeError):
        stats.count = 2  # type: ignore[misc]


# ── TraceSummary ──


def test_summary_get_existing():
    s1 = SpanStats(
        group="a",
        count=1,
        total_ns=100,
        mean_ns=100.0,
        median_ns=100.0,
        min_ns=100,
        max_ns=100,
        stddev_ns=0.0,
        p50_ns=100.0,
        p95_ns=100.0,
        p99_ns=100.0,
    )
    summary = TraceSummary(group_by="name", stats=(s1,), total_events=1, total_duration_ns=100)
    assert summary.get("a") is s1
    assert summary.get("nonexistent") is None


def test_summary_groups():
    s1 = SpanStats(
        group="x",
        count=1,
        total_ns=0,
        mean_ns=0.0,
        median_ns=0.0,
        min_ns=0,
        max_ns=0,
        stddev_ns=0.0,
        p50_ns=0.0,
        p95_ns=0.0,
        p99_ns=0.0,
    )
    s2 = SpanStats(
        group="y",
        count=1,
        total_ns=0,
        mean_ns=0.0,
        median_ns=0.0,
        min_ns=0,
        max_ns=0,
        stddev_ns=0.0,
        p50_ns=0.0,
        p95_ns=0.0,
        p99_ns=0.0,
    )
    summary = TraceSummary(group_by="name", stats=(s1, s2), total_events=2, total_duration_ns=0)
    assert summary.groups == ["x", "y"]


# ── summarize() ──


def test_summarize_from_tracer():
    tracer = Tracer()
    with tracer.span("op_a", category="compute"):
        pass
    with tracer.span("op_b", category="compute"):
        pass
    result = summarize(tracer)
    assert result.total_events == 2
    assert result.group_by == "name"
    assert set(result.groups) == {"op_a", "op_b"}


def test_summarize_from_event_list():
    events = [
        _make_event(event_id="0", name="foo", start_ns=0, end_ns=100),
        _make_event(event_id="1", name="foo", start_ns=100, end_ns=300),
        _make_event(event_id="2", name="bar", start_ns=300, end_ns=350),
    ]
    result = summarize(events)
    foo = result.get("foo")
    assert foo is not None
    assert foo.count == 2
    assert foo.total_ns == 300
    assert foo.mean_ns == 150.0
    assert foo.min_ns == 100
    assert foo.max_ns == 200

    bar = result.get("bar")
    assert bar is not None
    assert bar.count == 1
    assert bar.total_ns == 50


def test_summarize_group_by_category():
    events = [
        _make_event(event_id="0", name="a", category="compute", start_ns=0, end_ns=100),
        _make_event(event_id="1", name="b", category="compute", start_ns=0, end_ns=200),
        _make_event(event_id="2", name="c", category="memory", start_ns=0, end_ns=50),
    ]
    result = summarize(events, group_by="category")
    assert result.group_by == "category"
    compute = result.get("compute")
    assert compute is not None
    assert compute.count == 2


def test_summarize_group_by_scope():
    events = [
        _make_event(event_id="0", scope="decode.token.0", start_ns=0, end_ns=100),
        _make_event(event_id="1", scope="decode.token.0", start_ns=0, end_ns=200),
        _make_event(event_id="2", scope="prefill", start_ns=0, end_ns=50),
    ]
    result = summarize(events, group_by="scope")
    assert result.group_by == "scope"
    assert "decode.token.0" in result.groups
    assert "prefill" in result.groups


def test_summarize_group_by_token_index():
    events = [
        _make_event(event_id="0", token_index=0, start_ns=0, end_ns=100),
        _make_event(event_id="1", token_index=0, start_ns=0, end_ns=200),
        _make_event(event_id="2", token_index=1, start_ns=0, end_ns=50),
        _make_event(event_id="3", token_index=None, start_ns=0, end_ns=50),
    ]
    result = summarize(events, group_by="token_index")
    assert result.group_by == "token_index"
    assert "0" in result.groups
    assert "1" in result.groups
    assert "none" in result.groups


def test_summarize_group_by_callable():
    events = [
        _make_event(event_id="0", name="short", start_ns=0, end_ns=100),
        _make_event(event_id="1", name="very_long_name", start_ns=0, end_ns=200),
    ]
    result = summarize(events, group_by=lambda e: "long" if len(e.name) > 5 else "short")
    assert result.group_by == "custom"
    assert set(result.groups) == {"long", "short"}


def test_summarize_invalid_group_by():
    with pytest.raises(ValueError, match="Unknown group_by"):
        summarize([], group_by="invalid_key")


def test_summarize_empty_events():
    result = summarize([])
    assert result.total_events == 0
    assert result.stats == ()
    assert result.total_duration_ns == 0


def test_summarize_exclude_zero_duration():
    events = [
        _make_event(event_id="0", start_ns=100, end_ns=100),  # zero duration
        _make_event(event_id="1", start_ns=0, end_ns=500),
    ]
    result = summarize(events, exclude_zero_duration=True)
    assert result.total_events == 1


def test_summarize_single_event():
    events = [_make_event(event_id="0", start_ns=0, end_ns=1000)]
    result = summarize(events)
    stats = result.get("test")
    assert stats is not None
    assert stats.count == 1
    assert stats.mean_ns == 1000.0
    assert stats.median_ns == 1000.0
    assert stats.stddev_ns == 0.0
    assert stats.min_ns == 1000
    assert stats.max_ns == 1000


# ── Outlier detection ──


def test_outlier_detection():
    # 9 events at ~100ns, 1 event way out at 10000ns
    events = [_make_event(event_id=str(i), start_ns=0, end_ns=100) for i in range(9)]
    events.append(_make_event(event_id="9", start_ns=0, end_ns=10000))
    result = summarize(events, outlier_threshold=2.0)
    stats = result.get("test")
    assert stats is not None
    assert "9" in stats.outlier_ids


def test_outlier_detection_disabled():
    events = [_make_event(event_id=str(i), start_ns=0, end_ns=100) for i in range(9)]
    events.append(_make_event(event_id="9", start_ns=0, end_ns=10000))
    result = summarize(events, outlier_threshold=0)
    stats = result.get("test")
    assert stats is not None
    assert stats.outlier_ids == ()


def test_no_outliers_when_uniform():
    events = [_make_event(event_id=str(i), start_ns=0, end_ns=100) for i in range(10)]
    result = summarize(events, outlier_threshold=2.0)
    stats = result.get("test")
    assert stats is not None
    assert stats.outlier_ids == ()


# ── Statistics accuracy ──


def test_stddev_calculation():
    # events: durations 100, 200, 300
    events = [
        _make_event(event_id="0", start_ns=0, end_ns=100),
        _make_event(event_id="1", start_ns=0, end_ns=200),
        _make_event(event_id="2", start_ns=0, end_ns=300),
    ]
    result = summarize(events)
    stats = result.get("test")
    assert stats is not None
    # population stddev of [100, 200, 300] = sqrt(((100-200)^2 + 0 + (300-200)^2) / 3)
    expected = math.sqrt((10000 + 0 + 10000) / 3)
    assert stats.stddev_ns == pytest.approx(expected)


def test_total_duration():
    events = [
        _make_event(event_id="0", name="a", start_ns=0, end_ns=100),
        _make_event(event_id="1", name="b", start_ns=0, end_ns=200),
    ]
    result = summarize(events)
    assert result.total_duration_ns == 300


def test_groups_are_sorted():
    events = [
        _make_event(event_id="0", name="zebra", start_ns=0, end_ns=100),
        _make_event(event_id="1", name="alpha", start_ns=0, end_ns=100),
        _make_event(event_id="2", name="middle", start_ns=0, end_ns=100),
    ]
    result = summarize(events)
    assert result.groups == ["alpha", "middle", "zebra"]


# ── format_summary ──


def test_format_summary_empty():
    summary = TraceSummary(group_by="name", stats=(), total_events=0, total_duration_ns=0)
    output = format_summary(summary)
    assert "No events" in output


def test_format_summary_has_header():
    events = [_make_event(event_id="0", start_ns=0, end_ns=1000)]
    summary = summarize(events)
    output = format_summary(summary)
    assert "Group" in output
    assert "Count" in output
    assert "Mean" in output
    assert "P95" in output


def test_format_summary_contains_group_name():
    events = [_make_event(event_id="0", name="my_span", start_ns=0, end_ns=1000)]
    summary = summarize(events)
    output = format_summary(summary)
    assert "my_span" in output


def test_format_summary_truncates_long_names():
    events = [_make_event(event_id="0", name="a" * 50, start_ns=0, end_ns=1000)]
    summary = summarize(events)
    output = format_summary(summary)
    assert "..." in output


def test_format_summary_total_duration():
    events = [_make_event(event_id="0", start_ns=0, end_ns=5_000_000)]
    summary = summarize(events)
    output = format_summary(summary)
    assert "Total duration" in output


# ── _fmt_duration ──


def test_fmt_duration_nanoseconds():
    assert _fmt_duration(500) == "500ns"


def test_fmt_duration_microseconds():
    assert _fmt_duration(5_000) == "5.0us"


def test_fmt_duration_milliseconds():
    assert _fmt_duration(5_000_000) == "5.00ms"


def test_fmt_duration_seconds():
    assert _fmt_duration(2_500_000_000) == "2.500s"
