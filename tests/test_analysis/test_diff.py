"""Tests for trace diffing — structural comparison of two inference runs."""

from __future__ import annotations

import json
from io import StringIO
from typing import Any

import pytest

from argus.analysis.diff import (
    SpanDelta,
    TraceDiff,
    diff_traces,
    export_diff_chrome,
)
from argus.core.events import TraceEvent
from argus.core.tracer import Tracer


def _make_event(
    event_id: str = "0",
    name: str = "op",
    start_ns: int = 0,
    end_ns: int = 1000,
    category: str = "compute",
    scope: str = "",
    parent_id: str | None = None,
    token_index: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> TraceEvent:
    return TraceEvent(
        event_id=event_id,
        name=name,
        start_ns=start_ns,
        end_ns=end_ns,
        category=category,
        scope=scope,
        parent_id=parent_id,
        token_index=token_index,
        metadata=metadata or {},
    )


def _build_tracer(events: list[TraceEvent]) -> Tracer:
    t = Tracer()
    for e in events:
        t.record_event(e)
    return t


# ---------- SpanDelta ----------


class TestSpanDelta:
    def test_delta_ns(self) -> None:
        d = SpanDelta("s", "n", "compute", None, 100, 300)
        assert d.delta_ns == 200

    def test_delta_pct_normal(self) -> None:
        d = SpanDelta("s", "n", "compute", None, 100, 200)
        assert d.delta_pct == pytest.approx(100.0)

    def test_delta_pct_zero_baseline(self) -> None:
        d = SpanDelta("s", "n", "compute", None, 0, 100)
        assert d.delta_pct == float("inf")

    def test_delta_pct_both_zero(self) -> None:
        d = SpanDelta("s", "n", "compute", None, 0, 0)
        assert d.delta_pct == 0.0

    def test_negative_delta(self) -> None:
        d = SpanDelta("s", "n", "compute", None, 500, 200)
        assert d.delta_ns == -300
        assert d.delta_pct == pytest.approx(-60.0)


# ---------- diff_traces with Tracer instances ----------


class TestDiffTracesBasic:
    def test_empty_traces(self) -> None:
        result = diff_traces(Tracer(), Tracer())
        assert result.matched == []
        assert result.added == []
        assert result.removed == []
        assert result.outlier_tokens == []

    def test_identical_traces(self) -> None:
        events = [_make_event(scope="s1", name="op1")]
        result = diff_traces(_build_tracer(events), _build_tracer(events))
        assert len(result.matched) == 1
        assert result.matched[0].delta_ns == 0
        assert result.added == []
        assert result.removed == []

    def test_added_spans(self) -> None:
        a = _build_tracer([_make_event(scope="s1", name="op1")])
        b = _build_tracer(
            [
                _make_event(scope="s1", name="op1"),
                _make_event(event_id="1", scope="s2", name="op2"),
            ]
        )
        result = diff_traces(a, b)
        assert len(result.matched) == 1
        assert len(result.added) == 1
        assert result.added[0].scope == "s2"

    def test_removed_spans(self) -> None:
        a = _build_tracer(
            [
                _make_event(scope="s1", name="op1"),
                _make_event(event_id="1", scope="s2", name="op2"),
            ]
        )
        b = _build_tracer([_make_event(scope="s1", name="op1")])
        result = diff_traces(a, b)
        assert len(result.matched) == 1
        assert len(result.removed) == 1
        assert result.removed[0].scope == "s2"

    def test_duration_delta(self) -> None:
        a = _build_tracer([_make_event(scope="s1", name="op1", end_ns=1000)])
        b = _build_tracer([_make_event(scope="s1", name="op1", end_ns=3000)])
        result = diff_traces(a, b)
        assert len(result.matched) == 1
        assert result.matched[0].delta_ns == 2000
        assert result.matched[0].delta_pct == pytest.approx(200.0)


# ---------- diff_traces with event lists ----------


class TestDiffTracesEventLists:
    def test_event_lists(self) -> None:
        a = [_make_event(scope="s1", name="op1", end_ns=100)]
        b = [_make_event(scope="s1", name="op1", end_ns=200)]
        result = diff_traces(a, b)
        assert len(result.matched) == 1
        assert result.matched[0].delta_ns == 100

    def test_mixed_tracer_and_list(self) -> None:
        a = _build_tracer([_make_event(scope="s1", name="op1")])
        b = [_make_event(scope="s1", name="op1", end_ns=2000)]
        result = diff_traces(a, b)
        assert len(result.matched) == 1
        assert result.matched[0].delta_ns == 1000


# ---------- diff_traces from JSON files ----------


class TestDiffTracesJSON:
    def test_from_json_files(self, tmp_path: Any) -> None:
        events_a = [_make_event(scope="s1", name="op1", end_ns=1000)]
        events_b = [_make_event(scope="s1", name="op1", end_ns=3000)]

        path_a = tmp_path / "a.json"
        path_b = tmp_path / "b.json"

        # Write Chrome Trace JSON manually
        for path, events in [(path_a, events_a), (path_b, events_b)]:
            payload = {
                "traceEvents": [
                    {
                        "ph": "X",
                        "name": e.name,
                        "cat": e.category,
                        "ts": e.start_ns / 1000.0,
                        "dur": e.duration_ns / 1000.0,
                        "pid": 1,
                        "tid": 3,
                        "args": {
                            "event_id": e.event_id,
                            "scope": e.scope,
                        },
                    }
                    for e in events
                ]
            }
            path.write_text(json.dumps(payload))

        result = diff_traces(str(path_a), path_b)
        assert len(result.matched) == 1
        assert result.matched[0].duration_a_ns == 1000
        assert result.matched[0].duration_b_ns == 3000


# ---------- Outlier detection ----------


class TestOutlierDetection:
    def test_no_outliers_when_similar(self) -> None:
        a = [_make_event(scope="t0", name="op", token_index=0, end_ns=100)]
        b = [_make_event(scope="t0", name="op", token_index=0, end_ns=150)]
        result = diff_traces(a, b, outlier_threshold=2.0)
        assert result.outlier_tokens == []

    def test_detects_slower_outlier(self) -> None:
        a = [_make_event(scope="t0", name="op", token_index=0, end_ns=100)]
        b = [_make_event(scope="t0", name="op", token_index=0, end_ns=500)]
        result = diff_traces(a, b, outlier_threshold=2.0)
        assert len(result.outlier_tokens) == 1
        assert result.outlier_tokens[0].token_index == 0
        assert result.outlier_tokens[0].ratio == 5.0

    def test_detects_faster_outlier(self) -> None:
        a = [_make_event(scope="t0", name="op", token_index=0, end_ns=500)]
        b = [_make_event(scope="t0", name="op", token_index=0, end_ns=100)]
        result = diff_traces(a, b, outlier_threshold=2.0)
        assert len(result.outlier_tokens) == 1
        assert result.outlier_tokens[0].ratio == pytest.approx(0.2)

    def test_custom_threshold(self) -> None:
        a = [_make_event(scope="t0", name="op", token_index=0, end_ns=100)]
        b = [_make_event(scope="t0", name="op", token_index=0, end_ns=250)]
        assert diff_traces(a, b, outlier_threshold=2.0).outlier_tokens != []
        assert diff_traces(a, b, outlier_threshold=3.0).outlier_tokens == []

    def test_multiple_tokens_mixed(self) -> None:
        a = [
            _make_event(event_id="0", scope="t0", name="op", token_index=0, end_ns=100),
            _make_event(event_id="1", scope="t1", name="op", token_index=1, end_ns=100),
            _make_event(event_id="2", scope="t2", name="op", token_index=2, end_ns=100),
        ]
        b = [
            _make_event(event_id="0", scope="t0", name="op", token_index=0, end_ns=100),
            _make_event(event_id="1", scope="t1", name="op", token_index=1, end_ns=500),
            _make_event(event_id="2", scope="t2", name="op", token_index=2, end_ns=100),
        ]
        result = diff_traces(a, b, outlier_threshold=2.0)
        assert len(result.outlier_tokens) == 1
        assert result.outlier_tokens[0].token_index == 1

    def test_token_only_in_one_trace(self) -> None:
        a = [_make_event(scope="t0", name="op", token_index=0, end_ns=100)]
        b: list[TraceEvent] = []
        result = diff_traces(a, b, outlier_threshold=2.0)
        # Token 0 exists only in A, ratio=0 → 1/0 triggers inverse check
        assert len(result.outlier_tokens) == 1

    def test_aggregates_multiple_spans_per_token(self) -> None:
        a = [
            _make_event(event_id="0", scope="t0.fwd", name="fwd", token_index=0, end_ns=50),
            _make_event(event_id="1", scope="t0.attn", name="attn", token_index=0, end_ns=50),
        ]
        b = [
            _make_event(event_id="0", scope="t0.fwd", name="fwd", token_index=0, end_ns=200),
            _make_event(event_id="1", scope="t0.attn", name="attn", token_index=0, end_ns=200),
        ]
        result = diff_traces(a, b, outlier_threshold=2.0)
        assert len(result.outlier_tokens) == 1
        assert result.outlier_tokens[0].total_a_ns == 100
        assert result.outlier_tokens[0].total_b_ns == 400


# ---------- TraceDiff.summary() ----------


class TestTraceDiffSummary:
    def test_summary_structure(self) -> None:
        a = [
            _make_event(scope="s1", name="op1", end_ns=100, token_index=0),
            _make_event(event_id="1", scope="s2", name="op2", end_ns=200),
        ]
        b = [
            _make_event(scope="s1", name="op1", end_ns=300, token_index=0),
            _make_event(event_id="2", scope="s3", name="op3", end_ns=400),
        ]
        result = diff_traces(a, b)
        s = result.summary()
        assert s["matched_spans"] == 1
        assert s["added_spans"] == 1
        assert s["removed_spans"] == 1
        assert isinstance(s["deltas"], list)
        assert isinstance(s["added"], list)
        assert isinstance(s["removed"], list)
        assert isinstance(s["outliers"], list)

    def test_summary_serializable(self) -> None:
        a = [_make_event(scope="s1", name="op1", end_ns=100)]
        b = [_make_event(scope="s1", name="op1", end_ns=300)]
        result = diff_traces(a, b)
        json_str = json.dumps(result.summary())
        assert json.loads(json_str) == result.summary()


# ---------- Chrome Trace diff export ----------


class TestExportDiffChrome:
    def test_export_to_stringio(self) -> None:
        a = [_make_event(scope="s1", name="op1", end_ns=100)]
        b = [_make_event(scope="s1", name="op1", end_ns=300)]
        buf = StringIO()
        result = export_diff_chrome(a, b, buf)
        assert isinstance(result, TraceDiff)
        buf.seek(0)
        data = json.load(buf)
        assert "traceEvents" in data
        assert data["metadata"]["diff_mode"] is True

    def test_export_to_file(self, tmp_path: Any) -> None:
        a = [_make_event(scope="s1", name="op1", end_ns=100)]
        b = [_make_event(scope="s1", name="op1", end_ns=300)]
        path = tmp_path / "diff.json"
        export_diff_chrome(a, b, str(path))
        data = json.loads(path.read_text())
        assert "traceEvents" in data

    def test_events_have_trace_labels(self) -> None:
        a = [_make_event(scope="s1", name="op1")]
        b = [_make_event(scope="s1", name="op1")]
        buf = StringIO()
        export_diff_chrome(a, b, buf)
        buf.seek(0)
        data = json.load(buf)
        names = [e["name"] for e in data["traceEvents"] if e.get("ph") == "X"]
        assert any("[A]" in n for n in names)
        assert any("[B]" in n for n in names)

    def test_matched_spans_have_delta(self) -> None:
        a = [_make_event(scope="s1", name="op1", end_ns=100)]
        b = [_make_event(scope="s1", name="op1", end_ns=300)]
        buf = StringIO()
        export_diff_chrome(a, b, buf)
        buf.seek(0)
        data = json.load(buf)
        x_events = [e for e in data["traceEvents"] if e.get("ph") == "X"]
        assert all("delta_ns" in e["args"] for e in x_events)

    def test_outlier_tokens_flagged(self) -> None:
        a = [_make_event(scope="t0", name="op", token_index=0, end_ns=100)]
        b = [_make_event(scope="t0", name="op", token_index=0, end_ns=500)]
        buf = StringIO()
        export_diff_chrome(a, b, buf, outlier_threshold=2.0)
        buf.seek(0)
        data = json.load(buf)
        x_events = [e for e in data["traceEvents"] if e.get("ph") == "X"]
        assert any(e["args"].get("outlier") is True for e in x_events)
        assert any(e.get("cname") == "terrible" for e in x_events)

    def test_thread_name_metadata(self) -> None:
        a = [_make_event(scope="s1", name="op1")]
        b = [_make_event(scope="s1", name="op1")]
        buf = StringIO()
        export_diff_chrome(a, b, buf)
        buf.seek(0)
        data = json.load(buf)
        meta_events = [e for e in data["traceEvents"] if e.get("ph") == "M"]
        assert len(meta_events) > 0
        names = [e["args"]["name"] for e in meta_events]
        assert any("baseline" in n.lower() for n in names)
        assert any("comparison" in n.lower() for n in names)

    def test_tid_separation(self) -> None:
        a = [_make_event(scope="s1", name="op1", category="compute")]
        b = [_make_event(scope="s1", name="op1", category="compute")]
        buf = StringIO()
        export_diff_chrome(a, b, buf)
        buf.seek(0)
        data = json.load(buf)
        x_events = [e for e in data["traceEvents"] if e.get("ph") == "X"]
        tids = {e["tid"] for e in x_events}
        assert 103 in tids  # A: 100 + compute(3)
        assert 203 in tids  # B: 200 + compute(3)


# ---------- Duplicate span keys ----------


class TestDuplicateSpanKeys:
    def test_first_occurrence_used(self) -> None:
        a = [
            _make_event(event_id="0", scope="s1", name="op", end_ns=100),
            _make_event(event_id="1", scope="s1", name="op", end_ns=999),
        ]
        b = [_make_event(scope="s1", name="op", end_ns=200)]
        result = diff_traces(a, b)
        assert len(result.matched) == 1
        assert result.matched[0].duration_a_ns == 100


# ---------- Synthetic multi-token trace ----------


class TestSyntheticMultiTokenTrace:
    def _build_decode_trace(self, token_durations: list[int]) -> list[TraceEvent]:
        events: list[TraceEvent] = []
        t = 0
        for i, dur in enumerate(token_durations):
            events.append(
                _make_event(
                    event_id=str(i),
                    name="forward_pass",
                    scope=f"decode.token.{i}.forward",
                    category="compute",
                    token_index=i,
                    start_ns=t,
                    end_ns=t + dur,
                )
            )
            t += dur
        return events

    def test_full_decode_diff(self) -> None:
        a = self._build_decode_trace([100, 100, 100, 100, 100])
        b = self._build_decode_trace([100, 100, 500, 100, 100])
        result = diff_traces(a, b, outlier_threshold=2.0)
        assert len(result.matched) == 5
        assert len(result.outlier_tokens) == 1
        assert result.outlier_tokens[0].token_index == 2

    def test_different_token_counts(self) -> None:
        a = self._build_decode_trace([100, 100, 100])
        b = self._build_decode_trace([100, 100, 100, 100, 100])
        result = diff_traces(a, b)
        assert len(result.matched) == 3
        assert len(result.added) == 2
        assert len(result.removed) == 0
