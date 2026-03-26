"""Trace diffing — structural comparison of two inference runs.

Compares two traces by matching spans on (scope, name) keys and reports
duration deltas, percentage changes, new/missing spans, and outlier tokens.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

from argus.core.events import TraceEvent
from argus.core.tracer import Tracer
from argus.exporters.chrome import CATEGORY_TO_TID


@dataclass(frozen=True, slots=True)
class SpanDelta:
    """Duration difference for a single matched span pair."""

    scope: str
    name: str
    category: str
    token_index: int | None
    duration_a_ns: int
    duration_b_ns: int

    @property
    def delta_ns(self) -> int:
        return self.duration_b_ns - self.duration_a_ns

    @property
    def delta_pct(self) -> float:
        if self.duration_a_ns == 0:
            return float("inf") if self.duration_b_ns > 0 else 0.0
        return (self.delta_ns / self.duration_a_ns) * 100.0


@dataclass(frozen=True, slots=True)
class TokenOutlier:
    """A token whose total duration changed significantly between runs."""

    token_index: int
    total_a_ns: int
    total_b_ns: int
    delta_ns: int
    ratio: float


@dataclass(slots=True)
class TraceDiff:
    """Result of comparing two traces."""

    matched: list[SpanDelta] = field(default_factory=list)
    added: list[TraceEvent] = field(default_factory=list)
    removed: list[TraceEvent] = field(default_factory=list)
    outlier_tokens: list[TokenOutlier] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        """Return a summary dict suitable for serialization."""
        return {
            "matched_spans": len(self.matched),
            "added_spans": len(self.added),
            "removed_spans": len(self.removed),
            "outlier_tokens": len(self.outlier_tokens),
            "deltas": [
                {
                    "scope": d.scope,
                    "name": d.name,
                    "category": d.category,
                    "token_index": d.token_index,
                    "duration_a_ns": d.duration_a_ns,
                    "duration_b_ns": d.duration_b_ns,
                    "delta_ns": d.delta_ns,
                    "delta_pct": d.delta_pct,
                }
                for d in self.matched
            ],
            "added": [
                {"scope": e.scope, "name": e.name, "category": e.category} for e in self.added
            ],
            "removed": [
                {"scope": e.scope, "name": e.name, "category": e.category} for e in self.removed
            ],
            "outliers": [
                {
                    "token_index": o.token_index,
                    "total_a_ns": o.total_a_ns,
                    "total_b_ns": o.total_b_ns,
                    "delta_ns": o.delta_ns,
                    "ratio": o.ratio,
                }
                for o in self.outlier_tokens
            ],
        }


def _span_key(event: TraceEvent) -> tuple[str, str]:
    return (event.scope, event.name)


def _events_from_source(source: Tracer | list[TraceEvent]) -> list[TraceEvent]:
    if isinstance(source, Tracer):
        return source.events
    return list(source)


def _load_events_from_json(path: str | Path) -> list[TraceEvent]:
    """Load TraceEvent list from an exported Chrome Trace JSON file."""
    with open(path) as f:
        data = json.load(f)

    fallback = data if isinstance(data, list) else []
    raw_events: list[dict[str, Any]] = data.get("traceEvents", fallback)
    events: list[TraceEvent] = []
    for i, entry in enumerate(raw_events):
        args = entry.get("args", {})
        ts_us = entry.get("ts", 0)
        dur_us = entry.get("dur", 0)
        start_ns = int(ts_us * 1_000)
        end_ns = int((ts_us + dur_us) * 1_000)
        events.append(
            TraceEvent(
                event_id=str(args.get("event_id", str(i))),
                name=entry.get("name", ""),
                start_ns=start_ns,
                end_ns=end_ns,
                category=entry.get("cat", "compute"),
                scope=str(args.get("scope", "")),
                parent_id=args.get("parent_id"),
                token_index=args.get("token_index"),
                metadata={
                    k: v
                    for k, v in args.items()
                    if k not in {"event_id", "scope", "parent_id", "token_index"}
                },
            )
        )
    return events


def _resolve_events(source: Tracer | list[TraceEvent] | str | Path) -> list[TraceEvent]:
    if isinstance(source, (str, Path)):
        return _load_events_from_json(source)
    return _events_from_source(source)


def _find_outlier_tokens(
    events_a: list[TraceEvent],
    events_b: list[TraceEvent],
    threshold: float,
) -> list[TokenOutlier]:
    """Identify tokens whose total duration ratio exceeds threshold."""
    totals_a: dict[int, int] = {}
    for e in events_a:
        if e.token_index is not None:
            totals_a[e.token_index] = totals_a.get(e.token_index, 0) + e.duration_ns
    totals_b: dict[int, int] = {}
    for e in events_b:
        if e.token_index is not None:
            totals_b[e.token_index] = totals_b.get(e.token_index, 0) + e.duration_ns

    all_tokens = sorted(set(totals_a) | set(totals_b))
    outliers: list[TokenOutlier] = []
    for tok in all_tokens:
        a = totals_a.get(tok, 0)
        b = totals_b.get(tok, 0)
        if a == 0 and b == 0:
            continue
        ratio = float("inf") if a == 0 else b / a
        is_outlier = (
            ratio == 0.0 or ratio == float("inf") or ratio >= threshold or 1.0 / ratio >= threshold
        )
        if is_outlier:
            outliers.append(
                TokenOutlier(
                    token_index=tok,
                    total_a_ns=a,
                    total_b_ns=b,
                    delta_ns=b - a,
                    ratio=ratio,
                )
            )
    return outliers


def diff_traces(
    trace_a: Tracer | list[TraceEvent] | str | Path,
    trace_b: Tracer | list[TraceEvent] | str | Path,
    *,
    outlier_threshold: float = 2.0,
) -> TraceDiff:
    """Compare two traces and produce a structured diff.

    Args:
        trace_a: Baseline trace (Tracer, event list, or path to Chrome JSON).
        trace_b: Comparison trace (same types accepted).
        outlier_threshold: Ratio threshold for flagging outlier tokens.
            A token is an outlier if its duration ratio (slower/faster)
            is >= this value. Defaults to 2.0 (>2x slower or faster).

    Returns:
        TraceDiff with matched span deltas, added/removed spans, and outlier tokens.
    """
    events_a = _resolve_events(trace_a)
    events_b = _resolve_events(trace_b)

    index_a: dict[tuple[str, str], TraceEvent] = {}
    for e in events_a:
        key = _span_key(e)
        if key not in index_a:
            index_a[key] = e

    index_b: dict[tuple[str, str], TraceEvent] = {}
    for e in events_b:
        key = _span_key(e)
        if key not in index_b:
            index_b[key] = e

    matched: list[SpanDelta] = []
    for key, ea in index_a.items():
        eb = index_b.get(key)
        if eb is not None:
            matched.append(
                SpanDelta(
                    scope=ea.scope,
                    name=ea.name,
                    category=ea.category,
                    token_index=ea.token_index,
                    duration_a_ns=ea.duration_ns,
                    duration_b_ns=eb.duration_ns,
                )
            )

    keys_a = set(index_a)
    keys_b = set(index_b)
    added = [index_b[k] for k in sorted(keys_b - keys_a)]
    removed = [index_a[k] for k in sorted(keys_a - keys_b)]

    outliers = _find_outlier_tokens(events_a, events_b, outlier_threshold)

    return TraceDiff(matched=matched, added=added, removed=removed, outlier_tokens=outliers)


def export_diff_chrome(
    trace_a: Tracer | list[TraceEvent] | str | Path,
    trace_b: Tracer | list[TraceEvent] | str | Path,
    dest: str | Path | IO[str],
    *,
    outlier_threshold: float = 2.0,
) -> TraceDiff:
    """Export a Chrome Trace with diff annotations (color-coded overlays).

    Trace A events appear on TID 100+category_tid, trace B on TID 200+category_tid.
    Matched spans include delta metadata. Outlier tokens are highlighted via
    color annotations.

    Returns the TraceDiff for programmatic use.
    """
    events_a = _resolve_events(trace_a)
    events_b = _resolve_events(trace_b)
    result = diff_traces(events_a, events_b, outlier_threshold=outlier_threshold)

    delta_lookup: dict[tuple[str, str], SpanDelta] = {}
    for d in result.matched:
        delta_lookup[(d.scope, d.name)] = d

    outlier_set: set[int] = {o.token_index for o in result.outlier_tokens}

    chrome_events: list[dict[str, Any]] = []

    # Trace A events (baseline) — TID offset 100
    for e in events_a:
        tid_base = CATEGORY_TO_TID.get(e.category, 0)
        entry = _chrome_event(e, pid=1, tid=100 + tid_base, label="A")
        key = _span_key(e)
        delta = delta_lookup.get(key)
        if delta is not None:
            entry["args"]["delta_ns"] = delta.delta_ns
            entry["args"]["delta_pct"] = round(delta.delta_pct, 2)
        if e.token_index is not None and e.token_index in outlier_set:
            entry["args"]["outlier"] = True
            entry["cname"] = "terrible"
        chrome_events.append(entry)

    # Trace B events (comparison) — TID offset 200
    for e in events_b:
        tid_base = CATEGORY_TO_TID.get(e.category, 0)
        entry = _chrome_event(e, pid=1, tid=200 + tid_base, label="B")
        key = _span_key(e)
        delta = delta_lookup.get(key)
        if delta is not None:
            entry["args"]["delta_ns"] = delta.delta_ns
            entry["args"]["delta_pct"] = round(delta.delta_pct, 2)
        if e.token_index is not None and e.token_index in outlier_set:
            entry["args"]["outlier"] = True
            entry["cname"] = "terrible"
        chrome_events.append(entry)

    # Thread name metadata events for labeling
    for label, offset in [("A (baseline)", 100), ("B (comparison)", 200)]:
        for cat, tid_base in CATEGORY_TO_TID.items():
            chrome_events.append(
                {
                    "ph": "M",
                    "name": "thread_name",
                    "pid": 1,
                    "tid": offset + tid_base,
                    "args": {"name": f"{label} — {cat}"},
                }
            )

    payload: dict[str, Any] = {
        "traceEvents": chrome_events,
        "displayTimeUnit": "ns",
        "metadata": {
            "argus_version": "0.1.0",
            "clock_source": "monotonic_ns",
            "diff_mode": True,
            "outlier_threshold": outlier_threshold,
        },
    }

    if isinstance(dest, (str, Path)):
        with open(dest, "w") as f:
            json.dump(payload, f)
    else:
        json.dump(payload, dest)

    return result


def _chrome_event(
    event: TraceEvent,
    pid: int,
    tid: int,
    label: str,
) -> dict[str, Any]:
    """Convert a TraceEvent to a Chrome Trace dict with diff metadata."""
    args: dict[str, Any] = {
        "event_id": event.event_id,
        "scope": event.scope,
        "trace": label,
    }
    if event.parent_id is not None:
        args["parent_id"] = event.parent_id
    if event.token_index is not None:
        args["token_index"] = event.token_index
    for k, v in event.metadata.items():
        if k not in {"event_id", "scope", "parent_id", "token_index"} and not isinstance(
            v, (dict, list, set, tuple)
        ):
            args[k] = v

    return {
        "ph": "X",
        "name": f"[{label}] {event.name}",
        "cat": event.category,
        "ts": event.start_ns / 1_000.0,
        "dur": max(0, event.duration_ns) / 1_000.0,
        "pid": pid,
        "tid": tid,
        "args": args,
    }
