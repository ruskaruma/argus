"""Trace summary statistics — aggregate metrics from trace events."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from argus.core.events import TraceEvent


@dataclass(frozen=True, slots=True)
class LatencyStats:
    """Latency percentiles in milliseconds."""

    count: int
    min_ms: float
    max_ms: float
    mean_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float


@dataclass(frozen=True, slots=True)
class CategoryStats:
    """Per-category aggregate statistics."""

    count: int
    total_duration_ms: float


@dataclass(frozen=True, slots=True)
class TraceSummary:
    """Aggregate statistics computed from a list of TraceEvents."""

    total_events: int
    total_duration_ms: float
    categories: dict[str, CategoryStats] = field(default_factory=dict)
    token_count: int = 0
    token_latency: LatencyStats | None = None
    tokens_per_second: float | None = None
    prefill_duration_ms: float | None = None
    decode_duration_ms: float | None = None
    peak_memory_bytes: int | None = None


def _compute_latency_stats(durations_ms: list[float]) -> LatencyStats:
    """Compute latency percentiles from a list of durations in ms."""
    sorted_d = sorted(durations_ms)
    n = len(sorted_d)
    return LatencyStats(
        count=n,
        min_ms=sorted_d[0],
        max_ms=sorted_d[-1],
        mean_ms=statistics.mean(sorted_d),
        p50_ms=sorted_d[int(n * 0.50)] if n > 1 else sorted_d[0],
        p90_ms=sorted_d[int(n * 0.90)] if n > 1 else sorted_d[0],
        p95_ms=sorted_d[int(n * 0.95)] if n > 1 else sorted_d[0],
        p99_ms=sorted_d[int(n * 0.99)] if n > 1 else sorted_d[0],
    )


def summarize(events: list[TraceEvent]) -> TraceSummary:
    """Compute aggregate statistics from a list of trace events.

    Analyzes events to produce per-category counts, token latency percentiles,
    throughput, phase durations, and peak memory usage.
    """
    if not events:
        return TraceSummary(total_events=0, total_duration_ms=0.0)

    min_start = min(e.start_ns for e in events)
    max_end = max(e.end_ns for e in events)
    total_duration_ms = (max_end - min_start) / 1_000_000

    categories: dict[str, CategoryStats] = {}
    cat_counts: dict[str, int] = {}
    cat_durations: dict[str, float] = {}
    for e in events:
        cat_counts[e.category] = cat_counts.get(e.category, 0) + 1
        cat_durations[e.category] = cat_durations.get(e.category, 0.0) + e.duration_ms
    for cat in cat_counts:
        categories[cat] = CategoryStats(
            count=cat_counts[cat],
            total_duration_ms=cat_durations[cat],
        )

    token_events = [e for e in events if e.category == "token"]
    token_count = len(token_events)
    token_latency: LatencyStats | None = None
    tokens_per_second: float | None = None
    if token_events:
        durations_ms = [e.duration_ms for e in token_events]
        token_latency = _compute_latency_stats(durations_ms)
        decode_events = [e for e in events if e.category == "phase" and e.name == "decode"]
        if decode_events:
            decode_wall_ms = decode_events[0].duration_ms
            if decode_wall_ms > 0:
                tokens_per_second = token_count / (decode_wall_ms / 1_000)

    prefill_duration_ms: float | None = None
    decode_duration_ms: float | None = None
    for e in events:
        if e.category == "phase" and e.name == "prefill":
            prefill_duration_ms = e.duration_ms
        elif e.category == "phase" and e.name == "decode":
            decode_duration_ms = e.duration_ms

    peak_memory_bytes: int | None = None
    for e in events:
        if e.category == "memory" and "cache_size_bytes" in e.metadata:
            size = e.metadata["cache_size_bytes"]
            if isinstance(size, int) and (peak_memory_bytes is None or size > peak_memory_bytes):
                peak_memory_bytes = size

    return TraceSummary(
        total_events=len(events),
        total_duration_ms=total_duration_ms,
        categories=categories,
        token_count=token_count,
        token_latency=token_latency,
        tokens_per_second=tokens_per_second,
        prefill_duration_ms=prefill_duration_ms,
        decode_duration_ms=decode_duration_ms,
        peak_memory_bytes=peak_memory_bytes,
    )
