"""Trace summary statistics — aggregate timing stats from trace events."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from argus.core.events import TraceEvent
    from argus.core.tracer import Tracer


@dataclass(frozen=True, slots=True)
class SpanStats:
    """Aggregate timing statistics for a group of spans."""

    group: str
    count: int
    total_ns: int
    mean_ns: float
    median_ns: float
    min_ns: int
    max_ns: int
    stddev_ns: float
    p50_ns: float
    p95_ns: float
    p99_ns: float
    outlier_ids: tuple[str, ...] = field(default_factory=tuple)

    @property
    def mean_us(self) -> float:
        return self.mean_ns / 1_000

    @property
    def mean_ms(self) -> float:
        return self.mean_ns / 1_000_000

    @property
    def median_us(self) -> float:
        return self.median_ns / 1_000

    @property
    def p95_us(self) -> float:
        return self.p95_ns / 1_000

    @property
    def p99_us(self) -> float:
        return self.p99_ns / 1_000


def _percentile(sorted_values: list[int], p: float) -> float:
    """Compute the p-th percentile from a sorted list using linear interpolation."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = p / 100.0 * (len(sorted_values) - 1)
    low = int(math.floor(rank))
    high = min(low + 1, len(sorted_values) - 1)
    frac = rank - low
    return sorted_values[low] + frac * (sorted_values[high] - sorted_values[low])


def _compute_stats(
    group: str,
    events: list[TraceEvent],
    outlier_threshold: float,
) -> SpanStats:
    """Compute aggregate stats for a list of events belonging to one group."""
    durations = sorted(e.duration_ns for e in events)
    n = len(durations)
    total = sum(durations)
    mean = total / n
    median = _percentile(durations, 50)
    p50 = median
    p95 = _percentile(durations, 95)
    p99 = _percentile(durations, 99)

    variance = sum((d - mean) ** 2 for d in durations) / n
    stddev = math.sqrt(variance)

    outlier_ids: list[str] = []
    if outlier_threshold > 0 and stddev > 0:
        cutoff = mean + outlier_threshold * stddev
        outlier_ids = [e.event_id for e in events if e.duration_ns > cutoff]

    return SpanStats(
        group=group,
        count=n,
        total_ns=total,
        mean_ns=mean,
        median_ns=median,
        min_ns=durations[0],
        max_ns=durations[-1],
        stddev_ns=stddev,
        p50_ns=p50,
        p95_ns=p95,
        p99_ns=p99,
        outlier_ids=tuple(outlier_ids),
    )


@dataclass(frozen=True, slots=True)
class TraceSummary:
    """Summary of trace statistics grouped by a chosen key."""

    group_by: str
    stats: tuple[SpanStats, ...]
    total_events: int
    total_duration_ns: int

    def get(self, group: str) -> SpanStats | None:
        """Look up stats for a specific group."""
        for s in self.stats:
            if s.group == group:
                return s
        return None

    @property
    def groups(self) -> list[str]:
        return [s.group for s in self.stats]


def _resolve_group_fn(
    group_by: str | Callable[[TraceEvent], str],
) -> tuple[str, Callable[[TraceEvent], str]]:
    """Return (label, key_function) for the given group_by argument."""
    if callable(group_by):
        return "custom", group_by
    if group_by == "name":
        return "name", lambda e: e.name
    if group_by == "category":
        return "category", lambda e: e.category
    if group_by == "scope":
        return "scope", lambda e: e.scope
    if group_by == "token_index":
        return "token_index", lambda e: str(e.token_index) if e.token_index is not None else "none"
    raise ValueError(
        f"Unknown group_by key: {group_by!r}. "
        "Use 'name', 'category', 'scope', 'token_index', or a callable."
    )


def summarize(
    source: Tracer | Sequence[TraceEvent],
    group_by: str | Callable[[TraceEvent], str] = "name",
    outlier_threshold: float = 2.0,
    exclude_zero_duration: bool = False,
) -> TraceSummary:
    """Compute summary statistics from trace events.

    Args:
        source: A Tracer instance or a sequence of TraceEvent objects.
        group_by: How to group events. One of "name", "category", "scope",
            "token_index", or a callable(TraceEvent) -> str.
        outlier_threshold: Number of standard deviations above the mean to
            flag as an outlier. Set to 0 to disable outlier detection.
        exclude_zero_duration: If True, skip zero-duration (instant) events.

    Returns:
        A TraceSummary with per-group SpanStats.
    """
    from argus.core.tracer import Tracer

    events = source.events if isinstance(source, Tracer) else list(source)

    if exclude_zero_duration:
        events = [e for e in events if e.duration_ns > 0]

    label, key_fn = _resolve_group_fn(group_by)

    grouped: dict[str, list[TraceEvent]] = defaultdict(list)
    for event in events:
        grouped[key_fn(event)].append(event)

    all_stats: list[SpanStats] = []
    for group_name in sorted(grouped):
        group_events = grouped[group_name]
        all_stats.append(_compute_stats(group_name, group_events, outlier_threshold))

    total_dur = sum(e.duration_ns for e in events)

    return TraceSummary(
        group_by=label,
        stats=tuple(all_stats),
        total_events=len(events),
        total_duration_ns=total_dur,
    )


def format_summary(summary: TraceSummary) -> str:
    """Format a TraceSummary as a human-readable table.

    Returns a string with aligned columns showing key stats per group.
    """
    if not summary.stats:
        return f"No events to summarize (grouped by {summary.group_by})."

    cols = [
        f"{'Group':<30}",
        f"{'Count':>6}",
        f"{'Mean':>12}",
        f"{'Median':>12}",
        f"{'P95':>12}",
        f"{'P99':>12}",
        f"{'Min':>12}",
        f"{'Max':>12}",
        f"{'Outliers':>8}",
    ]
    header = " ".join(cols)
    separator = "-" * len(header)

    lines: list[str] = [
        f"Trace Summary (grouped by {summary.group_by}, {summary.total_events} events)",
        separator,
        header,
        separator,
    ]

    for s in summary.stats:
        group_label = s.group if len(s.group) <= 30 else s.group[:27] + "..."
        lines.append(
            f"{group_label:<30} {s.count:>6} {_fmt_duration(s.mean_ns):>12} "
            f"{_fmt_duration(s.median_ns):>12} {_fmt_duration(s.p95_ns):>12} "
            f"{_fmt_duration(s.p99_ns):>12} {_fmt_duration(s.min_ns):>12} "
            f"{_fmt_duration(s.max_ns):>12} {len(s.outlier_ids):>8}"
        )

    lines.append(separator)
    lines.append(f"Total duration: {_fmt_duration(summary.total_duration_ns)}")
    return "\n".join(lines)


def _fmt_duration(ns: float) -> str:
    """Format a nanosecond duration into a human-readable string."""
    if ns < 1_000:
        return f"{ns:.0f}ns"
    if ns < 1_000_000:
        return f"{ns / 1_000:.1f}us"
    if ns < 1_000_000_000:
        return f"{ns / 1_000_000:.2f}ms"
    return f"{ns / 1_000_000_000:.3f}s"
