"""Argus analysis — statistical summaries and reporting for trace data."""

from __future__ import annotations

from argus.analysis.summary import (
    SpanStats,
    TraceSummary,
    format_summary,
    summarize,
)

__all__ = ["SpanStats", "TraceSummary", "format_summary", "summarize"]
