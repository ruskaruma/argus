from __future__ import annotations

import time

import pytest

from argus.core.tracer import Tracer


@pytest.mark.slow
def test_span_creation_overhead():
    """Per-span overhead must be < 3000 ns (spec target 1000 ns, 3x margin for CI/VM)."""
    tracer = Tracer()
    # warmup: let list resize, dict creation paths settle
    for _ in range(1_000):
        with tracer.span("warmup", category="compute"):
            pass
    tracer.reset()
    n = 100_000
    start = time.monotonic_ns()
    for _ in range(n):
        with tracer.span("test", category="compute"):
            pass
    elapsed = time.monotonic_ns() - start
    per_span = elapsed / n
    assert per_span < 3000, f"Span overhead: {per_span:.0f} ns (limit: 3000 ns)"


@pytest.mark.slow
def test_clock_read_overhead():
    """Clock read must be < 200 ns."""
    from argus.core.clock import monotonic_ns

    # warmup
    for _ in range(1_000):
        monotonic_ns()
    n = 100_000
    start = time.monotonic_ns()
    for _ in range(n):
        monotonic_ns()
    elapsed = time.monotonic_ns() - start
    per_call = elapsed / n
    assert per_call < 200, f"Clock overhead: {per_call:.0f} ns (limit: 200 ns)"
