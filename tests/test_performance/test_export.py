from __future__ import annotations

import time
from io import StringIO

import pytest

from argus.core.events import TraceEvent
from argus.exporters.chrome import export_chrome_trace


@pytest.mark.slow
def test_export_10k_events_time():
    """Exporting 10k events must take < 500 ms."""
    events = [
        TraceEvent(
            event_id=str(i),
            name="test",
            start_ns=i * 1000,
            end_ns=i * 1000 + 500,
            category="compute",
            scope="test",
        )
        for i in range(10_000)
    ]
    sio = StringIO()
    start = time.monotonic_ns()
    export_chrome_trace(events, sio)
    elapsed_ms = (time.monotonic_ns() - start) / 1_000_000
    assert elapsed_ms < 500, f"Export time: {elapsed_ms:.0f} ms (limit: 500 ms)"
