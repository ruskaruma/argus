from __future__ import annotations

import sys

import pytest

from argus.core.events import TraceEvent


@pytest.mark.slow
def test_memory_per_event():
    """Each TraceEvent must be < 512 bytes (rough check via sys.getsizeof)."""
    event = TraceEvent(
        event_id="0",
        name="test",
        start_ns=0,
        end_ns=1000,
        category="compute",
        scope="test",
        metadata={},
    )
    size = sys.getsizeof(event)
    assert size < 512, f"Event size: {size} bytes (limit: 512)"
