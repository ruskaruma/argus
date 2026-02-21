from __future__ import annotations

import time


def monotonic_ns() -> int:
    """Single source of truth for all Argus timestamps.

    Wraps time.monotonic_ns() so a future C extension can replace it.
    """
    return time.monotonic_ns()
