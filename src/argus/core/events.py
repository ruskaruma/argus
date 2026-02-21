from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

VALID_CATEGORIES = frozenset({"compute", "memory", "phase", "token", "kernel", "system"})


@dataclass(frozen=True, slots=True)
class TraceEvent:
    """Immutable trace event — the atom of the Argus system.

    All timestamps are nanoseconds from a monotonic clock.
    Duration is computed, never stored.
    """

    event_id: str
    name: str
    start_ns: int
    end_ns: int
    category: str
    scope: str
    parent_id: str | None = None
    token_index: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ns(self) -> int:
        return self.end_ns - self.start_ns

    @property
    def duration_us(self) -> float:
        return self.duration_ns / 1_000

    @property
    def duration_ms(self) -> float:
        return self.duration_ns / 1_000_000
