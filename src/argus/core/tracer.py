from __future__ import annotations

from typing import Any

from argus.core.clock import monotonic_ns
from argus.core.events import TraceEvent


class SpanContext:
    """Context manager returned by Tracer.span(). Records timing on exit."""

    __slots__ = (
        "_tracer",
        "_event_id",
        "_name",
        "_category",
        "_scope",
        "_metadata",
        "_token_index",
        "_parent_id",
        "_start_ns",
    )

    def __init__(
        self,
        tracer: Tracer,
        event_id: str,
        name: str,
        category: str,
        scope: str,
        metadata: dict[str, Any],
        token_index: int | None,
    ) -> None:
        self._tracer = tracer
        self._event_id = event_id
        self._name = name
        self._category = category
        self._scope = scope
        self._metadata = metadata
        self._token_index = token_index
        self._parent_id: str | None = None
        self._start_ns: int = 0

    @property
    def event_id(self) -> str:
        return self._event_id

    def add_metadata(self, key: str, value: str | int | float | bool | None) -> None:
        self._metadata[key] = value

    def __enter__(self) -> SpanContext:
        self._parent_id = self._tracer._parent_stack[-1] if self._tracer._parent_stack else None
        self._tracer._parent_stack.append(self._event_id)
        self._start_ns = monotonic_ns()
        return self

    def __exit__(self, *_: object) -> None:
        end_ns = monotonic_ns()
        if end_ns < self._start_ns:
            end_ns = self._start_ns
        event = TraceEvent(
            event_id=self._event_id,
            name=self._name,
            start_ns=self._start_ns,
            end_ns=end_ns,
            category=self._category,
            scope=self._scope,
            parent_id=self._parent_id,
            token_index=self._token_index,
            metadata=dict(self._metadata),
        )
        self._tracer._events.append(event)
        if self._tracer._parent_stack and self._tracer._parent_stack[-1] == self._event_id:
            self._tracer._parent_stack.pop()


class Tracer:
    """Collects trace events via span context managers.

    Not thread-safe. Single-threaded tracing only in v0.1.
    """

    __slots__ = ("_events", "_next_id", "_parent_stack")

    def __init__(self) -> None:
        self._events: list[TraceEvent] = []
        self._next_id: int = 0
        self._parent_stack: list[str] = []

    def _generate_id(self) -> str:
        eid = str(self._next_id)
        self._next_id += 1
        return eid

    def span(
        self,
        name: str,
        category: str = "compute",
        scope: str = "",
        metadata: dict[str, Any] | None = None,
        token_index: int | None = None,
    ) -> SpanContext:
        """Open a traced span. Use as a context manager."""
        event_id = self._generate_id()
        return SpanContext(
            tracer=self,
            event_id=event_id,
            name=name,
            category=category,
            scope=scope,
            metadata=metadata if metadata is not None else {},
            token_index=token_index,
        )

    def record_event(self, event: TraceEvent) -> None:
        """Append a pre-built TraceEvent. Public API for hooks."""
        self._events.append(event)

    def instant(
        self,
        name: str,
        category: str = "compute",
        scope: str = "",
        metadata: dict[str, Any] | None = None,
        token_index: int | None = None,
    ) -> TraceEvent:
        """Record a zero-duration point-in-time event."""
        now = monotonic_ns()
        event = TraceEvent(
            event_id=self._generate_id(),
            name=name,
            start_ns=now,
            end_ns=now,
            category=category,
            scope=scope,
            parent_id=self._parent_stack[-1] if self._parent_stack else None,
            token_index=token_index,
            metadata=metadata if metadata is not None else {},
        )
        self._events.append(event)
        return event

    @property
    def events(self) -> list[TraceEvent]:
        return list(self._events)

    def reset(self) -> None:
        self._events.clear()
        self._next_id = 0
        self._parent_stack.clear()

    def get_events(
        self,
        category: str | None = None,
        token_index: int | None = None,
    ) -> list[TraceEvent]:
        result = list(self._events)
        if category is not None:
            result = [e for e in result if e.category == category]
        if token_index is not None:
            result = [e for e in result if e.token_index == token_index]
        return result
