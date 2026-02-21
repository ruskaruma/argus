from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

if TYPE_CHECKING:
    from argus.core.events import TraceEvent

CATEGORY_TO_TID: dict[str, int] = {
    "phase": 1,
    "token": 2,
    "compute": 3,
    "memory": 4,
    "kernel": 5,
    "system": 6,
}

_VALID_CATEGORIES = frozenset(CATEGORY_TO_TID)


def _event_to_chrome(event: TraceEvent) -> dict[str, Any]:
    if event.category not in _VALID_CATEGORIES:
        warnings.warn(
            f"Unknown category '{event.category}' in event '{event.name}'",
            stacklevel=2,
        )

    args: dict[str, Any] = {}
    for k, v in event.metadata.items():
        if not isinstance(v, (dict, list, set, tuple)):
            args[k] = v
    args["event_id"] = event.event_id
    args["scope"] = event.scope
    if event.parent_id is not None:
        args["parent_id"] = event.parent_id
    else:
        args.pop("parent_id", None)
    if event.token_index is not None:
        args["token_index"] = event.token_index
    else:
        args.pop("token_index", None)

    is_counter = event.category == "memory" and event.duration_ns == 0

    result: dict[str, Any] = {
        "ph": "C" if is_counter else "X",
        "name": event.name,
        "cat": event.category,
        "ts": event.start_ns / 1_000.0,
        "pid": 1,
        "tid": CATEGORY_TO_TID.get(event.category, 0),
        "args": args,
    }
    if not is_counter:
        result["dur"] = max(0, event.duration_ns) / 1_000.0
    return result


def events_to_chrome(events: list[TraceEvent]) -> list[dict[str, Any]]:
    return [_event_to_chrome(e) for e in events]


def export_chrome_trace(
    events: list[TraceEvent],
    dest: str | Path | IO[str],
) -> None:
    chrome_events = events_to_chrome(events)
    payload = {
        "traceEvents": chrome_events,
        "displayTimeUnit": "ns",
        "metadata": {
            "argus_version": "0.1.0",
            "clock_source": "monotonic_ns",
        },
    }
    if isinstance(dest, (str, Path)):
        with open(dest, "w") as f:
            json.dump(payload, f)
    else:
        json.dump(payload, dest)
