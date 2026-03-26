from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

if TYPE_CHECKING:
    from argus.core.events import TraceEvent


def _build_stacks(
    events: list[TraceEvent],
) -> list[tuple[str, int]]:
    """Build folded stack lines with self-time from parent-child relationships.

    Returns a list of (folded_stack_string, self_time_ns) tuples.
    Only events with positive self-time are included.
    """
    by_id: dict[str, TraceEvent] = {e.event_id: e for e in events}
    children: dict[str, list[str]] = defaultdict(list)
    for e in events:
        if e.parent_id is not None:
            children[e.parent_id].append(e.event_id)

    results: list[tuple[str, int]] = []
    for event in events:
        if event.duration_ns <= 0:
            continue

        child_duration = sum(
            by_id[cid].duration_ns
            for cid in children.get(event.event_id, [])
            if cid in by_id and by_id[cid].duration_ns > 0
        )
        self_time = event.duration_ns - child_duration
        if self_time <= 0:
            continue

        frames: list[str] = []
        current: TraceEvent | None = event
        while current is not None:
            label = current.scope if current.scope else current.name
            frames.append(label)
            current = by_id.get(current.parent_id) if current.parent_id else None
        frames.reverse()

        results.append((";".join(frames), self_time))

    return results


def _events_for_mode(
    events: list[TraceEvent],
    mode: str,
) -> dict[int | None, list[TraceEvent]]:
    """Group events by token_index for per_token mode, or return all for aggregated."""
    if mode == "aggregated":
        return {None: events}

    grouped: dict[int | None, list[TraceEvent]] = defaultdict(list)
    no_token: list[TraceEvent] = []
    token_indices: set[int] = set()

    for e in events:
        if e.token_index is not None:
            grouped[e.token_index].append(e)
            token_indices.add(e.token_index)
        else:
            no_token.append(e)

    # Attach parent events (no token_index) to each token group so stacks resolve
    if no_token and token_indices:
        for ti in token_indices:
            grouped[ti] = no_token + grouped[ti]
    elif no_token and not token_indices:
        grouped[None] = no_token

    return grouped


def generate_folded_stacks(
    events: list[TraceEvent],
    mode: str = "aggregated",
) -> str:
    """Generate folded stack text from trace events.

    Args:
        events: List of TraceEvent objects.
        mode: "aggregated" merges all events; "per_token" groups by token_index.

    Returns:
        Folded stack text compatible with flamegraph.pl.
    """
    groups = _events_for_mode(events, mode)
    lines: list[str] = []

    for key in sorted(groups, key=lambda k: (k is None, k)):
        group_events = groups[key]
        stacks = _build_stacks(group_events)
        if mode == "per_token" and key is not None:
            stacks = [(f"token_{key};{stack}", time) for stack, time in stacks]
        for stack, self_time in stacks:
            lines.append(f"{stack} {self_time}")

    return "\n".join(lines) + "\n" if lines else ""


def generate_speedscope(
    events: list[TraceEvent],
    mode: str = "aggregated",
) -> dict[str, Any]:
    """Generate speedscope JSON structure from trace events.

    Args:
        events: List of TraceEvent objects.
        mode: "aggregated" merges all events; "per_token" creates one profile per token.

    Returns:
        Dict matching the speedscope file format schema.
    """
    groups = _events_for_mode(events, mode)
    frame_index: dict[str, int] = {}
    frames: list[dict[str, str]] = []
    profiles: list[dict[str, Any]] = []

    def _get_frame_idx(name: str) -> int:
        if name not in frame_index:
            frame_index[name] = len(frames)
            frames.append({"name": name})
        return frame_index[name]

    for key in sorted(groups, key=lambda k: (k is None, k)):
        group_events = groups[key]
        stacks = _build_stacks(group_events)

        if not stacks:
            continue

        if mode == "per_token" and key is not None:
            stacks = [(f"token_{key};{stack}", time) for stack, time in stacks]

        samples: list[list[int]] = []
        weights: list[int] = []
        for stack_str, self_time in stacks:
            frame_names = stack_str.split(";")
            sample = [_get_frame_idx(f) for f in frame_names]
            samples.append(sample)
            weights.append(self_time)

        profile_name = f"token_{key}" if key is not None else "aggregated"
        end_value = sum(weights)

        profiles.append(
            {
                "type": "sampled",
                "name": profile_name,
                "unit": "nanoseconds",
                "startValue": 0,
                "endValue": end_value,
                "samples": samples,
                "weights": weights,
            }
        )

    return {
        "$schema": "https://www.speedscope.app/file-format-schema.json",
        "shared": {"frames": frames},
        "profiles": profiles,
        "name": "Argus Flamegraph",
        "exporter": "argus@0.1.0",
    }


def export_flamegraph(
    tracer: Any,
    dest: str | Path | IO[str],
    mode: str = "aggregated",
) -> None:
    """Export trace events as flamegraph data.

    Output format is determined by file extension:
    - .json: speedscope JSON format (loadable at speedscope.app)
    - anything else (.txt, .folded, etc.): folded stacks (compatible with flamegraph.pl)

    Args:
        tracer: An argus Tracer instance.
        dest: File path (str/Path) or file-like object.
            File-like objects always produce folded stack output;
            use generate_speedscope() directly for JSON to a stream.
        mode: "aggregated" (default) or "per_token".

    Raises:
        ValueError: If mode is not "aggregated" or "per_token".
    """
    if mode not in ("aggregated", "per_token"):
        raise ValueError(f"mode must be 'aggregated' or 'per_token', got '{mode}'")

    events = tracer.events

    if isinstance(dest, (str, Path)):
        path = Path(dest)
        if path.suffix == ".json":
            payload = generate_speedscope(events, mode)
            with open(path, "w") as f:
                json.dump(payload, f)
        else:
            text = generate_folded_stacks(events, mode)
            with open(path, "w") as f:
                f.write(text)
    else:
        text = generate_folded_stacks(events, mode)
        dest.write(text)
