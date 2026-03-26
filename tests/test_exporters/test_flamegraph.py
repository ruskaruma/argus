from __future__ import annotations

import json
from io import StringIO

from argus.core.events import TraceEvent
from argus.exporters.flamegraph import (
    export_flamegraph,
    generate_folded_stacks,
    generate_speedscope,
)


def _make_event(**overrides: object) -> TraceEvent:
    defaults: dict[str, object] = {
        "event_id": "0",
        "name": "test",
        "start_ns": 1000,
        "end_ns": 2000,
        "category": "compute",
        "scope": "test",
    }
    defaults.update(overrides)
    return TraceEvent(**defaults)  # type: ignore[arg-type]


# ---------- folded stacks: aggregated ----------


def test_empty_events_produce_empty_string():
    assert generate_folded_stacks([]) == ""


def test_single_event_folded():
    e = _make_event(scope="root", start_ns=0, end_ns=5000)
    result = generate_folded_stacks([e])
    assert result == "root 5000\n"


def test_parent_child_stack_order():
    parent = _make_event(
        event_id="0", name="parent", scope="decode", start_ns=0, end_ns=10000
    )
    child = _make_event(
        event_id="1",
        name="child",
        scope="decode.forward",
        parent_id="0",
        start_ns=1000,
        end_ns=8000,
    )
    result = generate_folded_stacks([parent, child])
    lines = result.strip().split("\n")
    assert len(lines) == 2
    child_line = [l for l in lines if "decode.forward" in l][0]
    assert child_line.startswith("decode;decode.forward")
    parent_line = [
        l for l in lines if l.startswith("decode ") or l.startswith("decode\t")
    ][0]
    assert "decode " in parent_line


def test_self_time_computation():
    parent = _make_event(event_id="0", scope="root", start_ns=0, end_ns=10000)
    child = _make_event(
        event_id="1", scope="root.child", parent_id="0", start_ns=1000, end_ns=8000
    )
    result = generate_folded_stacks([parent, child])
    lines = result.strip().split("\n")
    parent_line = [l for l in lines if not "child" in l][0]
    child_line = [l for l in lines if "child" in l][0]
    # Parent self-time: 10000 - 7000 = 3000
    assert parent_line == "root 3000"
    # Child self-time: 7000 (no children)
    assert child_line == "root;root.child 7000"


def test_zero_duration_events_excluded():
    e = _make_event(start_ns=1000, end_ns=1000)
    result = generate_folded_stacks([e])
    assert result == ""


def test_scope_used_as_frame_name():
    e = _make_event(scope="decode.token.0.layer.5", start_ns=0, end_ns=1000)
    result = generate_folded_stacks([e])
    assert "decode.token.0.layer.5" in result


def test_name_used_when_scope_empty():
    e = _make_event(name="my_op", scope="", start_ns=0, end_ns=1000)
    result = generate_folded_stacks([e])
    assert "my_op " in result


def test_three_level_nesting():
    root = _make_event(event_id="0", scope="a", start_ns=0, end_ns=10000)
    mid = _make_event(event_id="1", scope="b", parent_id="0", start_ns=0, end_ns=8000)
    leaf = _make_event(event_id="2", scope="c", parent_id="1", start_ns=0, end_ns=5000)
    result = generate_folded_stacks([root, mid, leaf])
    lines = result.strip().split("\n")
    leaf_line = [l for l in lines if "c" in l][0]
    assert leaf_line == "a;b;c 5000"


# ---------- folded stacks: per_token ----------


def test_per_token_mode_groups_by_token():
    e0 = _make_event(
        event_id="0", scope="layer.0", token_index=0, start_ns=0, end_ns=1000
    )
    e1 = _make_event(
        event_id="1", scope="layer.0", token_index=1, start_ns=2000, end_ns=3000
    )
    result = generate_folded_stacks([e0, e1], mode="per_token")
    lines = result.strip().split("\n")
    assert len(lines) == 2
    assert any("token_0;" in l for l in lines)
    assert any("token_1;" in l for l in lines)


def test_per_token_includes_parent_context():
    parent = _make_event(event_id="0", scope="decode", start_ns=0, end_ns=10000)
    child = _make_event(
        event_id="1",
        scope="decode.forward",
        parent_id="0",
        token_index=0,
        start_ns=1000,
        end_ns=5000,
    )
    result = generate_folded_stacks([parent, child], mode="per_token")
    assert "decode;decode.forward" in result


# ---------- speedscope JSON ----------


def test_speedscope_schema_field():
    data = generate_speedscope([_make_event(start_ns=0, end_ns=1000)])
    assert data["$schema"] == "https://www.speedscope.app/file-format-schema.json"


def test_speedscope_has_required_keys():
    data = generate_speedscope([_make_event(start_ns=0, end_ns=1000)])
    assert "shared" in data
    assert "profiles" in data
    assert "frames" in data["shared"]


def test_speedscope_frames_populated():
    e = _make_event(scope="root.child", start_ns=0, end_ns=1000)
    data = generate_speedscope([e])
    frame_names = [f["name"] for f in data["shared"]["frames"]]
    assert "root.child" in frame_names


def test_speedscope_profile_structure():
    e = _make_event(start_ns=0, end_ns=1000)
    data = generate_speedscope([e])
    assert len(data["profiles"]) == 1
    profile = data["profiles"][0]
    assert profile["type"] == "sampled"
    assert profile["unit"] == "nanoseconds"
    assert "samples" in profile
    assert "weights" in profile
    assert profile["startValue"] == 0
    assert profile["endValue"] > 0


def test_speedscope_weights_match_self_time():
    parent = _make_event(event_id="0", scope="root", start_ns=0, end_ns=10000)
    child = _make_event(
        event_id="1", scope="root.child", parent_id="0", start_ns=0, end_ns=7000
    )
    data = generate_speedscope([parent, child])
    weights = data["profiles"][0]["weights"]
    assert sorted(weights) == [3000, 7000]


def test_speedscope_per_token_creates_multiple_profiles():
    events = [
        _make_event(
            event_id=str(i), scope=f"op_{i}", token_index=i, start_ns=0, end_ns=1000
        )
        for i in range(3)
    ]
    data = generate_speedscope(events, mode="per_token")
    assert len(data["profiles"]) == 3


def test_speedscope_empty_events():
    data = generate_speedscope([])
    assert data["profiles"] == []
    assert data["shared"]["frames"] == []


def test_speedscope_exporter_field():
    data = generate_speedscope([_make_event(start_ns=0, end_ns=1000)])
    assert data["exporter"] == "argus@0.1.0"


def test_speedscope_name_field():
    data = generate_speedscope([_make_event(start_ns=0, end_ns=1000)])
    assert data["name"] == "Argus Flamegraph"


# ---------- export_flamegraph ----------


def test_export_to_txt_file(tmp_path):
    from argus.core.tracer import Tracer

    tracer = Tracer()
    with tracer.span("op", scope="root"):
        pass
    path = tmp_path / "flame.txt"
    export_flamegraph(tracer, str(path))
    content = path.read_text()
    assert "root" in content


def test_export_to_json_file(tmp_path):
    from argus.core.tracer import Tracer

    tracer = Tracer()
    with tracer.span("op", scope="root"):
        pass
    path = tmp_path / "flame.json"
    export_flamegraph(tracer, str(path))
    data = json.loads(path.read_text())
    assert "$schema" in data
    assert "profiles" in data


def test_export_to_stringio():
    from argus.core.tracer import Tracer

    tracer = Tracer()
    with tracer.span("op", scope="root"):
        pass
    sio = StringIO()
    export_flamegraph(tracer, sio)
    assert "root" in sio.getvalue()


def test_export_invalid_mode_raises():
    from argus.core.tracer import Tracer

    tracer = Tracer()
    import pytest

    with pytest.raises(ValueError, match="mode must be"):
        export_flamegraph(tracer, StringIO(), mode="bad")


def test_export_per_token_mode(tmp_path):
    from argus.core.tracer import Tracer

    tracer = Tracer()
    for i in range(3):
        with tracer.span("op", scope=f"layer.{i}", token_index=i):
            pass
    path = tmp_path / "flame.txt"
    export_flamegraph(tracer, str(path), mode="per_token")
    content = path.read_text()
    assert "token_0" in content
    assert "token_1" in content
    assert "token_2" in content


def test_export_path_object(tmp_path):
    from argus.core.tracer import Tracer
    from pathlib import Path

    tracer = Tracer()
    with tracer.span("op", scope="root"):
        pass
    path = Path(tmp_path / "flame.txt")
    export_flamegraph(tracer, path)
    assert path.read_text().strip()


# ---------- integration-style ----------


def test_nested_trace_produces_valid_flamegraph():
    from argus.core.tracer import Tracer

    tracer = Tracer()
    with tracer.span("decode", category="phase", scope="decode"):
        for i in range(3):
            with tracer.span(
                "token_generate",
                category="token",
                scope=f"decode.token.{i}",
                token_index=i,
            ):
                with tracer.span(
                    "forward",
                    category="compute",
                    scope=f"decode.token.{i}.forward",
                    token_index=i,
                ):
                    pass

    folded = generate_folded_stacks(tracer.events)
    lines = folded.strip().split("\n")
    assert len(lines) > 0
    for line in lines:
        parts = line.rsplit(" ", 1)
        assert len(parts) == 2
        assert int(parts[1]) > 0

    speedscope = generate_speedscope(tracer.events)
    assert len(speedscope["profiles"]) == 1
    assert len(speedscope["shared"]["frames"]) > 0


def test_per_token_flamegraph_isolation():
    from argus.core.tracer import Tracer

    tracer = Tracer()
    with tracer.span("decode", category="phase", scope="decode"):
        for i in range(2):
            with tracer.span("token", scope=f"decode.t{i}", token_index=i):
                with tracer.span("attn", scope=f"decode.t{i}.attn", token_index=i):
                    pass

    speedscope = generate_speedscope(tracer.events, mode="per_token")
    assert len(speedscope["profiles"]) == 2
    profile_names = {p["name"] for p in speedscope["profiles"]}
    assert "token_0" in profile_names
    assert "token_1" in profile_names


def test_folded_stacks_flamegraph_pl_compatible():
    """Each line must be: semicolon-separated-stack SPACE positive-integer."""
    from argus.core.tracer import Tracer

    tracer = Tracer()
    with tracer.span("root", scope="main"):
        with tracer.span("child", scope="main.work"):
            pass

    folded = generate_folded_stacks(tracer.events)
    for line in folded.strip().split("\n"):
        stack, _, value = line.rpartition(" ")
        assert stack
        assert int(value) > 0
        assert ";" in stack or stack == "main"
