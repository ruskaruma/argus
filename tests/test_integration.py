from __future__ import annotations

import json
from io import StringIO


def test_import_argus():
    import argus

    assert hasattr(argus, "Tracer")
    assert hasattr(argus, "TraceEvent")
    assert hasattr(argus, "export_chrome")
    assert argus.__version__ == "0.1.0"


def test_basic_trace_to_json():
    import argus

    tracer = argus.Tracer()
    with tracer.span("a", category="compute"):
        pass
    with tracer.span("b", category="phase"):
        pass
    with tracer.span("c", category="memory"):
        pass
    sio = StringIO()
    argus.export_chrome(tracer, sio)
    data = json.loads(sio.getvalue())
    assert len(data["traceEvents"]) == 3


def test_nested_trace_to_json():
    import argus

    tracer = argus.Tracer()
    with tracer.span("outer", category="phase"):  # noqa: SIM117
        with tracer.span("inner", category="compute"):
            pass
    sio = StringIO()
    argus.export_chrome(tracer, sio)
    data = json.loads(sio.getvalue())
    events = data["traceEvents"]
    assert len(events) == 2
    inner_args = events[0]["args"]
    outer_args = events[1]["args"]
    assert inner_args["parent_id"] == outer_args["event_id"]


def test_export_to_tmpfile(tmp_path):
    import argus

    tracer = argus.Tracer()
    with tracer.span("op"):
        pass
    path = tmp_path / "trace.json"
    argus.export_chrome(tracer, str(path))
    data = json.loads(path.read_text())
    assert len(data["traceEvents"]) == 1


def test_target_usage_from_api_design():
    """Reproduce the basic tracing example from api-design.md."""
    import argus

    tracer = argus.Tracer()
    with tracer.span("my_operation", category="compute"):
        _ = sum(range(100))
    with tracer.span("outer", category="phase"):  # noqa: SIM117
        with tracer.span("inner", category="compute"):
            _ = sum(range(100))
    events = tracer.events
    assert len(events) == 3
