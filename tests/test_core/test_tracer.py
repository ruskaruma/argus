from __future__ import annotations

from argus.core.tracer import Tracer


def test_empty_trace():
    assert Tracer().events == []


def test_single_span():
    t = Tracer()
    with t.span("op", category="compute"):
        pass
    assert len(t.events) == 1


def test_single_span_fields():
    t = Tracer()
    with t.span("op", category="compute", scope="test"):
        pass
    e = t.events[0]
    assert e.name == "op"
    assert e.category == "compute"
    assert e.scope == "test"
    assert e.event_id == "0"


def test_single_span_timing():
    t = Tracer()
    with t.span("op"):
        pass
    e = t.events[0]
    assert e.start_ns > 0
    assert e.end_ns >= e.start_ns
    assert e.duration_ns >= 0


def test_sequential_spans():
    t = Tracer()
    with t.span("a"):
        pass
    with t.span("b"):
        pass
    assert len(t.events) == 2
    assert t.events[0].parent_id is None
    assert t.events[1].parent_id is None


def test_nested_spans_parent_child():
    t = Tracer()
    with t.span("outer") as outer:  # noqa: SIM117
        with t.span("inner"):
            pass
    inner, outer_ev = t.events[0], t.events[1]
    assert inner.parent_id == outer.event_id
    assert outer_ev.parent_id is None


def test_deeply_nested_10_levels():
    t = Tracer()
    contexts = []
    for i in range(10):
        ctx = t.span(f"level_{i}")
        ctx.__enter__()
        contexts.append(ctx)
    for ctx in reversed(contexts):
        ctx.__exit__(None, None, None)
    events = t.events
    assert len(events) == 10
    # innermost exits first, so events[0] is deepest
    for i in range(1, 10):
        child = events[i - 1]
        parent = events[i]
        assert child.parent_id == parent.event_id


def test_parent_child_timing_containment():
    t = Tracer()
    with t.span("parent"):  # noqa: SIM117
        with t.span("child"):
            _ = sum(range(100))
    child, parent = t.events[0], t.events[1]
    assert parent.start_ns <= child.start_ns
    assert child.end_ns <= parent.end_ns


def test_events_chronological_order():
    t = Tracer()
    with t.span("outer"):  # noqa: SIM117
        with t.span("inner"):
            pass
    # inner closes first
    assert t.events[0].name == "inner"
    assert t.events[1].name == "outer"


def test_reset_clears_events():
    t = Tracer()
    with t.span("op"):
        pass
    t.reset()
    assert t.events == []


def test_reset_resets_id_counter():
    t = Tracer()
    with t.span("op"):
        pass
    t.reset()
    with t.span("op2"):
        pass
    assert t.events[0].event_id == "0"


def test_get_events_by_category():
    t = Tracer()
    with t.span("a", category="compute"):
        pass
    with t.span("b", category="memory"):
        pass
    with t.span("c", category="compute"):
        pass
    result = t.get_events(category="compute")
    assert len(result) == 2
    assert all(e.category == "compute" for e in result)


def test_get_events_by_token_index():
    t = Tracer()
    with t.span("a", token_index=5):
        pass
    with t.span("b", token_index=3):
        pass
    with t.span("c", token_index=5):
        pass
    result = t.get_events(token_index=5)
    assert len(result) == 2


def test_get_events_both_filters():
    t = Tracer()
    with t.span("a", category="token", token_index=3):
        pass
    with t.span("b", category="compute", token_index=3):
        pass
    with t.span("c", category="token", token_index=7):
        pass
    result = t.get_events(category="token", token_index=3)
    assert len(result) == 1
    assert result[0].name == "a"


def test_get_events_no_match():
    t = Tracer()
    with t.span("a", category="compute"):
        pass
    assert t.get_events(category="nonexistent") == []


def test_span_context_event_id_available():
    t = Tracer()
    with t.span("op") as ctx:
        eid = ctx.event_id
    assert isinstance(eid, str)
    assert eid == "0"


def test_span_context_add_metadata():
    t = Tracer()
    with t.span("op") as ctx:
        ctx.add_metadata("key", "value")
    assert t.events[0].metadata == {"key": "value"}


def test_monotonic_event_ids():
    t = Tracer()
    for _ in range(3):
        with t.span("op"):
            pass
    ids = [e.event_id for e in t.events]
    assert ids == ["0", "1", "2"]


def test_span_with_metadata_param():
    t = Tracer()
    with t.span("op", metadata={"a": 1}):
        pass
    assert t.events[0].metadata == {"a": 1}


def test_span_with_token_index():
    t = Tracer()
    with t.span("op", token_index=7):
        pass
    assert t.events[0].token_index == 7


def test_span_default_category():
    t = Tracer()
    with t.span("op"):
        pass
    assert t.events[0].category == "compute"


def test_get_events_returns_copy():
    t = Tracer()
    with t.span("a", category="compute"):
        pass
    result = t.get_events(category="compute")
    result.clear()
    assert len(t.get_events(category="compute")) == 1


def test_get_events_no_filter_returns_copy():
    t = Tracer()
    with t.span("a"):
        pass
    result = t.get_events()
    result.clear()
    assert len(t.get_events()) == 1
