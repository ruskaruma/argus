from __future__ import annotations

import sys

import pytest

from argus.core.tracer import Tracer


@pytest.mark.requires_torch
def test_trace_generate_produces_events(tiny_model):
    from argus.hooks.pytorch import trace_generate

    model, input_ids = tiny_model
    t = Tracer()
    trace_generate(model, input_ids, tracer=t, max_new_tokens=3)
    assert len(t.events) > 0


@pytest.mark.requires_torch
def test_prefill_phase_exists(tiny_model):
    from argus.hooks.pytorch import trace_generate

    model, input_ids = tiny_model
    t = Tracer()
    trace_generate(model, input_ids, tracer=t, max_new_tokens=3)
    phases = [e for e in t.events if e.name == "prefill" and e.category == "phase"]
    assert len(phases) == 1


@pytest.mark.requires_torch
def test_decode_phase_exists(tiny_model):
    from argus.hooks.pytorch import trace_generate

    model, input_ids = tiny_model
    t = Tracer()
    trace_generate(model, input_ids, tracer=t, max_new_tokens=3)
    phases = [e for e in t.events if e.name == "decode" and e.category == "phase"]
    assert len(phases) == 1


@pytest.mark.requires_torch
def test_token_events_per_token(tiny_model):
    from argus.hooks.pytorch import trace_generate

    model, input_ids = tiny_model
    t = Tracer()
    trace_generate(model, input_ids, tracer=t, max_new_tokens=5)
    token_events = [e for e in t.events if e.category == "token"]
    assert len(token_events) >= 1


@pytest.mark.requires_torch
def test_forward_pass_events(tiny_model):
    from argus.hooks.pytorch import trace_generate

    model, input_ids = tiny_model
    t = Tracer()
    trace_generate(model, input_ids, tracer=t, max_new_tokens=3)
    forwards = [e for e in t.events if e.name == "forward_pass"]
    # at least 1 for prefill + 1 per decode token
    assert len(forwards) >= 2


@pytest.mark.requires_torch
def test_span_nesting_structure(tiny_model):
    from argus.hooks.pytorch import trace_generate

    model, input_ids = tiny_model
    t = Tracer()
    trace_generate(model, input_ids, tracer=t, max_new_tokens=3)
    events_by_id = {e.event_id: e for e in t.events}
    for e in t.events:
        if e.parent_id is not None:
            assert e.parent_id in events_by_id


@pytest.mark.requires_torch
def test_token_index_attribution(tiny_model):
    from argus.hooks.pytorch import trace_generate

    model, input_ids = tiny_model
    t = Tracer()
    trace_generate(model, input_ids, tracer=t, max_new_tokens=3)
    token_events = sorted(
        [e for e in t.events if e.category == "token"],
        key=lambda e: e.token_index or 0,
    )
    for i, e in enumerate(token_events):
        assert e.token_index == i


@pytest.mark.requires_torch
def test_kv_tracker_integration(tiny_model):
    from argus.hooks.kvcache import KVCacheTracker
    from argus.hooks.pytorch import trace_generate

    model, input_ids = tiny_model
    t = Tracer()
    kv = KVCacheTracker(t)
    trace_generate(model, input_ids, tracer=t, max_new_tokens=3, kv_tracker=kv)
    memory_events = [e for e in t.events if e.category == "memory"]
    assert len(memory_events) >= 1


@pytest.mark.requires_torch
def test_returns_generated_tokens(tiny_model):
    import torch

    from argus.hooks.pytorch import trace_generate

    model, input_ids = tiny_model
    t = Tracer()
    result = trace_generate(model, input_ids, tracer=t, max_new_tokens=3)
    assert isinstance(result, torch.Tensor)
    assert result.shape[1] > input_ids.shape[1]


def test_lazy_import_no_torch_on_load():
    if "torch" in sys.modules:
        pytest.skip("torch already imported by test infrastructure")
    import argus.hooks.pytorch  # noqa: F401

    assert "torch" not in sys.modules
