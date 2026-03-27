from __future__ import annotations

import pytest

from argus import Tracer
from argus.hooks.layers import (
    LayerHookHandle,
    _get_layer_blocks,
    register_layer_hooks,
)

# ---------------------------------------------------------------------------
# Mock nn.Module that supports register_forward_pre_hook / register_forward_hook
# ---------------------------------------------------------------------------


class _HookHandle:
    def __init__(self, hooks: dict, key: int):
        self._hooks = hooks
        self._key = key

    def remove(self) -> None:
        self._hooks.pop(self._key, None)


class MockModule:
    """Minimal torch.nn.Module stand-in with hook support."""

    def __init__(self) -> None:
        self._forward_pre_hooks: dict[int, object] = {}
        self._forward_hooks: dict[int, object] = {}
        self._hook_id = 0

    def register_forward_pre_hook(self, hook):
        key = self._hook_id
        self._hook_id += 1
        self._forward_pre_hooks[key] = hook
        return _HookHandle(self._forward_pre_hooks, key)

    def register_forward_hook(self, hook):
        key = self._hook_id
        self._hook_id += 1
        self._forward_hooks[key] = hook
        return _HookHandle(self._forward_hooks, key)

    def __call__(self, *args, **kwargs):
        for hook in list(self._forward_pre_hooks.values()):
            hook(self, args)
        output = self.forward(*args, **kwargs)
        for hook in list(self._forward_hooks.values()):
            hook(self, args, output)
        return output

    def forward(self, *args, **kwargs):
        return args


# ---------------------------------------------------------------------------
# Mock model architectures
# ---------------------------------------------------------------------------


class MockGPT2Block(MockModule):
    __qualname__ = "GPT2Block"

    def __init__(self):
        super().__init__()
        self.attn = MockModule()
        self.mlp = MockModule()
        self.ln_1 = MockModule()
        self.ln_2 = MockModule()


# Override class name to match _LAYER_MAPS key
MockGPT2Block.__name__ = "GPT2Block"


class MockLlamaDecoderLayer(MockModule):
    __qualname__ = "LlamaDecoderLayer"

    def __init__(self):
        super().__init__()
        self.self_attn = MockModule()
        self.mlp = MockModule()
        self.input_layernorm = MockModule()
        self.post_attention_layernorm = MockModule()


MockLlamaDecoderLayer.__name__ = "LlamaDecoderLayer"


class MockMistralDecoderLayer(MockModule):
    __qualname__ = "MistralDecoderLayer"

    def __init__(self):
        super().__init__()
        self.self_attn = MockModule()
        self.mlp = MockModule()
        self.input_layernorm = MockModule()
        self.post_attention_layernorm = MockModule()


MockMistralDecoderLayer.__name__ = "MistralDecoderLayer"


def _make_gpt2_model(num_layers: int = 2):
    class FakeTransformer:
        def __init__(self):
            self.h = [MockGPT2Block() for _ in range(num_layers)]

    class FakeModel:
        def __init__(self):
            self.transformer = FakeTransformer()

    return FakeModel()


def _make_llama_model(num_layers: int = 2):
    class FakeInner:
        def __init__(self):
            self.layers = [MockLlamaDecoderLayer() for _ in range(num_layers)]

    class FakeModel:
        def __init__(self):
            self.model = FakeInner()

    return FakeModel()


def _make_mistral_model(num_layers: int = 2):
    class FakeInner:
        def __init__(self):
            self.layers = [MockMistralDecoderLayer() for _ in range(num_layers)]

    class FakeModel:
        def __init__(self):
            self.model = FakeInner()

    return FakeModel()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLayerHookHandle:
    def test_remove_clears_hooks(self):
        handle = LayerHookHandle()
        module = MockModule()
        h = module.register_forward_pre_hook(lambda m, i: None)
        handle.add(h)
        assert len(module._forward_pre_hooks) == 1
        handle.remove()
        assert len(module._forward_pre_hooks) == 0

    def test_context_manager_removes_hooks(self):
        module = MockModule()
        with LayerHookHandle() as handle:
            h = module.register_forward_pre_hook(lambda m, i: None)
            handle.add(h)
            assert len(module._forward_pre_hooks) == 1
        assert len(module._forward_pre_hooks) == 0


class TestGetLayerBlocks:
    def test_gpt2_architecture(self):
        model = _make_gpt2_model(3)
        blocks = _get_layer_blocks(model)
        assert len(blocks) == 3

    def test_llama_architecture(self):
        model = _make_llama_model(4)
        blocks = _get_layer_blocks(model)
        assert len(blocks) == 4

    def test_unsupported_architecture(self):
        class BadModel:
            pass

        with pytest.raises(ValueError, match="Unsupported model architecture"):
            _get_layer_blocks(BadModel())


class TestRegisterLayerHooksGPT2:
    def test_registers_hooks_on_all_layers(self):
        tracer = Tracer()
        model = _make_gpt2_model(2)
        handle = register_layer_hooks(model, tracer)
        try:
            for block in model.transformer.h:
                assert len(block._forward_pre_hooks) == 1
                assert len(block._forward_hooks) == 1
                assert len(block.attn._forward_pre_hooks) == 1
                assert len(block.mlp._forward_pre_hooks) == 1
                assert len(block.ln_1._forward_pre_hooks) == 1
                assert len(block.ln_2._forward_pre_hooks) == 1
        finally:
            handle.remove()

    def test_forward_creates_spans(self):
        tracer = Tracer()
        model = _make_gpt2_model(1)
        block = model.transformer.h[0]
        with register_layer_hooks(model, tracer):
            block("dummy_input")

        events = tracer.events
        names = [e.name for e in events]
        assert "layer.0" in names

    def test_sublayer_spans_created(self):
        tracer = Tracer()
        model = _make_gpt2_model(1)
        block = model.transformer.h[0]
        with register_layer_hooks(model, tracer):
            block.attn("x")
            block.mlp("x")

        names = [e.name for e in tracer.events]
        assert "layer.0.attention" in names
        assert "layer.0.mlp" in names

    def test_scope_prefix(self):
        tracer = Tracer()
        model = _make_gpt2_model(1)
        block = model.transformer.h[0]
        with register_layer_hooks(model, tracer, scope_prefix="decode.token.5"):
            block("x")

        scopes = [e.scope for e in tracer.events]
        assert any("decode.token.5.layer.0" in s for s in scopes)

    def test_spans_have_compute_category(self):
        tracer = Tracer()
        model = _make_gpt2_model(1)
        block = model.transformer.h[0]
        with register_layer_hooks(model, tracer):
            block("x")

        for event in tracer.events:
            assert event.category == "compute"

    def test_hooks_removed_after_context_exit(self):
        tracer = Tracer()
        model = _make_gpt2_model(1)
        block = model.transformer.h[0]
        with register_layer_hooks(model, tracer):
            pass

        assert len(block._forward_pre_hooks) == 0
        assert len(block._forward_hooks) == 0
        assert len(block.attn._forward_pre_hooks) == 0

    def test_parent_child_nesting(self):
        tracer = Tracer()
        model = _make_gpt2_model(1)
        block = model.transformer.h[0]
        with register_layer_hooks(model, tracer), tracer.span("outer", scope="outer"):
            block("x")

        block_event = next(e for e in tracer.events if e.name == "layer.0")
        outer_event = next(e for e in tracer.events if e.name == "outer")
        assert block_event.parent_id == outer_event.event_id


class TestRegisterLayerHooksLLaMA:
    def test_llama_layers_detected(self):
        tracer = Tracer()
        model = _make_llama_model(2)
        with register_layer_hooks(model, tracer):
            model.model.layers[0]("x")
            model.model.layers[1]("x")

        names = [e.name for e in tracer.events]
        assert "layer.0" in names
        assert "layer.1" in names

    def test_llama_sublayers(self):
        tracer = Tracer()
        model = _make_llama_model(1)
        block = model.model.layers[0]
        with register_layer_hooks(model, tracer):
            block.self_attn("x")

        names = [e.name for e in tracer.events]
        assert "layer.0.attention" in names

    def test_llama_scope_dot_notation(self):
        tracer = Tracer()
        model = _make_llama_model(1)
        block = model.model.layers[0]
        with register_layer_hooks(model, tracer):
            block.self_attn("x")

        scopes = [e.scope for e in tracer.events]
        assert "layer.0.attention" in scopes


class TestRegisterLayerHooksMistral:
    def test_mistral_layers_detected(self):
        tracer = Tracer()
        model = _make_mistral_model(2)
        with register_layer_hooks(model, tracer):
            model.model.layers[0]("x")

        names = [e.name for e in tracer.events]
        assert "layer.0" in names


class TestMultipleLayerSpans:
    def test_two_layers_produce_separate_spans(self):
        tracer = Tracer()
        model = _make_gpt2_model(2)
        with register_layer_hooks(model, tracer):
            model.transformer.h[0]("x")
            model.transformer.h[1]("x")

        layer_events = [
            e for e in tracer.events if e.name.startswith("layer.") and "." not in e.name[6:]
        ]
        assert len(layer_events) == 2
        assert layer_events[0].scope == "layer.0"
        assert layer_events[1].scope == "layer.1"

    def test_no_spans_after_hook_removal(self):
        tracer = Tracer()
        model = _make_gpt2_model(1)
        block = model.transformer.h[0]
        handle = register_layer_hooks(model, tracer)
        handle.remove()

        block("x")
        assert len(tracer.events) == 0
