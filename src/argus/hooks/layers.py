from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from argus.core.tracer import Tracer


_LAYER_MAPS: dict[str, dict[str, str]] = {
    "GPT2Block": {
        "attn": "attention",
        "mlp": "mlp",
        "ln_1": "ln_1",
        "ln_2": "ln_2",
    },
    "LlamaDecoderLayer": {
        "self_attn": "attention",
        "mlp": "mlp",
        "input_layernorm": "input_norm",
        "post_attention_layernorm": "post_norm",
    },
    "MistralDecoderLayer": {
        "self_attn": "attention",
        "mlp": "mlp",
        "input_layernorm": "input_norm",
        "post_attention_layernorm": "post_norm",
    },
}


def _get_layer_blocks(model: Any) -> list[Any]:
    """Extract the list of transformer layer blocks from a HuggingFace model."""
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    raise ValueError(
        f"Unsupported model architecture: {type(model).__name__}. "
        "Expected GPT2, LLaMA, or Mistral model family."
    )


def _get_sublayer_map(block: Any) -> dict[str, str]:
    """Return {attr_name: span_label} for a given block type."""
    block_type = type(block).__name__
    if block_type in _LAYER_MAPS:
        return _LAYER_MAPS[block_type]
    return {}


class LayerHookHandle:
    """Manages forward hooks registered on model layers. Use as a context manager."""

    __slots__ = ("_handles",)

    def __init__(self) -> None:
        self._handles: list[Any] = []

    def add(self, handle: Any) -> None:
        self._handles.append(handle)

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __enter__(self) -> LayerHookHandle:
        return self

    def __exit__(self, *_: object) -> None:
        self.remove()


def _make_pre_hook(tracer: Tracer, name: str, category: str, scope: str) -> Any:
    def pre_hook(module: Any, input: Any) -> None:  # noqa: A002
        ctx = tracer.span(name, category=category, scope=scope)
        ctx.__enter__()
        module.__argus_span_ctx__ = ctx

    return pre_hook


def _make_post_hook(
    tracer: Tracer,
) -> Any:
    def post_hook(module: Any, input: Any, output: Any) -> None:  # noqa: A002
        ctx = getattr(module, "__argus_span_ctx__", None)
        if ctx is not None:
            ctx.__exit__(None, None, None)
            del module.__argus_span_ctx__

    return post_hook


def register_layer_hooks(
    model: Any,
    tracer: Tracer,
    scope_prefix: str = "",
) -> LayerHookHandle:
    """Register forward hooks on each layer of a HuggingFace model for per-layer tracing.

    Attaches pre/post forward hooks to each transformer block and its sublayers
    (attention, MLP, normalization). Each hook creates a span in the tracer.

    Args:
        model: A HuggingFace causal language model (GPT2, LLaMA, Mistral).
        tracer: The Tracer instance to record spans into.
        scope_prefix: Optional prefix for scope strings (e.g. "decode.token.5").

    Returns:
        LayerHookHandle that removes all hooks when used as a context manager or
        when .remove() is called.
    """
    blocks = _get_layer_blocks(model)
    handle = LayerHookHandle()

    for layer_idx, block in enumerate(blocks):
        prefix = f"{scope_prefix}.layer.{layer_idx}" if scope_prefix else f"layer.{layer_idx}"
        block_name = f"layer.{layer_idx}"

        h_pre = block.register_forward_pre_hook(
            _make_pre_hook(tracer, block_name, "compute", prefix)
        )
        h_post = block.register_forward_hook(_make_post_hook(tracer))
        handle.add(h_pre)
        handle.add(h_post)

        sublayer_map = _get_sublayer_map(block)
        for attr_name, span_label in sublayer_map.items():
            sublayer = getattr(block, attr_name, None)
            if sublayer is None:
                continue
            sub_scope = f"{prefix}.{span_label}"
            sub_name = f"layer.{layer_idx}.{span_label}"

            h_pre = sublayer.register_forward_pre_hook(
                _make_pre_hook(tracer, sub_name, "compute", sub_scope)
            )
            h_post = sublayer.register_forward_hook(_make_post_hook(tracer))
            handle.add(h_pre)
            handle.add(h_post)

    return handle
