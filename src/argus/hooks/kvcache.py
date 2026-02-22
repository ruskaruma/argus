from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from argus.core.tracer import Tracer


def _compute_kv_cache_bytes(past_key_values: object) -> tuple[int, int]:
    """Returns (total_bytes, num_layers) from KV cache tensors."""
    total = 0
    num_layers = 0
    for layer_kv in past_key_values:  # type: ignore[union-attr]
        num_layers += 1
        for tensor in layer_kv:
            if tensor is None:
                continue
            total += tensor.nelement() * tensor.element_size()
    return total, num_layers


class KVCacheTracker:
    """Tracks KV cache size per token, emitting memory events to a tracer."""

    __slots__ = ("_tracer",)

    def __init__(self, tracer: Tracer) -> None:
        self._tracer = tracer

    def record(self, past_key_values: object, token_index: int) -> None:
        cache_bytes, num_layers = _compute_kv_cache_bytes(past_key_values)
        self._tracer.instant(
            name="kv_cache_grow",
            category="memory",
            scope=f"decode.token.{token_index}",
            token_index=token_index,
            metadata={
                "cache_size_bytes": cache_bytes,
                "num_layers": num_layers,
                "token_index": token_index,
            },
        )
