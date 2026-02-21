from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from argus.core.tracer import Tracer
    from argus.hooks.kvcache import KVCacheTracker


def trace_generate(
    model: object,
    input_ids: object,
    tracer: Tracer,
    max_new_tokens: int = 128,
    kv_tracker: KVCacheTracker | None = None,
) -> object:
    """Trace a model's greedy decode loop with token-level spans.

    Reimplements greedy decode — does not call model.generate().
    """
    import torch

    generated = input_ids.clone()
    past_key_values = None

    with tracer.span("prefill", category="phase", scope="prefill"):
        with tracer.span("forward_pass", category="compute", scope="prefill.forward"):
            outputs = model(input_ids, use_cache=True)
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
        if kv_tracker is not None:
            kv_tracker.record(past_key_values, token_index=0)

    with tracer.span("decode", category="phase", scope="decode"):
        for i in range(max_new_tokens):
            with tracer.span(
                "token_generate",
                category="token",
                scope=f"decode.token.{i}",
                token_index=i,
            ):
                next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token_id], dim=-1)

                eos_token_id = getattr(model.config, "eos_token_id", None)
                if eos_token_id is not None and next_token_id.item() == eos_token_id:
                    break

                if i == max_new_tokens - 1:
                    break

                with tracer.span(
                    "forward_pass",
                    category="compute",
                    scope=f"decode.token.{i}.forward",
                ):
                    outputs = model(
                        next_token_id,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    past_key_values = outputs.past_key_values
                    next_token_logits = outputs.logits[:, -1, :]

                if kv_tracker is not None:
                    kv_tracker.record(past_key_values, token_index=i + 1)

    return generated
