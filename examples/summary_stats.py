"""Example: using argus.summarize() to get aggregate timing stats."""

from __future__ import annotations

import time

import argus


def main() -> None:
    tracer = argus.Tracer()

    with tracer.span("full_pipeline", category="phase", scope="pipeline"):
        with tracer.span("preprocessing", category="compute", scope="pipeline.preprocess"):
            time.sleep(0.001)

        with tracer.span("inference", category="phase", scope="pipeline.inference"):
            for i in range(10):
                with tracer.span(
                    "token_generate",
                    category="token",
                    scope=f"decode.token.{i}",
                    token_index=i,
                ):
                    time.sleep(0.0005)
                    with tracer.span(
                        "forward_pass",
                        category="compute",
                        scope=f"decode.token.{i}.forward",
                        token_index=i,
                    ):
                        time.sleep(0.001)

        with tracer.span("postprocessing", category="compute", scope="pipeline.postprocess"):
            time.sleep(0.001)

    # Summarize by span name
    summary = argus.summarize(tracer)
    print(argus.format_summary(summary))
    print()

    # Summarize by category
    by_cat = argus.summarize(tracer, group_by="category")
    print(argus.format_summary(by_cat))
    print()

    # Summarize by token index (only token-indexed events)
    token_events = tracer.get_events(category="token")
    by_token = argus.summarize(token_events, group_by="token_index")
    print(argus.format_summary(by_token))


if __name__ == "__main__":
    main()
