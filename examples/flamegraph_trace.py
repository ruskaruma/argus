"""Flamegraph export example — produces folded stacks and speedscope JSON."""

import time

import argus
from argus.exporters.flamegraph import export_flamegraph


def main() -> None:
    tracer = argus.Tracer()

    with tracer.span("decode", category="phase", scope="decode"):
        for tok in range(5):
            with tracer.span(
                "token_generate",
                category="token",
                scope=f"decode.token.{tok}",
                token_index=tok,
            ):
                with tracer.span(
                    "attention",
                    category="compute",
                    scope=f"decode.token.{tok}.attention",
                    token_index=tok,
                ):
                    time.sleep(0.001 * (tok + 1))

                with tracer.span(
                    "ffn",
                    category="compute",
                    scope=f"decode.token.{tok}.ffn",
                    token_index=tok,
                ):
                    time.sleep(0.0005)

                with tracer.span(
                    "sample",
                    category="system",
                    scope=f"decode.token.{tok}.sample",
                    token_index=tok,
                ):
                    time.sleep(0.0002)

    print(f"Collected {len(tracer.events)} events")

    # Aggregated folded stacks (flamegraph.pl compatible)
    export_flamegraph(tracer, "flamegraph_aggregated.txt", mode="aggregated")
    print("Wrote flamegraph_aggregated.txt (pipe to flamegraph.pl)")

    # Per-token folded stacks
    export_flamegraph(tracer, "flamegraph_per_token.txt", mode="per_token")
    print("Wrote flamegraph_per_token.txt")

    # Speedscope JSON (open at https://speedscope.app)
    export_flamegraph(tracer, "flamegraph_aggregated.json", mode="aggregated")
    print("Wrote flamegraph_aggregated.json (open at speedscope.app)")

    # Per-token speedscope JSON
    export_flamegraph(tracer, "flamegraph_per_token.json", mode="per_token")
    print("Wrote flamegraph_per_token.json (open at speedscope.app)")


if __name__ == "__main__":
    main()
