"""Nested Argus tracing — demonstrates parent-child span relationships."""

import argus


def main():
    tracer = argus.Tracer()

    with tracer.span("decode", category="phase", scope="decode"):
        for i in range(5):
            with tracer.span(
                "token_generate",
                category="token",
                scope=f"decode.token.{i}",
                token_index=i,
            ) as ctx:
                ctx.add_metadata("token_id", i * 100)

                with tracer.span(
                    "forward_pass",
                    category="compute",
                    scope=f"decode.token.{i}.forward",
                ):
                    _ = sum(range(1000))

                with tracer.span(
                    "sample",
                    category="system",
                    scope=f"decode.token.{i}.sample",
                ):
                    _ = max(range(100))

    print(f"Collected {len(tracer.events)} events")
    for e in tracer.events:
        indent = "  " if e.parent_id else ""
        indent += "  " if e.category in ("compute", "system") else ""
        print(f"{indent}{e.name} [{e.category}] {e.duration_ns}ns parent={e.parent_id}")

    argus.export_chrome(tracer, "nested_trace.json")
    print("\nTrace exported to nested_trace.json")


if __name__ == "__main__":
    main()
