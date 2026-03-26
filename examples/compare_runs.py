"""Compare two inference runs — demonstrates Argus trace diffing.

Creates two synthetic decode traces with different per-token timings,
diffs them, and exports a Chrome Trace with diff annotations.
"""

import time

import argus


def simulate_decode(tracer: argus.Tracer, token_delays_us: list[int]) -> None:
    """Simulate a decode loop where each token takes a configurable amount of time."""
    with tracer.span("decode", category="phase", scope="decode"):
        for i, delay_us in enumerate(token_delays_us):
            with tracer.span(
                "token_generate",
                category="token",
                scope=f"decode.token.{i}",
                token_index=i,
            ):
                with tracer.span(
                    "forward_pass",
                    category="compute",
                    scope=f"decode.token.{i}.forward",
                    token_index=i,
                ):
                    time.sleep(delay_us / 1_000_000)

                with tracer.span(
                    "sample",
                    category="system",
                    scope=f"decode.token.{i}.sample",
                    token_index=i,
                ):
                    time.sleep(10 / 1_000_000)


def main() -> None:
    # Run A: baseline — all tokens ~200μs
    tracer_a = argus.Tracer()
    simulate_decode(tracer_a, [200, 200, 200, 200, 200, 200, 200, 200])

    # Run B: token 3 and 6 are much slower (simulating cache miss, etc.)
    tracer_b = argus.Tracer()
    simulate_decode(tracer_b, [200, 200, 200, 800, 200, 200, 1000, 200])

    # Diff the two runs
    result = argus.diff_traces(tracer_a, tracer_b, outlier_threshold=2.0)

    # Print summary
    print("=== Trace Diff Summary ===")
    print(f"Matched spans:  {len(result.matched)}")
    print(f"Added spans:    {len(result.added)}")
    print(f"Removed spans:  {len(result.removed)}")
    print(f"Outlier tokens: {len(result.outlier_tokens)}")

    print("\n--- Per-span deltas (matched) ---")
    for d in result.matched:
        sign = "+" if d.delta_ns >= 0 else ""
        print(
            f"  {d.scope:40s}  "
            f"A={d.duration_a_ns / 1000:.0f}μs  "
            f"B={d.duration_b_ns / 1000:.0f}μs  "
            f"Δ={sign}{d.delta_ns / 1000:.0f}μs ({sign}{d.delta_pct:.1f}%)"
        )

    if result.outlier_tokens:
        print("\n--- Outlier tokens ---")
        for o in result.outlier_tokens:
            print(
                f"  token {o.token_index}: "
                f"A={o.total_a_ns / 1000:.0f}μs → B={o.total_b_ns / 1000:.0f}μs "
                f"({o.ratio:.1f}x)"
            )

    # Export Chrome Trace with diff overlay
    argus.export_diff_chrome(tracer_a, tracer_b, "compare_runs_diff.json")
    print("\nDiff trace exported to compare_runs_diff.json")
    print("Open in https://ui.perfetto.dev to visualize side-by-side.")


if __name__ == "__main__":
    main()
