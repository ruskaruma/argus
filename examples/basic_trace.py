"""Basic Argus tracing — creates spans and exports to Chrome Trace Format."""

import argus


def main():
    tracer = argus.Tracer()

    with tracer.span("full_pipeline", category="phase", scope="pipeline"):
        with tracer.span("preprocessing", category="compute", scope="pipeline.preprocess"):
            data = list(range(1000))

        with tracer.span("inference", category="compute", scope="pipeline.inference"):
            result = sum(data)

        with tracer.span("postprocessing", category="compute", scope="pipeline.postprocess"):
            output = str(result)

    print(f"Collected {len(tracer.events)} events")
    argus.export_chrome(tracer, "basic_trace.json")
    print("Trace exported to basic_trace.json")
    print("Open https://ui.perfetto.dev/ and load the file to view")


if __name__ == "__main__":
    main()
