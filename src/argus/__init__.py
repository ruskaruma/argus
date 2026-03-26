"""Argus — lightweight profiling and tracing for Python ML workloads."""

from __future__ import annotations

from typing import IO, TYPE_CHECKING

from argus.core.clock import monotonic_ns
from argus.core.events import TraceEvent
from argus.core.tracer import Tracer
from argus.exporters.chrome import export_chrome_trace
from argus.hooks.layers import LayerHookHandle, register_layer_hooks

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "LayerHookHandle",
    "TraceEvent",
    "Tracer",
    "export_chrome",
    "monotonic_ns",
    "register_layer_hooks",
]

__version__ = "0.1.0"


def export_chrome(tracer: Tracer, dest: str | Path | IO[str]) -> None:
    export_chrome_trace(tracer.events, dest)
