from __future__ import annotations

from argus.core.clock import monotonic_ns


def test_returns_int():
    assert isinstance(monotonic_ns(), int)


def test_monotonicity():
    values = [monotonic_ns() for _ in range(100)]
    for i in range(1, len(values)):
        assert values[i] >= values[i - 1]


def test_resolution():
    a = monotonic_ns()
    b = monotonic_ns()
    assert b - a >= 0


def test_rapid_calls_no_exceptions():
    for _ in range(1000):
        monotonic_ns()


def test_positive_value():
    assert monotonic_ns() > 0
