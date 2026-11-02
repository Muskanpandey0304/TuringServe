"""Small timing helpers."""

from __future__ import annotations

import time
from collections import defaultdict


class StepTimer:
    """Accumulates wall-clock seconds per named phase for one engine step."""

    def __init__(self) -> None:
        self.t: dict[str, float] = defaultdict(float)
        self._start: dict[str, float] = {}

    def start(self, name: str) -> None:
        self._start[name] = time.perf_counter()

    def stop(self, name: str) -> None:
        self.t[name] += time.perf_counter() - self._start.pop(name)

    def reset(self) -> None:
        self.t.clear()
        self._start.clear()


def percentile(values: list[float], q: float) -> float:
    """q in [0, 100]; linear interpolation; nan for empty input."""
    if not values:
        return float("nan")
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)
