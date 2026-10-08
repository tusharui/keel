from __future__ import annotations

import time
from typing import Protocol, runtime_checkable


@runtime_checkable
class Clock(Protocol):
    """Time source. Everything that needs a timestamp takes one of these.

    Injecting this rather than calling ``time.time`` directly is what makes the
    scheduler testable: scheduling bugs are ordering bugs, and ordering is far
    easier to assert on when the clock is under test control.
    """

    def now(self) -> float:
        """Monotonic seconds. Only differences between two values are meaningful."""
        ...

    def advance(self, seconds: float) -> None:
        """Move time forward by the given amount."""
        ...


class SystemClock:
    __slots__ = ()

    def now(self) -> float:
        return time.perf_counter()

    def advance(self, seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)


class ManualClock:
    """Virtual clock that only moves when told to.

    The simulated engine charges simulated cost per forward pass, so the
    scheduler advances this explicitly instead of sleeping. That makes an entire
    10k-request benchmark run in milliseconds and produce byte-identical
    numbers on every run, which is what you want when a regression needs to be
    traced to a specific commit.
    """

    __slots__ = ("_now",)

    def __init__(self, start: float = 0.0) -> None:
        self._now = start

    def now(self) -> float:
        return self._now

    def advance(self, seconds: float) -> float:
        if seconds < 0:
            raise ValueError("cannot advance a clock backwards")
        self._now += seconds
        return self._now


_default = SystemClock()


def default_clock() -> Clock:
    return _default
