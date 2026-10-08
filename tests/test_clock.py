from __future__ import annotations

import pytest

from keel.clock import ManualClock, SystemClock


def test_manual_clock_only_moves_when_told() -> None:
    clock = ManualClock()
    assert clock.now() == 0.0

    clock.advance(0.5)
    assert clock.now() == 0.5

    clock.advance(0.25)
    assert clock.now() == 0.75


def test_manual_clock_starts_at_offset() -> None:
    assert ManualClock(start=100.0).now() == 100.0


def test_manual_clock_rejects_backwards_time() -> None:
    clock = ManualClock()
    with pytest.raises(ValueError, match="backwards"):
        clock.advance(-1.0)


def test_system_clock_is_monotonic() -> None:
    clock = SystemClock()
    samples = [clock.now() for _ in range(50)]
    assert samples == sorted(samples)
    assert clock.now() >= samples[0]
