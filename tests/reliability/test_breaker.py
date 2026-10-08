from __future__ import annotations

import pytest

from keel.clock import ManualClock
from keel.errors import BackendTimeout, CircuitOpen, RateLimited, ValidationError
from keel.reliability.breaker import CircuitBreaker, CircuitState

# --- validation --------------------------------------------------------------------


def test_thresholds_must_be_sane() -> None:
    with pytest.raises(ValueError, match="failure_threshold"):
        CircuitBreaker(failure_threshold=0)
    with pytest.raises(ValueError, match="reset_timeout_s"):
        CircuitBreaker(reset_timeout_s=0)


# --- closed ------------------------------------------------------------------------


def test_starts_closed_and_admits_calls() -> None:
    breaker = CircuitBreaker(clock=ManualClock())
    assert breaker.state is CircuitState.CLOSED
    assert breaker.is_call_permitted
    breaker.call()


def test_success_resets_the_failure_count() -> None:
    breaker = CircuitBreaker(failure_threshold=3, clock=ManualClock())
    breaker.record_failure()
    breaker.record_failure()
    breaker.record_success()
    assert breaker.failures == 0
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state is CircuitState.CLOSED


# --- opening -----------------------------------------------------------------------


def test_opens_after_consecutive_failures() -> None:
    breaker = CircuitBreaker(failure_threshold=3, clock=ManualClock())
    for _ in range(3):
        breaker.record_failure()
    assert breaker.state is CircuitState.OPEN
    assert not breaker.is_call_permitted


def test_one_short_of_threshold_stays_closed() -> None:
    breaker = CircuitBreaker(failure_threshold=3, clock=ManualClock())
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.is_call_permitted


def test_call_is_rejected_while_open() -> None:
    breaker = CircuitBreaker(failure_threshold=1, clock=ManualClock())
    breaker.record_failure()
    with pytest.raises(CircuitOpen, match="circuit open"):
        breaker.call("primary")
    assert breaker.rejected == 1


def test_non_retryable_failure_does_not_trip_the_breaker() -> None:
    """A 400 means the backend answered. Counting it as an outage would let one
    client with a malformed request take down everyone else's traffic."""
    breaker = CircuitBreaker(failure_threshold=2, clock=ManualClock())
    for _ in range(10):
        breaker.record_failure(ValidationError("bad prompt"))
    assert breaker.state is CircuitState.CLOSED
    assert breaker.failures == 0


# --- half open ---------------------------------------------------------------------


def test_admits_a_probe_after_the_reset_timeout() -> None:
    clock = ManualClock()
    breaker = CircuitBreaker(failure_threshold=1, reset_timeout_s=30.0, clock=clock)
    breaker.record_failure()
    assert not breaker.is_call_permitted

    clock.advance(29.0)
    assert not breaker.is_call_permitted

    clock.advance(2.0)
    assert breaker.is_call_permitted
    assert breaker.state is CircuitState.HALF_OPEN


def test_only_one_probe_is_admitted() -> None:
    clock = ManualClock()
    breaker = CircuitBreaker(failure_threshold=1, reset_timeout_s=10.0, clock=clock)
    breaker.record_failure()
    clock.advance(11.0)
    assert breaker.is_call_permitted
    assert not breaker.is_call_permitted, "a second caller must not join the probe"


def test_successful_probe_closes_the_circuit() -> None:
    clock = ManualClock()
    breaker = CircuitBreaker(failure_threshold=1, reset_timeout_s=10.0, clock=clock)
    breaker.record_failure()
    clock.advance(11.0)
    assert breaker.is_call_permitted
    breaker.record_success()
    assert breaker.state is CircuitState.CLOSED
    assert breaker.is_call_permitted


def test_failed_probe_reopens_immediately() -> None:
    clock = ManualClock()
    breaker = CircuitBreaker(failure_threshold=5, reset_timeout_s=10.0, clock=clock)
    for _ in range(5):
        breaker.record_failure()
    clock.advance(11.0)
    assert breaker.is_call_permitted

    breaker.record_failure(BackendTimeout("still down"))
    assert breaker.state is CircuitState.OPEN
    assert not breaker.is_call_permitted


def test_reset_returns_to_closed() -> None:
    breaker = CircuitBreaker(failure_threshold=1, clock=ManualClock())
    breaker.record_failure()
    breaker.reset()
    assert breaker.state is CircuitState.CLOSED
    assert breaker.is_call_permitted


def test_rate_limited_counts_as_a_failure() -> None:
    breaker = CircuitBreaker(failure_threshold=2, clock=ManualClock())
    breaker.record_failure(RateLimited("429"))
    breaker.record_failure(RateLimited("429"))
    assert breaker.state is CircuitState.OPEN
