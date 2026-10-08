from __future__ import annotations

import random

import pytest

from keel.clock import ManualClock
from keel.errors import BackendTimeout, ContextLengthExceeded, RateLimited, ValidationError
from keel.reliability.retry import Retryer, RetryPolicy


async def no_sleep(_delay: float) -> None:
    return None


def constant(value: float):
    return lambda: value


def test_attempts_are_one_based() -> None:
    policy = RetryPolicy(base_delay_s=0.1, max_delay_s=1.0, jitter="none")
    assert policy.ceiling_for(1) == pytest.approx(0.1)
    assert policy.ceiling_for(2) == pytest.approx(0.2)
    assert policy.ceiling_for(3) == pytest.approx(0.4)


def test_ceiling_is_capped() -> None:
    policy = RetryPolicy(base_delay_s=1.0, max_delay_s=2.0, jitter="none")
    assert policy.ceiling_for(10) == 2.0


def test_rejects_non_positive_attempt() -> None:
    with pytest.raises(ValueError, match="1-based"):
        RetryPolicy().ceiling_for(0)


def test_policy_validates_its_bounds() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        RetryPolicy(max_attempts=0)
    with pytest.raises(ValueError, match="max_delay_s"):
        RetryPolicy(base_delay_s=1.0, max_delay_s=0.5)
    with pytest.raises(ValueError, match="deadline"):
        RetryPolicy(deadline_s=-1.0)


# --- jitter -----------------------------------------------------------------------


def test_no_jitter_returns_the_ceiling() -> None:
    policy = RetryPolicy(base_delay_s=0.1, jitter="none")
    assert policy.delay_for(3, rand=constant(0.9)) == pytest.approx(0.4)


def test_equal_jitter_never_drops_below_half_the_ceiling() -> None:
    policy = RetryPolicy(base_delay_s=0.1, jitter="equal")
    assert policy.delay_for(3, rand=constant(0.0)) == pytest.approx(0.2)
    assert policy.delay_for(3, rand=constant(1.0)) == pytest.approx(0.4)


def test_full_jitter_spans_zero_to_the_ceiling() -> None:
    policy = RetryPolicy(base_delay_s=0.1, jitter="full")
    assert policy.delay_for(3, rand=constant(0.0)) == 0.0
    assert policy.delay_for(3, rand=constant(1.0)) == pytest.approx(0.4)


def test_full_jitter_actually_spreads_retries() -> None:
    """Without spread every client retries on the same instant, which is how a
    recovering backend gets knocked over again."""
    policy = RetryPolicy(base_delay_s=0.1, jitter="full")
    samples = [policy.delay_for(3, rand=random.Random(seed).random) for seed in range(400)]
    assert min(samples) < 0.1
    assert max(samples) > 0.3
    assert len(set(round(s, 4) for s in samples)) > 300


def test_jitter_cannot_exceed_the_ceiling() -> None:
    policy = RetryPolicy(base_delay_s=0.1, max_delay_s=1.0, jitter="full")
    for attempt in range(1, 8):
        for _ in range(50):
            assert policy.delay_for(attempt, rand=random.random) <= policy.ceiling_for(attempt)


# --- retry decisions ---------------------------------------------------------------


def test_retryable_errors_are_retried() -> None:
    assert RetryPolicy().should_retry(1, BackendTimeout("t"), 0.0)


def test_non_retryable_errors_are_not() -> None:
    assert not RetryPolicy().should_retry(1, ValidationError("bad"), 0.0)
    assert not RetryPolicy().should_retry(1, ContextLengthExceeded("long"), 0.0)


def test_non_keel_errors_are_not_retried() -> None:
    assert not RetryPolicy().should_retry(1, ValueError("who knows"), 0.0)


def test_attempt_cap_stops_retrying() -> None:
    policy = RetryPolicy(max_attempts=2)
    assert policy.should_retry(1, RateLimited("429"), 0.0)
    assert not policy.should_retry(2, RateLimited("429"), 0.0)


def test_deadline_stops_retrying() -> None:
    policy = RetryPolicy(max_attempts=10, deadline_s=1.0)
    assert policy.should_retry(1, RateLimited("429"), 0.5)
    assert not policy.should_retry(1, RateLimited("429"), 1.0)


# --- retryer -----------------------------------------------------------------------


async def test_success_on_first_attempt_does_not_sleep() -> None:
    async def ok() -> str:
        return "ok"

    retryer = Retryer(RetryPolicy(), clock=ManualClock(), sleep=no_sleep)
    assert await retryer.run(ok) == "ok"
    assert retryer.attempts == [1]
    assert retryer.delays == []


async def test_retries_until_success() -> None:
    calls = {"n": 0}

    async def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise BackendTimeout("boom")
        return "ok"

    retryer = Retryer(RetryPolicy(max_attempts=5), clock=ManualClock(), sleep=no_sleep)
    assert await retryer.run(flaky) == "ok"
    assert retryer.attempts == [1, 2, 3]
    assert len(retryer.delays) == 2


async def test_non_retryable_propagates_immediately() -> None:
    calls = {"n": 0}

    async def bad() -> str:
        calls["n"] += 1
        raise ValidationError("nope")

    retryer = Retryer(RetryPolicy(max_attempts=5), clock=ManualClock(), sleep=no_sleep)
    with pytest.raises(ValidationError):
        await retryer.run(bad)
    assert calls["n"] == 1
    assert retryer.delays == []


async def test_attempts_are_capped() -> None:
    calls = {"n": 0}

    async def always_fails() -> str:
        calls["n"] += 1
        raise BackendTimeout("boom")

    retryer = Retryer(
        RetryPolicy(max_attempts=3), clock=ManualClock(), sleep=no_sleep, rand=constant(0.5)
    )
    with pytest.raises(BackendTimeout):
        await retryer.run(always_fails)
    assert calls["n"] == 3
    assert retryer.attempts == [1, 2, 3]


async def test_delays_are_recorded_and_clock_advances() -> None:
    clock = ManualClock()
    calls = {"n": 0}

    async def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 2:
            raise BackendTimeout("boom")
        return "ok"

    retryer = Retryer(RetryPolicy(jitter="none"), clock=clock, sleep=no_sleep, rand=constant(1.0))
    await retryer.run(flaky)
    assert retryer.delays == [pytest.approx(0.05)]
    assert clock.now() == pytest.approx(0.05)
