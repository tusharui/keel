from __future__ import annotations

import pytest

from keel.clock import ManualClock
from keel.policy.ratelimit import SlidingWindowLimiter, TenantRateLimiter, TokenBucket


@pytest.fixture
def clock() -> ManualClock:
    return ManualClock()


# --- token bucket ------------------------------------------------------------------


def test_bucket_rejects_non_positive_settings() -> None:
    with pytest.raises(ValueError, match="rate_per_s"):
        TokenBucket(rate_per_s=0, capacity=5.0)
    with pytest.raises(ValueError, match="capacity"):
        TokenBucket(rate_per_s=1.0, capacity=0)


def test_bucket_starts_full(clock: ManualClock) -> None:
    bucket = TokenBucket(rate_per_s=1.0, capacity=5.0, clock=clock)
    assert all(bucket.try_consume() for _ in range(5))
    assert not bucket.try_consume()


def test_bucket_refills_at_the_configured_rate(clock: ManualClock) -> None:
    bucket = TokenBucket(rate_per_s=2.0, capacity=5.0, clock=clock)
    for _ in range(5):
        bucket.try_consume()
    assert not bucket.try_consume()

    clock.advance(1.0)
    assert bucket.try_consume()
    assert bucket.try_consume()
    assert not bucket.try_consume()


def test_refill_is_capped_at_capacity(clock: ManualClock) -> None:
    bucket = TokenBucket(rate_per_s=10.0, capacity=3.0, clock=clock)
    bucket.try_consume()
    clock.advance(1_000.0)
    assert all(bucket.try_consume() for _ in range(3))
    assert not bucket.try_consume(), "idling must not bank more than the burst allowance"


def test_burst_is_allowed_up_to_capacity(clock: ManualClock) -> None:
    bucket = TokenBucket(rate_per_s=0.1, capacity=10.0, clock=clock)
    assert all(bucket.try_consume() for _ in range(10))


def test_multi_token_consume(clock: ManualClock) -> None:
    bucket = TokenBucket(rate_per_s=1.0, capacity=10.0, clock=clock)
    assert bucket.try_consume(8.0)
    assert not bucket.try_consume(5.0)
    assert bucket.try_consume(2.0)


def test_negative_consume_is_rejected(clock: ManualClock) -> None:
    bucket = TokenBucket(rate_per_s=1.0, capacity=10.0, clock=clock)
    with pytest.raises(ValueError, match="negative"):
        bucket.try_consume(-1.0)


def test_retry_after_is_zero_when_tokens_are_available(clock: ManualClock) -> None:
    bucket = TokenBucket(rate_per_s=1.0, capacity=5.0, clock=clock)
    assert bucket.retry_after() == 0.0


def test_retry_after_reflects_the_refill_rate(clock: ManualClock) -> None:
    bucket = TokenBucket(rate_per_s=2.0, capacity=2.0, clock=clock)
    bucket.try_consume(2.0)
    assert bucket.retry_after() == pytest.approx(0.5)
    assert bucket.retry_after(4.0) == pytest.approx(2.0)


def test_time_going_backwards_does_not_create_tokens() -> None:
    class SettableClock:
        def __init__(self) -> None:
            self.t = 0.0

        def now(self) -> float:
            return self.t

        def advance(self, seconds: float) -> None:
            self.t += seconds

    clock = SettableClock()
    bucket = TokenBucket(rate_per_s=1.0, capacity=5.0, clock=clock)  # type: ignore[arg-type]
    bucket.try_consume(5.0)

    clock.t = -100.0
    assert not bucket.try_consume(), "a backwards clock must not mint tokens"


# --- sliding window ----------------------------------------------------------------


def test_window_caps_events_in_the_interval(clock: ManualClock) -> None:
    limiter = SlidingWindowLimiter(limit=3, window_s=10.0, clock=clock)
    assert all(limiter.try_consume() for _ in range(3))
    assert not limiter.try_consume()


def test_window_slides(clock: ManualClock) -> None:
    limiter = SlidingWindowLimiter(limit=2, window_s=5.0, clock=clock)
    assert limiter.try_consume()
    assert limiter.try_consume()
    assert not limiter.try_consume()

    clock.advance(5.0)
    assert limiter.try_consume()


def test_window_has_no_burst_allowance(clock: ManualClock) -> None:
    limiter = SlidingWindowLimiter(limit=2, window_s=100.0, clock=clock)
    assert limiter.try_consume()
    assert limiter.try_consume()
    assert not limiter.try_consume(), "the strict window refuses the burst"


def test_used_reflects_recent_events(clock: ManualClock) -> None:
    limiter = SlidingWindowLimiter(limit=5, window_s=10.0, clock=clock)
    for _ in range(3):
        limiter.try_consume()
    assert limiter.used() == 3
    clock.advance(11.0)
    assert limiter.used() == 0


def test_window_retry_after(clock: ManualClock) -> None:
    limiter = SlidingWindowLimiter(limit=1, window_s=4.0, clock=clock)
    assert limiter.try_consume()
    assert not limiter.try_consume()
    assert limiter.retry_after() == pytest.approx(4.0)
    clock.advance(4.0)
    assert limiter.retry_after() == 0.0


def test_window_validates_settings(clock: ManualClock) -> None:
    with pytest.raises(ValueError, match="limit"):
        SlidingWindowLimiter(limit=0, window_s=1.0)
    with pytest.raises(ValueError, match="window_s"):
        SlidingWindowLimiter(limit=1, window_s=0)


# --- per tenant --------------------------------------------------------------------


def test_tenants_are_limited_independently(clock: ManualClock) -> None:
    limiter = TenantRateLimiter(rate_per_s=1.0, capacity=2.0, clock=clock)
    assert limiter.try_consume("a")
    assert limiter.try_consume("a")
    assert not limiter.try_consume("a")
    assert limiter.try_consume("b"), "one tenant must not exhaust another's budget"


def test_buckets_are_created_on_first_use(clock: ManualClock) -> None:
    limiter = TenantRateLimiter(rate_per_s=1.0, capacity=1.0, clock=clock)
    assert limiter.tenants == set()
    limiter.try_consume("a")
    assert limiter.tenants == {"a"}


def test_same_bucket_is_reused_for_a_tenant(clock: ManualClock) -> None:
    limiter = TenantRateLimiter(rate_per_s=1.0, capacity=3.0, clock=clock)
    assert limiter.bucket_for("a") is limiter.bucket_for("a")
