from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from keel.clock import Clock, default_clock


@dataclass(slots=True)
class TokenBucket:
    """Steady rate with a bounded burst.

    Lets a client spend up to ``capacity`` immediately and then settles to
    ``rate_per_s``. A sliding window would refuse the whole burst, which pushes
    retries into a synchronised wave at the next boundary and makes the
    limiters own behaviour the source of the load spike.
    """

    rate_per_s: float
    capacity: float
    clock: Clock = field(default_factory=default_clock)
    tokens: float = field(init=False)
    _updated: float = field(init=False)

    def __post_init__(self) -> None:
        if self.rate_per_s <= 0:
            raise ValueError("rate_per_s must be positive")
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        self.tokens = self.capacity
        self._updated = self.clock.now()

    def _refill(self) -> None:
        now = self.clock.now()
        elapsed = max(0.0, now - self._updated)
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate_per_s)
        self._updated = now

    def try_consume(self, tokens: float = 1.0) -> bool:
        if tokens < 0:
            raise ValueError("cannot consume a negative amount")
        self._refill()
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    def retry_after(self, tokens: float = 1.0) -> float:
        """Seconds until ``tokens`` are available. Zero if they already are."""
        self._refill()
        if self.tokens >= tokens:
            return 0.0
        return (tokens - self.tokens) / self.rate_per_s


@dataclass(slots=True)
class SlidingWindowLimiter:
    """Strict cap on events in any window.

    Unlike the bucket this gives no burst allowance, so it suits limits where
    the concern is a sustained rate rather than a spike.
    """

    limit: int
    window_s: float
    clock: Clock = field(default_factory=default_clock)
    _events: deque[float] = field(default_factory=deque, init=False)

    def __post_init__(self) -> None:
        if self.limit < 1:
            raise ValueError("limit must be at least 1")
        if self.window_s <= 0:
            raise ValueError("window_s must be positive")

    def try_consume(self) -> bool:
        now = self.clock.now()
        horizon = now - self.window_s
        while self._events and self._events[0] <= horizon:
            self._events.popleft()

        if len(self._events) >= self.limit:
            return False
        self._events.append(now)
        return True

    def used(self) -> int:
        now = self.clock.now()
        horizon = now - self.window_s
        return sum(1 for t in self._events if t > horizon)

    def retry_after(self) -> float:
        now = self.clock.now()
        horizon = now - self.window_s
        while self._events and self._events[0] <= horizon:
            self._events.popleft()
        if not self._events:
            return 0.0
        return max(0.0, (self._events[0] + self.window_s) - now)


@dataclass(slots=True)
class TenantRateLimiter:
    """Per-tenant buckets, created on first use."""

    rate_per_s: float
    capacity: float
    clock: Clock = field(default_factory=default_clock)
    _buckets: dict[str, TokenBucket] = field(default_factory=dict)

    def bucket_for(self, tenant_id: str) -> TokenBucket:
        bucket = self._buckets.get(tenant_id)
        if bucket is None:
            bucket = TokenBucket(self.rate_per_s, self.capacity, clock=self.clock)
            self._buckets[tenant_id] = bucket
        return bucket

    def try_consume(self, tenant_id: str, tokens: float = 1.0) -> bool:
        return self.bucket_for(tenant_id).try_consume(tokens)

    def retry_after(self, tenant_id: str, tokens: float = 1.0) -> float:
        return self.bucket_for(tenant_id).retry_after(tokens)

    @property
    def tenants(self) -> set[str]:
        return set(self._buckets)
