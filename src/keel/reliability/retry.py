from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Literal, TypeVar

from keel.clock import Clock, default_clock
from keel.errors import KeelError

T = TypeVar("T")

JitterStrategy = Literal["none", "equal", "full"]


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Exponential backoff with an explicit jitter strategy.

    Jitter is a parameter rather than a detail because the three strategies have
    measurably different behaviour under load. Full jitter draws uniformly from
    zero to the ceiling and maximises spread; no jitter collapses every client
    onto the same retry instants, which is how a recovering backend gets knocked
    over again a second later.
    """

    max_attempts: int = 3
    base_delay_s: float = 0.05
    max_delay_s: float = 2.0
    jitter: JitterStrategy = "full"
    deadline_s: float | None = None

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.max_delay_s < self.base_delay_s:
            raise ValueError("max_delay_s must be >= base_delay_s")
        if self.deadline_s is not None and self.deadline_s < 0:
            raise ValueError("deadline_s cannot be negative")

    def ceiling_for(self, attempt: int) -> float:
        """Un-jittered delay before ``attempt`` is retried. ``attempt`` is 1-based."""
        if attempt < 1:
            raise ValueError("attempt is 1-based")
        return min(self.max_delay_s, self.base_delay_s * 2.0 ** (attempt - 1))

    def delay_for(self, attempt: int, *, rand: Callable[[], float]) -> float:
        ceiling = self.ceiling_for(attempt)
        if self.jitter == "none":
            return ceiling
        if self.jitter == "equal":
            half = ceiling / 2
            return half + rand() * half
        return rand() * ceiling

    def should_retry(self, attempt: int, error: BaseException, elapsed_s: float) -> bool:
        if attempt >= self.max_attempts:
            return False
        if self.deadline_s is not None and elapsed_s >= self.deadline_s:
            return False
        return isinstance(error, KeelError) and error.retryable


@dataclass(slots=True)
class Retryer:
    """Applies a policy to an operation, recording what it did.

    The clock, sleep and randomness are all injected. That is what lets a test
    assert the exact delay sequence and the spread of the jitter distribution
    instead of sleeping through it.
    """

    policy: RetryPolicy
    clock: Clock = field(default_factory=default_clock)
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep
    rand: Callable[[], float] = random.random
    attempts: list[int] = field(default_factory=list)
    delays: list[float] = field(default_factory=list)

    async def run(self, operation: Callable[[], Awaitable[T]]) -> T:
        started = self.clock.now()
        attempt = 0
        while True:
            attempt += 1
            try:
                result = await operation()
            except Exception as error:
                self.attempts.append(attempt)
                if not self.policy.should_retry(attempt, error, self.clock.now() - started):
                    raise
                delay = self.policy.delay_for(attempt, rand=self.rand)
                self.delays.append(delay)
                self.clock.advance(delay)
                await self.sleep(delay)
                continue
            self.attempts.append(attempt)
            return result


async def retry_async[T](
    operation: Callable[[], Awaitable[T]],
    policy: RetryPolicy | None = None,
    **kwargs: object,
) -> T:
    retryer = Retryer(policy or RetryPolicy(), **kwargs)  # type: ignore[arg-type]
    return await retryer.run(operation)
