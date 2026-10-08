from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TypeVar

from keel.clock import Clock, default_clock
from keel.errors import CircuitOpen, KeelError
from keel.reliability.breaker import CircuitBreaker

T = TypeVar("T")


@dataclass(slots=True)
class Provider:
    name: str
    call: Callable[[], Awaitable[T]]
    weight: float = 1.0


@dataclass(slots=True)
class ProviderHealth:
    attempts: int = 0
    failures: int = 0
    consecutive_failures: int = 0
    breaker: CircuitBreaker = field(default_factory=CircuitBreaker)

    @property
    def available(self) -> bool:
        return self.breaker.is_call_permitted

    @property
    def reliability(self) -> float:
        """Fraction of attempts that succeeded. Used to rank providers."""
        return 1.0 - (self.failures / self.attempts) if self.attempts else 1.0


@dataclass(slots=True)
class FallbackChain:
    """Tries providers in order until one succeeds.

    Ordering is by observed reliability rather than by configuration, so a
    provider that starts failing is skipped ahead of a healthy one that happens
    to be listed later. Breakers sit in front of each provider so a dead one is
    skipped without paying a timeout to discover it again.
    """

    providers: list[Provider]
    clock: Clock = field(default_factory=default_clock)
    health: dict[str, ProviderHealth] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for provider in self.providers:
            self.health.setdefault(provider.name, ProviderHealth())

    def _ranked(self) -> list[Provider]:
        return sorted(
            self.providers,
            key=lambda p: (
                not self.health[p.name].available,
                -self.health[p.name].reliability,
                p.name,
            ),
        )

    def skipped(self) -> list[str]:
        return [p.name for p in self.providers if not self.health[p.name].available]

    async def run(self) -> tuple[T, str]:
        attempted = 0
        last_error: KeelError | None = None

        for provider in self._ranked():
            entry = self.health[provider.name]
            if not entry.breaker.is_call_permitted:
                continue

            attempted += 1
            entry.attempts += 1
            try:
                result: T = await provider.call()
            except KeelError as error:
                entry.failures += 1
                entry.consecutive_failures += 1
                entry.breaker.record_failure(error)
                if not error.retryable:
                    # The request itself is bad. Another provider would return the
                    # same rejection, and failing over turns one 400 into two.
                    raise
                last_error = error
                continue

            entry.consecutive_failures = 0
            entry.breaker.record_success()
            return result, provider.name

        if last_error is not None:
            raise last_error
        raise CircuitOpen(f"all {len(self.providers)} providers are unavailable")
