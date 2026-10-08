from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from keel.clock import Clock, default_clock
from keel.errors import CircuitOpen, KeelError


class CircuitState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass(slots=True)
class CircuitBreaker:
    """Fails fast once a backend stops working.

    Retrying an already-dead backend spends a connection, a timeout and a retry
    slot to arrive at a conclusion the breaker already reached. The window this
    buys matters under load, when the failures correlate: every client retrying
    a dead backend is what keeps it dead.

    Half-open admits a single probe. Admitting a probe per caller would
    reproduce the thundering herd the breaker exists to stop.
    """

    failure_threshold: int = 5
    reset_timeout_s: float = 30.0
    clock: Clock = field(default_factory=default_clock)
    state: CircuitState = CircuitState.CLOSED
    failures: int = 0
    successes: int = 0
    opened_at: float | None = None
    rejected: int = 0

    def __post_init__(self) -> None:
        if self.failure_threshold < 1:
            raise ValueError("failure_threshold must be at least 1")
        if self.reset_timeout_s <= 0:
            raise ValueError("reset_timeout_s must be positive")

    @property
    def is_call_permitted(self) -> bool:
        if self.state is CircuitState.CLOSED:
            return True
        if self.state is CircuitState.HALF_OPEN:
            return False
        if self.opened_at is None:
            return False
        if self.clock.now() - self.opened_at >= self.reset_timeout_s:
            self.state = CircuitState.HALF_OPEN
            return True
        return False

    def record_success(self) -> None:
        self.successes += 1
        self.failures = 0
        self.state = CircuitState.CLOSED
        self.opened_at = None

    def record_failure(self, error: BaseException | None = None) -> None:
        # A validation failure means the backend answered. Treating it as an
        # outage would trip the breaker on a malformed request.
        if error is not None and isinstance(error, KeelError) and not error.retryable:
            self.record_success()
            return

        self.failures += 1
        if self.state is CircuitState.HALF_OPEN or self.failures >= self.failure_threshold:
            self.state = CircuitState.OPEN
            self.opened_at = self.clock.now()

    def call(self, operation: str = "backend") -> None:
        if not self.is_call_permitted:
            self.rejected += 1
            raise CircuitOpen(f"circuit open for {operation}; {self.failures} consecutive failures")

    def reset(self) -> None:
        self.state = CircuitState.CLOSED
        self.failures = 0
        self.opened_at = None

    @property
    def failure_rate(self) -> float:
        total = self.failures + self.successes
        return self.failures / total if total else 0.0
