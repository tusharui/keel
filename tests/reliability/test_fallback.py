from __future__ import annotations

import pytest

from keel.clock import ManualClock
from keel.errors import BackendTimeout, CircuitOpen, ValidationError
from keel.reliability.breaker import CircuitState
from keel.reliability.fallback import FallbackChain, Provider


def make(name: str, result: str = "ok", failures: int = 0):
    state = {"calls": 0, "remaining": failures}

    async def call() -> str:
        state["calls"] += 1
        if state["remaining"] > 0:
            state["remaining"] -= 1
            raise BackendTimeout(f"{name} down")
        return result

    return Provider(name=name, call=call), state


async def test_uses_the_first_healthy_provider() -> None:
    primary, primary_state = make("primary", result="from-primary")
    secondary, _ = make("secondary")
    chain = FallbackChain([primary, secondary], clock=ManualClock())

    result, used = await chain.run()
    assert result == "from-primary"
    assert used == "primary"
    assert secondary is not None
    assert primary_state["calls"] == 1


async def test_falls_back_to_the_next_provider() -> None:
    primary, _ = make("primary", failures=99)
    secondary, secondary_state = make("secondary", result="from-secondary")
    chain = FallbackChain([primary, secondary], clock=ManualClock())

    result, used = await chain.run()
    assert result == "from-secondary"
    assert used == "secondary"
    assert secondary_state["calls"] == 1


async def test_raises_when_every_provider_fails() -> None:
    first, _ = make("first", failures=99)
    second, _ = make("second", failures=99)
    chain = FallbackChain([first, second], clock=ManualClock())

    with pytest.raises(BackendTimeout):
        await chain.run()


async def test_raises_when_every_provider_is_open() -> None:
    first, _ = make("first", failures=99)
    second, _ = make("second", failures=99)
    chain = FallbackChain([first, second], clock=ManualClock())
    for name in ("first", "second"):
        chain.health[name].breaker.failure_threshold = 1
        chain.health[name].breaker.record_failure()

    assert set(chain.skipped()) == {"first", "second"}
    with pytest.raises(CircuitOpen):
        await chain.run()


async def test_open_providers_are_skipped_without_calling_them() -> None:
    dead, dead_state = make("dead", failures=99)
    healthy, healthy_state = make("healthy", result="ok")
    chain = FallbackChain([dead, healthy], clock=ManualClock())

    chain.health["dead"].breaker.failure_threshold = 1
    chain.health["dead"].breaker.record_failure()
    chain.health["dead"].breaker.clock.advance(1.0)
    chain.health["dead"].breaker.reset()
    chain.health["dead"].breaker.failure_threshold = 1
    chain.health["dead"].breaker.record_failure()

    result, used = await chain.run()
    assert result == "ok"
    assert used == "healthy"
    assert dead_state["calls"] == 0
    assert healthy_state["calls"] == 1


async def test_ordering_adapts_to_observed_reliability() -> None:
    """A provider that fails once is ranked behind one with a clean record, even
    though it is listed first."""
    flaky, _ = make("flaky", result="flaky-ok", failures=1)
    steady, _ = make("steady", result="steady-ok")
    chain = FallbackChain([flaky, steady], clock=ManualClock())

    chain.health["flaky"].attempts = 5
    chain.health["flaky"].failures = 4

    _result, used = await chain.run()
    assert used == "steady"


async def test_validation_errors_do_not_fail_over() -> None:
    """A bad request will be bad everywhere. Retrying it on another provider just
    turns one 400 into two."""

    async def bad() -> str:
        raise ValidationError("nope")

    second, second_state = make("second")
    chain = FallbackChain([Provider(name="first", call=bad), second], clock=ManualClock())

    with pytest.raises(ValidationError):
        await chain.run()
    assert second_state["calls"] == 0


async def test_reliability_starts_at_one() -> None:
    chain = FallbackChain([], clock=ManualClock())
    assert chain.health == {}


async def test_rate_limited_triggers_failover() -> None:
    primary, _ = make("primary", failures=99)
    secondary, _ = make("secondary", result="secondary")
    chain = FallbackChain([primary, secondary], clock=ManualClock())

    chain.health["primary"].breaker.failure_threshold = 1
    result, used = await chain.run()
    assert result == "secondary"
    assert used == "secondary"
    assert chain.health["primary"].breaker.state is CircuitState.OPEN


async def test_circuit_open_is_reported_when_all_providers_trip() -> None:
    chain = FallbackChain([make("only")[0]], clock=ManualClock())
    chain.health["only"].breaker.failure_threshold = 1
    chain.health["only"].breaker.record_failure()

    with pytest.raises(CircuitOpen):
        await chain.run()
