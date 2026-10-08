from __future__ import annotations

import pytest

from keel.clock import ManualClock
from keel.errors import BudgetExceeded
from keel.policy.budgets import BudgetTracker


class SettableClock:
    """Lets a test jump the wall clock without sleeping."""

    def __init__(self, at: float = 0.0) -> None:
        self.t = at

    def now(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


@pytest.fixture
def tracker() -> BudgetTracker:
    return BudgetTracker(clock=ManualClock())


def test_unknown_tenant_has_no_budget(tracker: BudgetTracker) -> None:
    with pytest.raises(KeyError, match="no budget"):
        tracker.get("ghost")


def test_negative_limit_is_rejected(tracker: BudgetTracker) -> None:
    with pytest.raises(ValueError, match="negative"):
        tracker.set_limit("acme", -1)


def test_charge_accumulates(tracker: BudgetTracker) -> None:
    tracker.set_limit("acme", 1_000)
    tracker.charge("acme", 400)
    tracker.charge("acme", 100)
    assert tracker.remaining_micros("acme") == 500


def test_charge_beyond_the_limit_is_refused(tracker: BudgetTracker) -> None:
    tracker.set_limit("acme", 100)
    with pytest.raises(BudgetExceeded, match="exceed"):
        tracker.charge("acme", 200)


def test_charging_exactly_the_limit_is_allowed(tracker: BudgetTracker) -> None:
    tracker.set_limit("acme", 100)
    tracker.charge("acme", 100)
    assert tracker.remaining_micros("acme") == 0


def test_tenants_are_independent(tracker: BudgetTracker) -> None:
    tracker.set_limit("a", 100)
    tracker.set_limit("b", 100)
    tracker.charge("a", 100)
    assert tracker.remaining_micros("a") == 0
    assert tracker.remaining_micros("b") == 100


# --- reservations -------------------------------------------------------------------


def test_reservation_holds_headroom(tracker: BudgetTracker) -> None:
    tracker.set_limit("acme", 1_000)
    tracker.reserve("acme", 800)
    assert tracker.remaining_micros("acme") == 200


def test_concurrent_requests_cannot_overshoot(tracker: BudgetTracker) -> None:
    """Without reservations both requests see the full balance and the tenant
    lands over the limit by one request's cost."""
    tracker.set_limit("acme", 1_000)
    tracker.reserve("acme", 600)
    with pytest.raises(BudgetExceeded, match="would commit"):
        tracker.reserve("acme", 600)
    tracker.reserve("acme", 400)


def test_settle_converts_reservation_to_spend(tracker: BudgetTracker) -> None:
    tracker.set_limit("acme", 1_000)
    tracker.reserve("acme", 500)
    tracker.settle("acme", 500, 300)
    assert tracker.remaining_micros("acme") == 700


def test_settle_refunds_the_unused_reservation(tracker: BudgetTracker) -> None:
    tracker.set_limit("acme", 1_000)
    tracker.reserve("acme", 500)
    assert tracker.settle("acme", 500, 120) == 380
    assert tracker.remaining_micros("acme") == 880


def test_overrun_is_charged_not_refused(tracker: BudgetTracker) -> None:
    tracker.set_limit("acme", 1_000)
    tracker.reserve("acme", 100)
    assert tracker.settle("acme", 100, 900) == 0
    assert tracker.remaining_micros("acme") == 100


def test_settle_can_push_past_the_limit(tracker: BudgetTracker) -> None:
    """The tokens are already generated, so refusing to bill them would make the
    tracker disagree with the invoice."""
    tracker.set_limit("acme", 1_000)
    tracker.reserve("acme", 900)
    tracker.settle("acme", 900, 1_500)
    assert tracker.get("acme").spent_micros == 1_500
    assert tracker.remaining_micros("acme") == 0


def test_reservation_rejects_a_negative_estimate(tracker: BudgetTracker) -> None:
    tracker.set_limit("acme", 100)
    with pytest.raises(ValueError, match="negative"):
        tracker.reserve("acme", -5)


def test_charge_with_no_reservation_uses_the_whole_balance(tracker: BudgetTracker) -> None:
    tracker.set_limit("acme", 1_000)
    tracker.charge("acme", 1_000)
    assert tracker.get("acme").exhausted


# --- periods -----------------------------------------------------------------------


def test_period_key_is_monthly() -> None:
    from datetime import datetime

    tracker = BudgetTracker(clock=ManualClock())
    assert tracker.period_key(datetime(2026, 1, 31)) == "2026-01"
    assert tracker.period_key(datetime(2026, 2, 1)) == "2026-02"


def test_spend_resets_when_the_period_rolls_over() -> None:
    clock = SettableClock()
    tracker = BudgetTracker(clock=clock)  # type: ignore[arg-type]
    tracker.set_limit("acme", 1_000)
    tracker.charge("acme", 900)
    assert tracker.remaining_micros("acme") == 100

    # 40 days forward crosses a month boundary.
    clock.advance(40 * 86_400)
    assert tracker.remaining_micros("acme") == 1_000


def test_spend_survives_within_the_same_period() -> None:
    clock = SettableClock()
    tracker = BudgetTracker(clock=clock)  # type: ignore[arg-type]
    tracker.set_limit("acme", 1_000)
    tracker.charge("acme", 900)

    clock.advance(10 * 86_400)
    assert tracker.remaining_micros("acme") == 100


def test_rollover_clears_reservations_too() -> None:
    clock = SettableClock()
    tracker = BudgetTracker(clock=clock)  # type: ignore[arg-type]
    tracker.set_limit("acme", 1_000)
    tracker.reserve("acme", 900)
    assert tracker.get("acme").committed_micros == 900

    clock.advance(60 * 86_400)
    assert tracker.get("acme").committed_micros == 0


def test_setting_a_limit_again_preserves_spend() -> None:
    tracker = BudgetTracker(clock=ManualClock())
    tracker.set_limit("acme", 1_000)
    tracker.charge("acme", 300)
    tracker.set_limit("acme", 2_000)
    assert tracker.remaining_micros("acme") == 1_700
