from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from keel.clock import Clock, default_clock
from keel.errors import BudgetExceeded

MICROS_PER_UNIT = 1_000_000


@dataclass(slots=True)
class Budget:
    """A spend cap for one tenant over one period.

    Spend is held as integer microdollars. Summing fractional currency across a
    busy month drifts, and the drift always surfaces as a customer being charged
    over their agreed cap.

    ``reserved`` is the part of the cap committed to requests that are in flight
    but not yet billed. Without it, every request checks the same not-yet-updated
    balance, so N concurrent requests all see headroom for the full cost and the
    tenant lands over the limit by up to N times one request.
    """

    tenant_id: str
    limit_micros: int
    spent_micros: int = 0
    reserved_micros: int = 0
    period_key: str = ""

    @property
    def committed_micros(self) -> int:
        return self.spent_micros + self.reserved_micros

    @property
    def remaining_micros(self) -> int:
        return max(0, self.limit_micros - self.committed_micros)

    @property
    def exhausted(self) -> bool:
        return self.committed_micros >= self.limit_micros


@dataclass(slots=True)
class BudgetTracker:
    clock: Clock = field(default_factory=default_clock)
    _budgets: dict[str, Budget] = field(default_factory=dict)
    _period: str = ""

    def period_key(self, at: datetime | None = None) -> str:
        moment = at or datetime.fromtimestamp(self.clock.now())
        return f"{moment.year:04d}-{moment.month:02d}"

    def _current_period(self) -> str:
        return self.period_key()

    def _roll_if_needed(self) -> None:
        current = self._current_period()
        if current != self._period:
            for budget in self._budgets.values():
                budget.spent_micros = 0
                budget.reserved_micros = 0
                budget.period_key = current
            self._period = current

    def set_limit(self, tenant_id: str, limit_micros: int) -> Budget:
        if limit_micros < 0:
            raise ValueError("limit_micros cannot be negative")
        self._roll_if_needed()
        budget = self._budgets.get(tenant_id)
        if budget is None:
            budget = Budget(tenant_id=tenant_id, limit_micros=limit_micros, period_key=self._period)
            self._budgets[tenant_id] = budget
        else:
            budget.limit_micros = limit_micros
        return budget

    def get(self, tenant_id: str) -> Budget:
        self._roll_if_needed()
        budget = self._budgets.get(tenant_id)
        if budget is None:
            raise KeyError(f"no budget configured for tenant {tenant_id!r}")
        return budget

    def remaining_micros(self, tenant_id: str) -> int:
        return self.get(tenant_id).remaining_micros

    def reserve(self, tenant_id: str, estimate_micros: int) -> None:
        if estimate_micros < 0:
            raise ValueError("estimate_micros cannot be negative")
        budget = self.get(tenant_id)
        if budget.committed_micros + estimate_micros > budget.limit_micros:
            raise BudgetExceeded(
                f"tenant {tenant_id} would commit {budget.committed_micros + estimate_micros} "
                f"micros against a limit of {budget.limit_micros}"
            )
        budget.reserved_micros += estimate_micros

    def settle(self, tenant_id: str, reserved_micros: int, actual_micros: int) -> int:
        """Convert a reservation into actual spend. Returns the amount refunded.

        Overrun is charged rather than rejected: the tokens are already generated
        by the time the real cost is known, so refusing to bill it would make the
        tracker disagree with the invoice.
        """
        budget = self.get(tenant_id)
        budget.reserved_micros -= reserved_micros
        budget.spent_micros += actual_micros
        return max(0, reserved_micros - actual_micros)

    def charge(self, tenant_id: str, actual_micros: int) -> None:
        """Record spend with no prior reservation."""
        budget = self.get(tenant_id)
        if budget.committed_micros + actual_micros > budget.limit_micros:
            raise BudgetExceeded(
                f"tenant {tenant_id} would exceed its budget "
                f"({budget.committed_micros + actual_micros} > {budget.limit_micros})"
            )
        budget.spent_micros += actual_micros
