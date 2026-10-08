from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from keel.scheduler.request import InferenceRequest


@runtime_checkable
class AdmissionPolicy(Protocol):
    """Decides which waiting request runs next when the batch has room.

    The policy is a pure ordering function and nothing else. It cannot admit,
    reject, or see KV cache state, which keeps the scheduling decision testable
    in isolation and means adding a policy cannot quietly change admission
    control.
    """

    @property
    def name(self) -> str: ...

    def order(self, waiting: Sequence[InferenceRequest]) -> list[InferenceRequest]: ...


@dataclass(frozen=True, slots=True)
class FCFSPolicy:
    """First come, first served.

    The only policy with a fairness guarantee. Worst mean wait of the three,
    because one long request in front of a queue of short ones blocks every one
    of them for its entire duration.
    """

    @property
    def name(self) -> str:
        return "fcfs"

    def order(self, waiting: Sequence[InferenceRequest]) -> list[InferenceRequest]:
        return sorted(waiting, key=lambda r: (r.enqueued_at_ms, r.request_id))


@dataclass(frozen=True, slots=True)
class ShortestJobFirstPolicy:
    """Rank by estimated output length.

    Minimises mean wait at the cost of starving anything long: a steady stream
    of one-token requests can keep a large generation from being admitted at all.
    That starvation is real, not theoretical, and is why this is not the default.
    """

    @property
    def name(self) -> str:
        return "sjf"

    def order(self, waiting: Sequence[InferenceRequest]) -> list[InferenceRequest]:
        return sorted(
            waiting,
            key=lambda r: (r.estimated_output_tokens, r.enqueued_at_ms, r.request_id),
        )


@dataclass(frozen=True, slots=True)
class DeadlinePolicy:
    """Earliest deadline first, falling back to arrival order for undated work.

    The only policy that can reason about a service level objective. Requests
    without deadlines sort last rather than interleaving, so traffic without an
    SLO cannot crowd out traffic that has one.
    """

    @property
    def name(self) -> str:
        return "deadline"

    def order(self, waiting: Sequence[InferenceRequest]) -> list[InferenceRequest]:
        return sorted(
            waiting,
            key=lambda r: (
                r.deadline_ms if r.deadline_ms is not None else math.inf,
                r.enqueued_at_ms,
                r.request_id,
            ),
        )


@dataclass(frozen=True, slots=True)
class PriorityDeadlinePolicy:
    """Deadlines within a priority tier, tiers in ascending order.

    Lets a tenant with a paid SLA share a scheduler with one that has none
    without either starving: the priority tier bounds the damage rather than
    the tenant being deprioritised the moment a stricter deadline appears.
    """

    tier_of: dict[str, int]

    @property
    def name(self) -> str:
        return "priority_deadline"

    def order(self, waiting: Sequence[InferenceRequest]) -> list[InferenceRequest]:
        return sorted(
            waiting,
            key=lambda r: (
                self.tier_of.get(r.tenant_id, 0),
                r.deadline_ms if r.deadline_ms is not None else math.inf,
                r.enqueued_at_ms,
                r.request_id,
            ),
        )


_POLICIES: dict[str, type[FCFSPolicy | ShortestJobFirstPolicy | DeadlinePolicy]] = {
    "fcfs": FCFSPolicy,
    "sjf": ShortestJobFirstPolicy,
    "deadline": DeadlinePolicy,
}


def build_policy(name: str) -> AdmissionPolicy:
    try:
        return _POLICIES[name]()
    except KeyError:
        raise ValueError(
            f"unknown admission policy {name!r}; expected one of {sorted(_POLICIES)}"
        ) from None
