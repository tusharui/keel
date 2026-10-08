from __future__ import annotations

import pytest

from keel.scheduler import (
    DeadlinePolicy,
    FCFSPolicy,
    InferenceRequest,
    PriorityDeadlinePolicy,
    ShortestJobFirstPolicy,
    build_policy,
)


def make(request_id: str, **kwargs: object) -> InferenceRequest:
    return InferenceRequest(request_id=request_id, **kwargs)  # type: ignore[arg-type]


def order_of(policy: object, requests: list[InferenceRequest]) -> list[str]:
    return [r.request_id for r in policy.order(requests)]  # type: ignore[attr-defined]


# --- fcfs --------------------------------------------------------------------------


def test_fcfs_preserves_arrival_order() -> None:
    requests = [make("c", enqueued_at_ms=30.0), make("a", enqueued_at_ms=10.0)]
    assert order_of(FCFSPolicy(), requests) == ["a", "c"]


def test_fcfs_is_stable_for_equal_arrival() -> None:
    requests = [make("b", enqueued_at_ms=5.0), make("a", enqueued_at_ms=5.0)]
    assert order_of(FCFSPolicy(), requests) == ["a", "b"]


def test_fcfs_ignores_length_and_deadline() -> None:
    requests = [
        make("long", max_tokens=2000, enqueued_at_ms=1.0),
        make("short", max_tokens=1, enqueued_at_ms=2.0, deadline_ms=3.0),
    ]
    assert order_of(FCFSPolicy(), requests) == ["long", "short"]


# --- sjf ---------------------------------------------------------------------------


def test_sjf_ranks_by_estimated_output() -> None:
    requests = [
        make("long", max_tokens=500, enqueued_at_ms=1.0),
        make("short", max_tokens=5, enqueued_at_ms=9.0),
    ]
    assert order_of(ShortestJobFirstPolicy(), requests) == ["short", "long"]


def test_sjf_breaks_ties_by_arrival() -> None:
    requests = [
        make("second", max_tokens=10, enqueued_at_ms=2.0),
        make("first", max_tokens=10, enqueued_at_ms=1.0),
    ]
    assert order_of(ShortestJobFirstPolicy(), requests) == ["first", "second"]


# --- deadline ----------------------------------------------------------------------


def test_deadline_runs_earliest_first() -> None:
    requests = [
        make("late", deadline_ms=900.0, enqueued_at_ms=1.0),
        make("soon", deadline_ms=100.0, enqueued_at_ms=2.0),
    ]
    assert order_of(DeadlinePolicy(), requests) == ["soon", "late"]


def test_undated_requests_sort_last() -> None:
    requests = [
        make("undated", enqueued_at_ms=1.0),
        make("dated", deadline_ms=100.0, enqueued_at_ms=5.0),
    ]
    assert order_of(DeadlinePolicy(), requests) == ["dated", "undated"]


def test_two_undated_requests_fall_back_to_arrival() -> None:
    requests = [make("b", enqueued_at_ms=2.0), make("a", enqueued_at_ms=1.0)]
    assert order_of(DeadlinePolicy(), requests) == ["a", "b"]


# --- priority ----------------------------------------------------------------------


def test_priority_tier_beats_a_stricter_deadline_in_a_lower_tier() -> None:
    policy = PriorityDeadlinePolicy(tier_of={"gold": 0, "free": 1})
    requests = [
        make("free_urgent", tenant_id="free", deadline_ms=10.0, enqueued_at_ms=1.0),
        make("gold_relaxed", tenant_id="gold", deadline_ms=9000.0, enqueued_at_ms=2.0),
    ]
    assert order_of(policy, requests) == ["gold_relaxed", "free_urgent"]


def test_unknown_tenant_lands_in_the_base_tier() -> None:
    policy = PriorityDeadlinePolicy(tier_of={"gold": 0})
    requests = [
        make("unknown", tenant_id="mystery", enqueued_at_ms=1.0),
        make("gold", tenant_id="gold", deadline_ms=5.0, enqueued_at_ms=2.0),
    ]
    assert order_of(policy, requests) == ["gold", "unknown"]


# --- registry ----------------------------------------------------------------------


@pytest.mark.parametrize("name", ["fcfs", "sjf", "deadline"])
def test_registry_builds_by_name(name: str) -> None:
    assert build_policy(name).name == name


def test_unknown_policy_is_rejected_with_the_valid_names() -> None:
    with pytest.raises(ValueError, match="expected one of"):
        build_policy("round_robin")


def test_ordering_does_not_mutate_its_input() -> None:
    requests = [make("b", enqueued_at_ms=2.0), make("a", enqueued_at_ms=1.0)]
    before = list(requests)
    FCFSPolicy().order(requests)
    assert requests == before


def test_empty_queue_is_fine() -> None:
    assert FCFSPolicy().order([]) == []
