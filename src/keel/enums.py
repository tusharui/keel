from __future__ import annotations

from enum import StrEnum


class RequestStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    # Reclaimed KV blocks and went back to the waiting queue. Distinct from
    # FAILED because the work is not lost, only postponed.
    PREEMPTED = "preempted"


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


class NodeStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"
    CACHED = "cached"


class CacheTier(StrEnum):
    MISS = "miss"
    EXACT = "exact"
    PREFIX = "prefix"
    SEMANTIC = "semantic"
