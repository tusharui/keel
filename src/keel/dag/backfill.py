from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import date, timedelta

from keel.dag.executor import Executor
from keel.dag.graph import DAG
from keel.dag.state import RunStateStore, wrap_dag
from keel.enums import RunStatus

PartitionFactory = Callable[[str], Awaitable[RunStateStore]]


def daily_partitions(start: date, end: date) -> list[str]:
    """Inclusive date range, oldest first.

    Backfill has to be resumable, so partitions are addressed by their own value
    rather than by loop position. A run interrupted on day 40 and restarted must
    land on day 40, not on wherever the counter happens to be.
    """
    if end < start:
        raise ValueError("end must not precede start")
    span = (end - start).days
    return [(start + timedelta(days=offset)).isoformat() for offset in range(span + 1)]


def hourly_partitions(start: date, days: int) -> list[str]:
    """Hourly partitions for ``days`` days from ``start``."""
    if days < 1:
        raise ValueError("days must be at least 1")
    stamps: list[str] = []
    for day_offset in range(days):
        current = start + timedelta(days=day_offset)
        for hour in range(24):
            stamps.append(f"{current.isoformat()}T{hour:02d}")
    return stamps


@dataclass(slots=True)
class BackfillReport:
    requested: list[str]
    succeeded: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    reused_partitions: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failed

    @property
    def reused_count(self) -> int:
        return len(self.reused_partitions)


async def backfill(
    dag: DAG,
    store_for: PartitionFactory,
    partitions: list[str],
    *,
    version: str = "v1",
    max_parallelism: int = 4,
    dag_parallelism: int = 8,
    continue_on_failure: bool = True,
) -> BackfillReport:
    """Run one DAG across many partitions, bounded on both axes.

    Two independent limits: how many partitions run at once, and how many nodes
    run inside one partition. Collapsing them into a single number either
    serialises partitions unnecessarily or lets one partition's fan-out starve
    the others.

    ``store_for`` must hand back a store on its own session. Partitions run
    concurrently and an AsyncSession is not concurrency safe, so sharing one
    across them corrupts whichever transactions happen to overlap.
    """
    report = BackfillReport(requested=list(partitions))
    partition_gate = asyncio.Semaphore(max_parallelism)

    async def one(partition: str) -> None:
        async with partition_gate:
            store = await store_for(partition)
            await store.start(input_hash=partition)

            wrapped, reused = wrap_dag(dag, store, version=version)
            outcome = await Executor(
                wrapped,
                max_parallelism=dag_parallelism,
                continue_on_failure=continue_on_failure,
            ).run()

            if reused:
                report.reused_partitions.append(partition)

            if outcome.ok:
                await store.finish(RunStatus.SUCCEEDED)
                report.succeeded.append(partition)
            else:
                first_failure = next(iter(outcome.failures.values()), None)
                await store.finish(RunStatus.FAILED)
                report.failed[partition] = (
                    first_failure.error if first_failure else None
                ) or "unknown"

    await asyncio.gather(*(one(partition) for partition in partitions))
    return report
