from __future__ import annotations

import asyncio
from collections.abc import Iterable
from dataclasses import dataclass, field

from keel.clock import Clock, default_clock
from keel.dag.graph import DAG
from keel.enums import NodeStatus
from keel.errors import KeelError


@dataclass(slots=True)
class NodeResult:
    node: str
    status: NodeStatus
    output: object = None
    error: str | None = None
    attempts: int = 0
    duration_ms: float = 0.0
    skipped_reason: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.status is NodeStatus.SUCCEEDED


@dataclass(slots=True)
class RunResult:
    results: dict[str, NodeResult] = field(default_factory=dict)
    order: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return all(
            r.status in (NodeStatus.SUCCEEDED, NodeStatus.SKIPPED) for r in self.results.values()
        )

    @property
    def failures(self) -> dict[str, NodeResult]:
        return {name: r for name, r in self.results.items() if r.status is NodeStatus.FAILED}

    def outputs(self) -> dict[str, object]:
        return {name: r.output for name, r in self.results.items() if r.succeeded}


class Executor:
    """Runs a DAG level by level.

    A level only starts once every node in the previous one has finished, so
    independent branches inside a level run concurrently while dependent branches
    stay ordered. Levels are used instead of a rolling ready-queue because it
    makes the concurrency pattern obvious at a glance, and the parallelism bound
    is what actually matters for a pipeline whose branches hit the same backend.
    """

    def __init__(
        self,
        dag: DAG,
        *,
        max_parallelism: int = 8,
        continue_on_failure: bool = False,
        clock: Clock | None = None,
    ) -> None:
        if max_parallelism < 1:
            raise ValueError("max_parallelism must be at least 1")
        self._dag = dag
        self._max_parallelism = max_parallelism
        self._continue_on_failure = continue_on_failure
        self._clock = clock or default_clock()

    async def run(self, context: dict[str, object] | None = None) -> RunResult:
        ctx: dict[str, object] = dict(context or {})
        outcome = RunResult(order=self._dag.topological_order)
        semaphore = asyncio.Semaphore(self._max_parallelism)

        async def run_one(name: str) -> NodeResult:
            async with semaphore:
                result = await self._run_node(name, ctx)
                outcome.results[name] = result
                return result

        for level in self._dag.levels():
            runnable, blocked = self._partition(level, outcome)
            outcome.results.update(self._skipped(blocked, ctx))

            if not runnable:
                continue

            await asyncio.gather(*(run_one(name) for name in runnable))

            if not self._continue_on_failure and outcome.failures:
                outcome.results.update(
                    self._skipped(self._levels_after(level), ctx, reason="upstream failure")
                )
                break

        return outcome

    def _partition(self, level: Iterable[str], outcome: RunResult) -> tuple[list[str], list[str]]:
        runnable: list[str] = []
        blocked: list[str] = []
        for name in level:
            if any(
                outcome.results.get(dep) is not None and not outcome.results[dep].succeeded
                for dep in self._dag[name].depends_on
            ):
                blocked.append(name)
            else:
                runnable.append(name)
        return runnable, blocked

    def _levels_after(self, level: list[str]) -> list[str]:
        levels = self._dag.levels()
        index = levels.index(level)
        return [name for group in levels[index + 1 :] for name in group]

    def _skipped(
        self, names: Iterable[str], ctx: dict[str, object], reason: str | None = None
    ) -> dict[str, NodeResult]:
        return {
            name: NodeResult(
                node=name,
                status=NodeStatus.SKIPPED,
                skipped_reason=reason or "dependency did not succeed",
            )
            for name in names
        }

    async def _run_node(self, name: str, ctx: dict[str, object]) -> NodeResult:
        node = self._dag[name]
        started = self._clock.now()

        attempts = 0
        error: Exception | None = None

        while True:
            attempts += 1
            try:
                output = await node.run(ctx)
            except Exception as exc:
                error = exc
                retryable = not isinstance(exc, KeelError) or exc.retryable
                if not retryable or attempts > node.retries:
                    break
                if node.retry_policy is not None:
                    await self._sleep(node.retry_policy.ceiling_for(attempts))
                continue
            else:
                # Published under the node's own name, so a dependent reads
                # ctx[<dependency name>] the same way it reads a root's input.
                ctx[name] = output
                return NodeResult(
                    node=name,
                    status=NodeStatus.SUCCEEDED,
                    output=output,
                    attempts=attempts,
                    duration_ms=(self._clock.now() - started) * 1000.0,
                )

        return NodeResult(
            node=name,
            status=NodeStatus.FAILED,
            error=f"{type(error).__name__}: {error}",
            attempts=attempts,
            duration_ms=(self._clock.now() - started) * 1000.0,
        )

    async def _sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)
