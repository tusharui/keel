from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from keel.dag.graph import DAG, Node
from keel.db.models import Node as NodeRow
from keel.db.models import Run, RunEvent
from keel.enums import NodeStatus, RunStatus
from keel.sim_engine.model import stable_digest


def node_signature(
    dag_name: str,
    partition: str | None,
    node_name: str,
    version: str,
    inputs: dict[str, object],
) -> str:
    """Content hash of everything that determines this node's output.

    Includes the declared ``version`` string rather than the function's bytecode
    so a node can be marked changed deliberately. Hashing source would silently
    invalidate everything when an unrelated comment moved, and would fail to
    invalidate anything when a dependency's behaviour changed.
    """
    material = "|".join(f"{k}={v!r}" for k, v in sorted(inputs.items()))
    return f"{stable_digest(dag_name, partition, node_name, version, material):016x}"


def serialise(value: object) -> str | None:
    """JSON for a node output, or None when it cannot survive a round trip.

    Reusing a ``repr`` would hand the next node a different type than the one the
    pipeline actually produced, and the failure would surface far from the cause.
    Anything not JSON-serialisable is stored as unreusable rather than stored
    wrongly.
    """
    try:
        return json.dumps(value)
    except (TypeError, ValueError):
        return None


def deserialise(raw: str | None) -> object | None:
    if raw is None:
        return None
    try:
        decoded: object = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return decoded


@dataclass(slots=True)
class CachedNode:
    node: str
    signature: str
    output: object


class RunStateStore:
    """Durable DAG state, and the reason a crashed run can be resumed.

    Two separate mechanisms, often confused:

    * **Resume** replays the current run's own node rows. Same run, still going.
    * **Skip** finds an earlier successful node with a matching signature under a
      different run id. A rerun, or a backfill that has already covered this
      partition.
    """

    def __init__(self, session: AsyncSession, dag_name: str, partition: str | None = None) -> None:
        self._session = session
        self._dag_name = dag_name
        self._partition = partition
        self._run: Run | None = None

    @property
    def run_id(self) -> str:
        if self._run is None:
            raise RuntimeError("run has not been started")
        return self._run.id

    @property
    def dag_name(self) -> str:
        return self._dag_name

    @property
    def partition(self) -> str | None:
        return self._partition

    async def start(self, input_hash: str) -> Run:
        """Begin, or resume, the run identified by (dag, partition, input hash).

        Reusing the row rather than inserting a second one is what makes a
        rerun idempotent: the unique constraint on those three columns is the
        database refusing to let the same logical run exist twice, and the only
        honest response is to pick the existing run back up.
        """
        existing = await self._session.scalar(
            select(Run).where(
                Run.dag_name == self._dag_name,
                Run.partition == self._partition,
                Run.input_hash == input_hash,
            )
        )
        if existing is not None:
            existing.status = RunStatus.RUNNING
            await self._session.commit()
            await self._session.refresh(existing)
            self._run = existing
            await self.log("run_resumed")
            return existing

        run = Run(
            dag_name=self._dag_name,
            partition=self._partition,
            input_hash=input_hash,
            status=RunStatus.RUNNING,
        )
        self._session.add(run)
        await self._session.commit()
        await self._session.refresh(run)
        self._run = run
        await self.log("run_started")
        return run

    async def _node_row(self, name: str) -> NodeRow | None:
        if self._run is None:
            return None
        return await self._session.scalar(
            select(NodeRow).where(NodeRow.run_id == self._run.id, NodeRow.name == name)
        )

    async def record_success(
        self, name: str, signature: str, output: object, attempts: int = 1
    ) -> None:
        row = await self._node_row(name)
        if row is None:
            row = NodeRow(run_id=self.run_id, name=name, signature=signature)
            self._session.add(row)
        row.status = NodeStatus.SUCCEEDED
        row.signature = signature
        row.output = serialise(output)
        row.attempt = attempts
        row.error = None
        await self._session.commit()
        await self.log("node_succeeded", name, {"signature": signature})

    async def record_failure(self, name: str, error: str, attempts: int = 1) -> None:
        row = await self._node_row(name)
        if row is None:
            row = NodeRow(run_id=self.run_id, name=name, signature="")
            self._session.add(row)
        row.status = NodeStatus.FAILED
        row.error = error
        row.attempt = attempts
        await self._session.commit()
        await self.log("node_failed", name, {"error": error})

    async def record_skip(self, name: str, reason: str) -> None:
        row = await self._node_row(name)
        if row is None:
            row = NodeRow(run_id=self.run_id, name=name, signature="")
            self._session.add(row)
        row.status = NodeStatus.SKIPPED
        row.error = reason
        await self._session.commit()
        await self.log("node_skipped", name, {"reason": reason})

    async def log(self, event: str, node_name: str | None = None, detail: Any = None) -> None:
        self._session.add(
            RunEvent(run_id=self.run_id, node_name=node_name, event=event, detail=detail)
        )
        await self._session.commit()

    async def finish(self, status: RunStatus) -> None:
        if self._run is None:
            return
        self._run.status = status
        await self._session.commit()
        await self.log("run_finished", None, {"status": status.value})

    async def completed_signatures(self) -> dict[str, str]:
        """Signatures already satisfied inside this run, for resuming."""
        if self._run is None:
            return {}
        rows = await self._session.scalars(
            select(NodeRow).where(
                NodeRow.run_id == self._run.id, NodeRow.status == NodeStatus.SUCCEEDED
            )
        )
        return {row.name: row.signature for row in rows}

    async def cached_output(self, name: str, signature: str) -> CachedNode | None:
        """A matching success from any earlier run of this dag and partition."""
        statement = (
            select(NodeRow)
            .join(Run, Run.id == NodeRow.run_id)
            .where(
                Run.dag_name == self._dag_name,
                Run.partition == self._partition,
                NodeRow.name == name,
                NodeRow.signature == signature,
                NodeRow.status == NodeStatus.SUCCEEDED,
            )
            .order_by(Run.created_at.desc())
            .limit(1)
        )
        row = await self._session.scalar(statement)
        if row is None:
            return None
        output = deserialise(row.output)
        if output is None:
            return None
        return CachedNode(node=name, signature=signature, output=output)


def wrap_dag(
    dag: DAG,
    store: RunStateStore,
    *,
    version: str = "v1",
    use_cache: bool = True,
    inputs: dict[str, object] | None = None,
) -> tuple[DAG, list[str]]:
    """Wrap every node so its result is persisted and a prior result is reused.

    ``inputs`` declares the external data a pipeline was invoked with, and it
    applies only to root nodes. A root has no upstream output to key on, so
    without this it would be reused no matter what the caller changed. A node
    with dependencies is keyed on those outputs alone: it does not read the
    invocation data, and hashing it in would invalidate work that never looked at
    it. Anything a mid-graph node genuinely needs should arrive through a
    declared dependency, ideally from a root that normalises it.
    """
    reused: list[str] = []
    external = dict(inputs or {})

    def wrap(node: Node) -> Node:
        async def run(ctx: dict[str, object]) -> object:
            material: dict[str, object] = dict(external) if not node.depends_on else {}
            for dependency in node.depends_on:
                material[f"dep:{dependency}"] = ctx.get(dependency)
            signature = node_signature(
                store.dag_name,
                store.partition,
                node.name,
                version,
                material,
            )
            if use_cache:
                cached = await store.cached_output(node.name, signature)
                if cached is not None:
                    reused.append(node.name)
                    await store.record_skip(node.name, f"reused {signature}")
                    return cached.output

            output = await node.run(ctx)
            await store.record_success(node.name, signature, output)
            return output

        return replace(node, run=run)

    return DAG(nodes=tuple(wrap(node) for node in dag.nodes)), reused
