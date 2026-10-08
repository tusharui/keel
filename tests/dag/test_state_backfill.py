from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import date

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from keel.dag import DAG, Executor, Node
from keel.dag.backfill import backfill, daily_partitions, hourly_partitions
from keel.dag.state import RunStateStore, node_signature, wrap_dag
from keel.db.base import Base
from keel.db.models import Node as NodeRow
from keel.db.models import Run, RunEvent
from keel.db.session import build_engine, build_session_factory
from keel.enums import NodeStatus, RunStatus


@pytest.fixture
async def session_factory() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = build_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    yield build_session_factory(engine)
    await engine.dispose()


@pytest.fixture
async def session(session_factory: async_sessionmaker[AsyncSession]) -> AsyncIterator[AsyncSession]:
    async with session_factory() as db:
        yield db


@pytest.fixture
def stores(session_factory: async_sessionmaker[AsyncSession]):
    """One session per partition.

    Backfill runs partitions concurrently and an AsyncSession is not concurrency
    safe, so each partition gets its own. The factory tracks them so the test can
    close them afterwards.
    """
    opened: list[AsyncSession] = []

    async def store_for(partition: str) -> RunStateStore:
        db = session_factory()
        opened.append(db)
        return RunStateStore(db, "nightly", partition)

    store_for.opened = opened  # type: ignore[attr-defined]
    return store_for


async def close_stores(store_for) -> None:
    for db in getattr(store_for, "opened", []):
        await db.close()


def value(name: str):
    async def run(ctx: dict[str, object]) -> object:
        return f"{name}:{ctx.get('seed', '-')}"

    return run


def simple_dag() -> DAG:
    return DAG(
        nodes=(
            Node(name="extract", run=value("extract")),
            Node(name="transform", run=value("transform"), depends_on=("extract",)),
            Node(name="load", run=value("load"), depends_on=("transform",)),
        )
    )


# --- signatures ---------------------------------------------------------------------


def test_signature_is_stable_for_equal_inputs() -> None:
    a = node_signature("d", "p", "n", "v1", {"x": 1})
    assert a == node_signature("d", "p", "n", "v1", {"x": 1})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dag_name": "other"},
        {"partition": "other"},
        {"node_name": "other"},
        {"version": "v2"},
        {"inputs": {"x": 2}},
        {"inputs": {}},
    ],
)
def test_every_component_changes_the_signature(kwargs: dict[str, object]) -> None:
    base = {
        "dag_name": "d",
        "partition": "p",
        "node_name": "n",
        "version": "v1",
        "inputs": {"x": 1},
    }
    assert node_signature(**base) != node_signature(**{**base, **kwargs})


def test_input_order_does_not_matter() -> None:
    assert node_signature("d", "p", "n", "v1", {"a": 1, "b": 2}) == node_signature(
        "d", "p", "n", "v1", {"b": 2, "a": 1}
    )


def test_version_bumps_invalidate_a_node() -> None:
    """The declared version is what makes a deliberate change visible. Hashing
    source instead would invalidate on a comment move and miss a dependency
    change."""
    assert node_signature("d", "p", "n", "v1", {}) != node_signature("d", "p", "n", "v2", {})


# --- persistence ---------------------------------------------------------------------


async def test_start_records_a_run(session: AsyncSession) -> None:
    store = RunStateStore(session, "nightly", "2026-01-01")
    run = await store.start("hash-1")
    assert run.status is RunStatus.RUNNING
    assert store.run_id == run.id

    events = (await session.execute(select(RunEvent))).scalars().all()
    assert any(e.event == "run_started" for e in events)


async def test_run_id_before_start_is_an_error(session: AsyncSession) -> None:
    with pytest.raises(RuntimeError, match="not been started"):
        _ = RunStateStore(session, "nightly").run_id


async def test_node_success_is_persisted(session: AsyncSession) -> None:
    store = RunStateStore(session, "nightly", "2026-01-01")
    await store.start("h")
    await store.record_success("extract", "sig1", {"rows": 3})

    row = await session.scalar(select(NodeRow).where(NodeRow.name == "extract"))
    assert row is not None
    assert row.status is NodeStatus.SUCCEEDED
    assert row.signature == "sig1"
    assert row.output == '{"rows": 3}'


async def test_failure_is_recorded_with_its_message(session: AsyncSession) -> None:
    store = RunStateStore(session, "nightly", "2026-01-01")
    await store.start("h")
    await store.record_failure("extract", "RuntimeError: boom")

    row = await session.scalar(select(NodeRow).where(NodeRow.name == "extract"))
    assert row is not None
    assert row.status is NodeStatus.FAILED
    assert "boom" in (row.error or "")


async def test_skip_is_distinct_from_success(session: AsyncSession) -> None:
    store = RunStateStore(session, "nightly", "2026-01-01")
    await store.start("h")
    await store.record_skip("extract", "reused abc123")

    row = await session.scalar(select(NodeRow).where(NodeRow.name == "extract"))
    assert row is not None
    assert row.status is NodeStatus.SKIPPED
    assert row.signature == ""


async def test_partitions_are_isolated_from_each_other(session: AsyncSession) -> None:
    january = RunStateStore(session, "nightly", "2026-01-01")
    await january.start("h")
    await january.record_success("extract", "sig", "january-output")

    february = RunStateStore(session, "nightly", "2026-02-01")
    await february.start("h")
    assert await february.cached_output("extract", "sig") is None


# --- reuse ---------------------------------------------------------------------------


async def test_second_run_reuses_the_first(session: AsyncSession) -> None:
    dag = simple_dag()
    calls = {"n": 0}

    async def counted(ctx: dict[str, object]) -> object:
        calls["n"] += 1
        return "extracted"

    dag = DAG(
        nodes=(
            Node(name="extract", run=counted),
            Node(name="load", run=value("load"), depends_on=("extract",)),
        )
    )

    first = RunStateStore(session, "nightly", "2026-01-01")
    await first.start("h")
    wrapped, _ = wrap_dag(dag, first)
    result = await Executor(wrapped).run()
    assert result.ok
    assert calls["n"] == 1

    second = RunStateStore(session, "nightly", "2026-01-01")
    await second.start("h2")
    wrapped2, reused = wrap_dag(dag, second)
    result2 = await Executor(wrapped2).run()

    assert result2.ok
    assert calls["n"] == 1, "unchanged work must not run twice"
    assert "extract" in reused


async def test_version_bump_forces_recomputation(session: AsyncSession) -> None:
    dag = simple_dag()

    first = RunStateStore(session, "nightly", "2026-01-01")
    await first.start("h")
    await Executor(wrap_dag(dag, first)[0]).run()

    second = RunStateStore(session, "nightly", "2026-01-01")
    await second.start("h2")
    wrapped, reused = wrap_dag(dag, second, version="v2")
    await Executor(wrapped).run()

    assert reused == [], "a version bump must invalidate every node"


async def test_changed_input_invalidates_downstream(session: AsyncSession) -> None:
    """Changing what `a` returns changes `b`'s input, so `b` must rerun too."""

    async def produce(ctx: dict[str, object]) -> object:
        return f"a:{ctx.get('seed', '-')}"

    dag = DAG(
        nodes=(
            Node(name="a", run=produce),
            Node(name="b", run=value("b"), depends_on=("a",)),
        )
    )

    first = RunStateStore(session, "d", "p")
    await first.start("h")
    await Executor(wrap_dag(dag, first, inputs={"seed": "one"})[0]).run({"seed": "one"})

    second = RunStateStore(session, "d", "p")
    await second.start("h2")
    wrapped, reused = wrap_dag(dag, second, inputs={"seed": "two"})
    await Executor(wrapped).run({"seed": "two"})

    assert reused == [], "a changed, so b's input changed with it"


async def test_unchanged_output_reuses_downstream(session: AsyncSession) -> None:
    """The context changes but `a`'s output does not, so `b` is still reusable.
    Keying on the context instead of the value would recompute the whole graph on
    every run."""

    async def constant(_ctx: dict[str, object]) -> object:
        return "stable"

    dag = DAG(
        nodes=(
            Node(name="a", run=constant),
            Node(name="b", run=value("b"), depends_on=("a",)),
        )
    )

    first = RunStateStore(session, "d", "p")
    await first.start("h")
    await Executor(wrap_dag(dag, first, inputs={"seed": "one"})[0]).run({"seed": "one"})

    second = RunStateStore(session, "d", "p")
    await second.start("h2")
    wrapped, reused = wrap_dag(dag, second, inputs={"seed": "two"})
    await Executor(wrapped).run({"seed": "two"})

    # The declared inputs changed, so a reruns. a's output did not, so b is
    # still reusable. Keying b on the context rather than on what it actually
    # reads would recompute the whole downstream chain on every call.
    assert "a" not in reused
    assert "b" in reused


async def test_unserialisable_output_is_not_reused(session: AsyncSession) -> None:
    """Storing a repr would hand the next node a different type than the pipeline
    produced. Better to store nothing and recompute than to store it wrongly."""
    calls = {"n": 0}

    async def opaque(_ctx: dict[str, object]) -> object:
        calls["n"] += 1
        return object()

    dag = DAG(nodes=(Node(name="a", run=opaque),))

    first = RunStateStore(session, "d", "p")
    await first.start("h")
    await Executor(wrap_dag(dag, first)[0]).run()

    second = RunStateStore(session, "d", "p")
    await second.start("h2")
    wrapped, reused = wrap_dag(dag, second)
    await Executor(wrapped).run()

    assert calls["n"] == 2, "an unrepresentable output must be recomputed"
    assert reused == []


async def test_cached_output_survives_as_its_original_type(session: AsyncSession) -> None:
    seen: list[object] = []

    async def capture(ctx: dict[str, object]) -> object:
        seen.append(ctx.get("a"))
        return "ok"

    dag = DAG(
        nodes=(
            Node(name="a", run=_dict_output),
            Node(name="b", run=capture, depends_on=("a",)),
        )
    )

    first = RunStateStore(session, "d", "p")
    await first.start("h")
    await Executor(wrap_dag(dag, first)[0]).run()

    second = RunStateStore(session, "d", "p")
    await second.start("h2")
    wrapped, reused = wrap_dag(dag, second)
    await Executor(wrapped).run()

    assert set(reused) == {"a", "b"}
    # b's body never runs a second time, because a reused node short-circuits
    # before the wrapped function is invoked. The dict reaching b in the first run
    # is a dict, not a stringified one.
    assert seen == [{"rows": 3, "nested": [1, 2]}]


async def _dict_output(_ctx: dict[str, object]) -> object:
    return {"rows": 3, "nested": [1, 2]}


async def test_cache_can_be_disabled(session: AsyncSession) -> None:
    dag = simple_dag()
    first = RunStateStore(session, "d", "p")
    await first.start("h")
    await Executor(wrap_dag(dag, first)[0]).run()

    second = RunStateStore(session, "d", "p")
    await second.start("h2")
    wrapped, reused = wrap_dag(dag, second, use_cache=False)
    await Executor(wrapped).run()
    assert reused == []


# --- partitions ----------------------------------------------------------------------


def test_daily_partitions_are_inclusive() -> None:
    got = daily_partitions(date(2026, 1, 1), date(2026, 1, 3))
    assert got == ["2026-01-01", "2026-01-02", "2026-01-03"]


def test_single_day_partition() -> None:
    assert daily_partitions(date(2026, 1, 1), date(2026, 1, 1)) == ["2026-01-01"]


def test_reversed_range_is_rejected() -> None:
    with pytest.raises(ValueError, match="must not precede"):
        daily_partitions(date(2026, 1, 5), date(2026, 1, 1))


def test_partitions_address_by_value_not_position() -> None:
    """An interrupted backfill must resume on the same partition, not on
    wherever the loop counter happens to be."""
    partitions = daily_partitions(date(2026, 1, 1), date(2026, 1, 5))
    assert partitions[3] == "2026-01-04"
    assert partitions[3] == daily_partitions(date(2026, 1, 4), date(2026, 1, 4))[0]


def test_hourly_partitions_cover_the_days() -> None:
    got = hourly_partitions(date(2026, 1, 1), 2)
    assert len(got) == 48
    assert got[0] == "2026-01-01T00"
    assert got[23] == "2026-01-01T23"
    assert got[24] == "2026-01-02T00"


def test_hourly_partitions_reject_zero_days() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        hourly_partitions(date(2026, 1, 1), 0)


# --- backfill ---------------------------------------------------------------------------


async def test_backfill_runs_every_partition(stores) -> None:
    dag = simple_dag()
    partitions = daily_partitions(date(2026, 1, 1), date(2026, 1, 3))
    report = await backfill(dag, stores, partitions, max_parallelism=1)

    assert report.ok
    assert set(report.succeeded) == set(partitions)

    async with stores.opened[0] as db:
        runs = (await db.execute(select(Run))).scalars().all()
        assert len(runs) == 3
        assert all(r.status is RunStatus.SUCCEEDED for r in runs)
    await close_stores(stores)


async def test_backfill_reuses_unchanged_partitions(stores) -> None:
    dag = simple_dag()
    partitions = daily_partitions(date(2026, 1, 1), date(2026, 1, 2))

    first = await backfill(dag, stores, partitions, max_parallelism=1)
    assert first.ok
    await close_stores(stores)

    second = await backfill(dag, stores, partitions, max_parallelism=1)
    assert second.ok
    assert second.reused_count == 2
    await close_stores(stores)


async def test_backfill_records_failures_per_partition(stores) -> None:
    async def bad(ctx: dict[str, object]) -> object:
        raise RuntimeError(f"bad partition {ctx.get('partition')}")

    dag = DAG(nodes=(Node(name="only", run=bad),))

    report = await backfill(
        dag,
        stores,
        daily_partitions(date(2026, 1, 1), date(2026, 1, 3)),
        max_parallelism=1,
    )
    assert not report.ok
    assert set(report.failed) == {"2026-01-01", "2026-01-02", "2026-01-03"}
    assert report.failed["2026-01-02"] is not None
    await close_stores(stores)


async def test_failed_run_is_marked_failed_in_the_database(stores) -> None:
    async def bad(_ctx: dict[str, object]) -> object:
        raise RuntimeError("nope")

    dag = DAG(nodes=(Node(name="only", run=bad),))
    await backfill(dag, stores, ["2026-01-01"], max_parallelism=1)

    async with stores.opened[0] as db:
        run = await db.scalar(select(Run))
        assert run is not None
        assert run.status is RunStatus.FAILED
    await close_stores(stores)


async def test_empty_backfill_is_a_no_op(stores) -> None:
    report = await backfill(simple_dag(), stores, [], max_parallelism=1)
    assert report.ok
    assert report.succeeded == []
