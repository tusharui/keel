from __future__ import annotations

import asyncio

import pytest

from keel.dag import DAG, Executor, Node
from keel.errors import CycleDetected, MissingDependency, ValidationError


async def ok(value: object = "v") -> object:
    return value


async def boom(_ctx: dict[str, object]) -> object:
    raise RuntimeError("upstream exploded")


def node(name: str, deps: tuple[str, ...] = (), fn=ok) -> Node:
    return Node(name=name, run=fn, depends_on=deps)


# --- construction -------------------------------------------------------------------


def test_empty_graph_is_valid() -> None:
    dag = DAG()
    assert len(dag) == 0
    assert dag.topological_order == ()


def test_duplicate_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        DAG(nodes=(node("a"), node("a")))


def test_missing_dependency_is_named() -> None:
    with pytest.raises(MissingDependency, match="ghost"):
        DAG(nodes=(node("a", ("ghost",)),))


def test_self_dependency_is_rejected() -> None:
    with pytest.raises(CycleDetected, match="itself"):
        DAG(nodes=(node("a", ("a",)),))


def test_empty_node_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        Node(name="", run=ok)


def test_negative_retries_rejected() -> None:
    with pytest.raises(ValueError, match="retries"):
        Node(name="a", run=ok, retries=-1)


def test_unknown_node_lookup_raises() -> None:
    with pytest.raises(KeyError, match="no such node"):
        DAG()["nope"]


# --- cycle detection ----------------------------------------------------------------


def test_two_node_cycle_is_named() -> None:
    with pytest.raises(CycleDetected) as excinfo:
        DAG(nodes=(node("fetch", ("rank",)), node("rank", ("fetch",))))
    message = str(excinfo.value)
    assert "fetch" in message and "rank" in message
    assert "->" in message


def test_three_node_cycle_is_named() -> None:
    with pytest.raises(CycleDetected) as excinfo:
        DAG(
            nodes=(
                node("a", ("c",)),
                node("b", ("a",)),
                node("c", ("b",)),
            )
        )
    message = str(excinfo.value)
    assert message.endswith("a -> c -> b -> a")
    assert {"a", "b", "c"} <= set(message.replace("cycle among", "").replace(":", "").split())


def test_cycle_with_a_valid_branch_still_detected() -> None:
    with pytest.raises(CycleDetected):
        DAG(
            nodes=(
                node("start"),
                node("a", ("b",)),
                node("b", ("a",)),
                node("finish", ("a",)),
            )
        )


def test_a_long_chain_is_not_a_cycle() -> None:
    dag = DAG(
        nodes=tuple(
            node(chr(ord("a") + i), (chr(ord("a") + i - 1),) if i else ()) for i in range(26)
        )
    )
    assert len(dag.topological_order) == 26


# --- ordering ------------------------------------------------------------------------


def test_dependencies_come_before_dependents() -> None:
    dag = DAG(nodes=(node("c", ("a", "b")), node("a"), node("b")))
    order = dag.topological_order
    assert order.index("a") < order.index("c")
    assert order.index("b") < order.index("c")


def test_order_is_deterministic() -> None:
    dag = DAG(nodes=(node("z"), node("y"), node("x")))
    assert dag.topological_order == dag.topological_order


def test_roots_and_leaves() -> None:
    dag = DAG(nodes=(node("a"), node("b", ("a",)), node("c", ("a",))))
    assert set(dag.roots()) == {"a"}
    assert set(dag.leaves()) == {"b", "c"}


def test_ancestors_and_descendants() -> None:
    dag = DAG(nodes=(node("a"), node("b", ("a",)), node("c", ("b",)), node("d", ("c",))))
    assert dag.ancestors("d") == {"a", "b", "c"}
    assert dag.descendants("a") == {"b", "c", "d"}
    assert dag.ancestors("a") == set()
    assert dag.descendants("d") == set()


def test_levels_group_independent_work() -> None:
    dag = DAG(nodes=(node("a"), node("b"), node("c", ("a", "b"))))
    assert dag.levels() == [["a", "b"], ["c"]]


def test_levels_handle_a_diamond() -> None:
    dag = DAG(
        nodes=(
            node("start"),
            node("left", ("start",)),
            node("right", ("start",)),
            node("join", ("left", "right")),
        )
    )
    assert dag.levels() == [["start"], ["left", "right"], ["join"]]


def test_contains_and_names() -> None:
    dag = DAG(nodes=(node("a"), node("b", ("a",))))
    assert "a" in dag
    assert "nope" not in dag
    assert set(dag.names) == {"a", "b"}


# --- execution -------------------------------------------------------------------------


async def test_runs_every_node() -> None:
    dag = DAG(nodes=(node("a"), node("b", ("a",))))
    result = await Executor(dag).run()
    assert result.ok
    assert set(result.results) == {"a", "b"}


async def test_output_is_passed_down() -> None:
    seen_by: dict[str, dict[str, object]] = {}

    def capture(name: str):
        async def run(ctx: dict[str, object]) -> object:
            seen_by[name] = dict(ctx)
            return f"from-{name}"

        return run

    dag = DAG(nodes=(node("a", fn=capture("a")), node("b", ("a",), fn=capture("b"))))
    result = await Executor(dag).run()

    assert result.outputs() == {"a": "from-a", "b": "from-b"}
    assert seen_by["b"]["a"] == "from-a"


async def test_independent_nodes_run_concurrently() -> None:
    order: list[str] = []

    def slow(name: str):
        async def run(_ctx: dict[str, object]) -> object:
            order.append(f"start-{name}")
            await asyncio.sleep(0.01)
            order.append(f"end-{name}")
            return name

        return run

    dag = DAG(nodes=(node("a", fn=slow("a")), node("b", fn=slow("b"))))
    await Executor(dag).run()
    assert order.index("start-a") < order.index("end-a")
    assert order.index("start-b") < order.index("end-b")
    # Interleaved, not sequential: both are running before either finishes.
    assert order[-1].startswith("end-")
    assert len([o for o in order if o.startswith("start-")]) == 2


async def test_parallelism_is_bounded() -> None:
    active = 0
    peak = 0

    async def tracked(_ctx: dict[str, object]) -> object:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return "ok"

    dag = DAG(nodes=tuple(node(f"n{i}", fn=tracked) for i in range(12)))
    await Executor(dag, max_parallelism=3).run()
    assert peak <= 3


async def test_failure_is_reported() -> None:
    dag = DAG(nodes=(node("a", fn=boom),))
    result = await Executor(dag).run()
    assert not result.ok
    assert "a" in result.failures
    assert "upstream exploded" in (result.failures["a"].error or "")


async def test_downstream_is_skipped_after_a_failure() -> None:
    dag = DAG(nodes=(node("a", fn=boom), node("b", ("a",))))
    result = await Executor(dag).run()
    assert result.results["b"].status.value == "skipped"
    assert result.results["b"].skipped_reason


async def test_independent_branch_still_runs_after_a_failure() -> None:
    dag = DAG(nodes=(node("bad", fn=boom), node("good")))
    result = await Executor(dag, continue_on_failure=True).run()
    assert result.results["good"].succeeded


async def test_fail_fast_stops_later_levels() -> None:
    dag = DAG(nodes=(node("bad", fn=boom), node("later", ("bad",))))
    result = await Executor(dag, continue_on_failure=False).run()
    assert result.results["later"].status.value == "skipped"


async def test_node_is_retried_then_succeeds() -> None:
    calls = {"n": 0}

    async def flaky(_ctx: dict[str, object]) -> object:
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("transient")
        return "ok"

    dag = DAG(nodes=(Node(name="a", run=flaky, retries=3),))
    result = await Executor(dag).run()
    assert result.results["a"].succeeded
    assert result.results["a"].attempts == 3


async def test_retries_are_bounded() -> None:
    dag = DAG(nodes=(Node(name="a", run=boom, retries=2),))
    result = await Executor(dag).run()
    assert result.results["a"].status.value == "failed"
    assert result.results["a"].attempts == 3


async def test_validation_errors_are_not_retried() -> None:
    calls = {"n": 0}

    async def invalid(_ctx: dict[str, object]) -> object:
        calls["n"] += 1
        raise ValidationError("bad input")

    dag = DAG(nodes=(Node(name="a", run=invalid, retries=5),))
    result = await Executor(dag).run()
    assert calls["n"] == 1
    assert result.results["a"].status.value == "failed"


async def test_parallelism_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_parallelism"):
        Executor(DAG(), max_parallelism=0)


async def test_run_order_matches_topological_order() -> None:
    dag = DAG(nodes=(node("a"), node("b"), node("c", ("a", "b"))))
    result = await Executor(dag).run()
    assert result.order == dag.topological_order
    assert result.order.index("a") < result.order.index("c")
