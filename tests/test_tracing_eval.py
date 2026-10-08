from __future__ import annotations

import asyncio

import pytest

from keel.clock import ManualClock
from keel.errors import EvalGateFailed
from keel.eval.harness import Case, Suite, diff, gate
from keel.tracing.spans import Tracer, current_span_id


@pytest.fixture
def tracer() -> Tracer:
    return Tracer(clock=ManualClock())


# --- tracing --------------------------------------------------------------------------


def test_span_records_duration(tracer: Tracer) -> None:
    clock = ManualClock()
    tracer.clock = clock  # type: ignore[assignment]
    with tracer.span("admit"):
        clock.advance(0.5)
    assert tracer.spans[0].name == "admit"
    assert tracer.spans[0].duration_ms == pytest.approx(500.0)


def test_nested_spans_form_a_tree(tracer: Tracer) -> None:
    with tracer.span("request"):
        with tracer.span("prefill"):
            pass
        with tracer.span("decode"):
            pass

    roots = tracer.roots()
    assert len(roots) == 1
    assert {s.name for s in tracer.children_of(roots[0].span_id)} == {"prefill", "decode"}


def test_waterfall_is_depth_ordered(tracer: Tracer) -> None:
    with tracer.span("request"), tracer.span("prefill"), tracer.span("attention"):
        pass
    assert tracer.waterfall() == [
        (0, "request", tracer.spans[0].duration_ms),
        (1, "prefill", tracer.spans[1].duration_ms),
        (2, "attention", tracer.spans[2].duration_ms),
    ]


def test_attributes_are_captured(tracer: Tracer) -> None:
    with tracer.span("admit", seq_id="abc", blocks=12):
        pass
    assert tracer.spans[0].attributes == {"seq_id": "abc", "blocks": 12}


def test_span_is_closed_even_when_the_body_raises(tracer: Tracer) -> None:
    with pytest.raises(RuntimeError), tracer.span("boom"):
        raise RuntimeError("inside")
    assert tracer.spans[0].duration_ms >= 0.0
    assert current_span_id() is None, "a raised span must still pop the context"


def test_active_span_clears_after_the_block(tracer: Tracer) -> None:
    with tracer.span("outer"):
        assert current_span_id() is not None
    assert current_span_id() is None


def test_siblings_do_not_become_parents(tracer: Tracer) -> None:
    with tracer.span("a"):
        pass
    with tracer.span("b"):
        pass
    assert len(tracer.roots()) == 2


async def test_concurrent_spans_do_not_interleave(tracer: Tracer) -> None:
    """ContextVars rather than an instance stack, so concurrent tasks cannot
    parent each other's spans."""

    async def branch(name: str) -> None:
        with tracer.span(name):
            await asyncio.sleep(0)
            with tracer.span(f"{name}.child"):
                pass

    await asyncio.gather(branch("left"), branch("right"))

    by_name = {s.name: s for s in tracer.spans}
    assert by_name["left"].parent_id != by_name["left.child"].parent_id
    assert by_name["left.child"].parent_id == by_name["left"].span_id
    assert by_name["right.child"].parent_id == by_name["right"].span_id
    assert by_name["left"].parent_id is None
    assert by_name["right"].parent_id is None


def test_totals_by_name(tracer: Tracer) -> None:
    for _ in range(3):
        with tracer.span("decode"):
            tracer.clock.advance(0.1)  # type: ignore[attr-defined]
    assert tracer.total_ms("decode") == pytest.approx(300.0)


def test_reset_clears_spans(tracer: Tracer) -> None:
    with tracer.span("a"):
        pass
    original = tracer.trace_id
    tracer.reset()
    assert tracer.spans == []
    assert tracer.trace_id != original


def test_as_dicts_is_serialisable(tracer: Tracer) -> None:
    import json

    with tracer.span("a", k=1):
        pass
    json.dumps(tracer.as_dicts())


# --- eval ------------------------------------------------------------------------------


async def echo(payload: dict[str, object]) -> object:
    return str(payload.get("prompt", ""))


async def suite() -> Suite:
    return Suite(
        "summarise",
        [
            Case(name="mentions-invoice", payload={"prompt": "summarise invoice 42"}),
            Case(name="no-refusal", payload={"prompt": "hello"}, must_not_include=("cannot",)),
        ],
    )


async def test_all_cases_pass() -> None:
    report = await (await suite()).run(echo)
    assert report.ok
    assert report.passed_count == 2
    assert "2/2 passed" in report.summary()


async def test_content_assertion_catches_a_regression() -> None:
    """A gate that only checked for a 200 would let a prompt change turn every
    answer into a refusal without noticing."""

    async def refusing(_payload: dict[str, object]) -> object:
        return "I cannot help with that"

    report = await (await suite()).run(refusing)
    assert not report.ok
    assert "no-refusal" in report.failures()


async def test_required_content_is_enforced() -> None:
    suite_ = Suite("s", [Case(name="c", payload={}, must_include=("Paris",))])
    report = await suite_.run(echo)
    assert not report.ok
    assert "missing 'Paris'" in report.failures()["c"][0]


async def test_exact_output_is_enforced() -> None:
    suite_ = Suite("s", [Case(name="c", payload={"prompt": "x"}, exact="x")])
    assert (await suite_.run(echo)).ok


async def test_minimum_length_is_enforced() -> None:
    suite_ = Suite("s", [Case(name="c", payload={"prompt": "x"}, min_output_chars=10)])
    report = await suite_.run(echo)
    assert not report.ok
    assert "too short" in report.failures()["c"][0]


async def test_raised_error_fails_the_case() -> None:
    async def boom(_payload: dict[str, object]) -> object:
        raise RuntimeError("upstream")

    report = await (await suite()).run(boom)
    assert not report.ok
    assert all("upstream" in f for f in report.failures()["mentions-invoice"])


async def test_duplicate_case_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        Suite("s", [Case(name="a"), Case(name="a")])


async def test_empty_suite_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one"):
        Suite("s", [])


async def test_gate_passes_a_clean_report() -> None:
    gate(await (await suite()).run(echo))


async def test_gate_raises_on_failure() -> None:
    async def refusing(_payload: dict[str, object]) -> object:
        return "I cannot"

    with pytest.raises(EvalGateFailed, match="no-refusal"):
        gate(await (await suite()).run(refusing))


async def test_diff_finds_cases_that_regressed() -> None:
    baseline = await (await suite()).run(echo)

    async def refusing(_payload: dict[str, object]) -> object:
        return "I cannot"

    current = await (await suite()).run(refusing)
    assert diff(baseline, current) == ["no-refusal"]


async def test_diff_ignores_cases_that_were_already_failing() -> None:
    async def refusing(_payload: dict[str, object]) -> object:
        return "I cannot"

    baseline = await (await suite()).run(refusing)
    current = await (await suite()).run(refusing)
    assert diff(baseline, current) == []
