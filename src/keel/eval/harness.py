from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field

from keel.errors import EvalGateFailed

Target = Callable[[dict[str, object]], Awaitable[object]]


@dataclass(frozen=True, slots=True)
class Case:
    name: str
    payload: dict[str, object] = field(default_factory=dict)
    must_include: tuple[str, ...] = ()
    must_not_include: tuple[str, ...] = ()
    exact: object = None
    max_latency_ms: float | None = None
    min_output_chars: int = 0


@dataclass(slots=True)
class CaseResult:
    name: str
    passed: bool
    failures: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    output: object = None


@dataclass(slots=True)
class EvalReport:
    suite: str
    results: list[CaseResult] = field(default_factory=list)
    regressions: list[str] = field(default_factory=list)

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def pass_rate(self) -> float:
        return self.passed_count / self.total if self.total else 1.0

    @property
    def ok(self) -> bool:
        return self.passed_count == self.total and not self.regressions

    def failures(self) -> dict[str, list[str]]:
        return {r.name: r.failures for r in self.results if not r.passed}

    def summary(self) -> str:
        return f"{self.suite}: {self.passed_count}/{self.total} passed ({self.pass_rate:.0%})" + (
            f", {len(self.regressions)} regressed" if self.regressions else ""
        )


class Suite:
    """Prompt regression harness.

    A gate only earns its keep if it can fail, so a case is allowed to assert on
    content as well as latency. Checking that the endpoint returned 200 would let
    a prompt change quietly turn every answer into a refusal.
    """

    def __init__(self, name: str, cases: Sequence[Case]) -> None:
        if not cases:
            raise ValueError("a suite needs at least one case")
        seen: set[str] = set()
        for case in cases:
            if case.name in seen:
                raise ValueError(f"duplicate case name {case.name!r}")
            seen.add(case.name)
        self.name = name
        self.cases = tuple(cases)

    async def run(self, target: Target) -> EvalReport:
        results: list[CaseResult] = []
        for case in self.cases:
            results.append(await self._run_case(case, target))
        return EvalReport(suite=self.name, results=results)

    async def _run_case(self, case: Case, target: Target) -> CaseResult:
        failures: list[str] = []
        started = _monotonic()
        try:
            output = await target(case.payload)
        except Exception as error:
            return CaseResult(
                name=case.name,
                passed=False,
                failures=[f"{type(error).__name__}: {error}"],
                latency_ms=(_monotonic() - started) * 1000.0,
            )

        latency_ms = (_monotonic() - started) * 1000.0
        text = output if isinstance(output, str) else str(output)

        for needle in case.must_include:
            if needle not in text:
                failures.append(f"missing {needle!r}")
        for needle in case.must_not_include:
            if needle in text:
                failures.append(f"unexpected {needle!r}")
        if case.exact is not None and output != case.exact:
            failures.append(f"expected {case.exact!r}, got {output!r}")
        if len(text) < case.min_output_chars:
            failures.append(f"output too short: {len(text)} < {case.min_output_chars}")
        if case.max_latency_ms is not None and latency_ms > case.max_latency_ms:
            failures.append(f"latency {latency_ms:.0f}ms over {case.max_latency_ms:.0f}ms")

        return CaseResult(
            name=case.name,
            passed=not failures,
            failures=failures,
            latency_ms=latency_ms,
            output=output,
        )


def gate(report: EvalReport, baseline: EvalReport | None = None) -> None:
    """Raise unless the report is clean and no case regressed.

    A case that passed on the baseline and fails now is a regression even if the
    overall pass rate went up, which is the failure a rate-only gate misses.
    """
    problems: list[str] = list(report.regressions)
    for name, failures in report.failures().items():
        problems.append(f"{name}: {'; '.join(failures)}")

    if problems:
        raise EvalGateFailed(f"{report.summary()} -- " + " | ".join(problems))


def diff(baseline: EvalReport, current: EvalReport) -> list[str]:
    """Cases that passed on the baseline and fail now."""
    was_passing = {r.name for r in baseline.results if r.passed}
    now_failing = {r.name for r in current.results if not r.passed}
    return sorted(was_passing & now_failing)


def _monotonic() -> float:
    import time

    return time.perf_counter()
