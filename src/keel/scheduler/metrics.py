from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from keel.scheduler.request import InferenceRequest


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile. ``q`` is a fraction in [0, 1]."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * min(max(q, 0.0), 1.0)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[int(position)]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _clean(values: Sequence[float | None]) -> list[float]:
    return [v for v in values if v is not None]


@dataclass(frozen=True, slots=True)
class ServingMetrics:
    """Serving summary for one scheduler run.

    Averages are deliberately absent from the headline fields. A mean TTFT is
    close to useless for capacity planning because it hides exactly the tail
    that users complain about, and it is dominated by whichever prompt happened
    to be largest.
    """

    num_requests: int
    num_finished: int
    num_preempted: int
    makespan_ms: float
    prompt_tokens: int
    completion_tokens: int

    ttft_p50_ms: float
    ttft_p95_ms: float
    ttft_p99_ms: float
    tpot_p50_ms: float
    tpot_p95_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    queue_wait_p50_ms: float
    queue_wait_p95_ms: float

    output_tokens_per_second: float
    goodput_tokens_per_second: float
    mean_batch_size: float
    preemptions: int

    @classmethod
    def from_results(
        cls,
        results: Sequence[InferenceRequest],
        *,
        makespan_ms: float,
        mean_batch_size: float,
        preemptions: int = 0,
        ttft_slo_ms: float | None = None,
        tpot_slo_ms: float | None = None,
    ) -> ServingMetrics:
        finished = [r for r in results if r.is_terminal]
        ttft = _clean([r.ttft_ms for r in finished])
        tpot = _clean([r.tpot_ms for r in finished])
        latency = _clean([r.latency_ms for r in finished])
        wait = _clean([r.queue_wait_ms for r in finished])

        completion_tokens = sum(len(r.output) for r in finished)
        prompt_tokens = sum(r.num_prompt_tokens for r in finished)
        seconds = makespan_ms / 1000.0 if makespan_ms > 0 else 0.0

        meeting_slo = 0
        for request in finished:
            ok = True
            if ttft_slo_ms is not None:
                ok = ok and request.ttft_ms is not None and request.ttft_ms <= ttft_slo_ms
            if tpot_slo_ms is not None:
                ok = ok and request.tpot_ms is not None and request.tpot_ms <= tpot_slo_ms
            if ok:
                meeting_slo += len(request.output)

        return cls(
            num_requests=len(results),
            num_finished=len(finished),
            num_preempted=sum(r.preemptions for r in finished),
            makespan_ms=makespan_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            ttft_p50_ms=percentile(ttft, 0.50),
            ttft_p95_ms=percentile(ttft, 0.95),
            ttft_p99_ms=percentile(ttft, 0.99),
            tpot_p50_ms=percentile(tpot, 0.50),
            tpot_p95_ms=percentile(tpot, 0.95),
            latency_p50_ms=percentile(latency, 0.50),
            latency_p95_ms=percentile(latency, 0.95),
            queue_wait_p50_ms=percentile(wait, 0.50),
            queue_wait_p95_ms=percentile(wait, 0.95),
            output_tokens_per_second=(completion_tokens / seconds if seconds else 0.0),
            goodput_tokens_per_second=(meeting_slo / seconds if seconds else 0.0),
            mean_batch_size=mean_batch_size,
            preemptions=preemptions,
        )

    def as_rows(self) -> list[tuple[str, str]]:
        return [
            ("requests", f"{self.num_finished}/{self.num_requests}"),
            ("makespan", f"{self.makespan_ms / 1000:.2f}s"),
            ("completion tokens", str(self.completion_tokens)),
            ("output tok/s", f"{self.output_tokens_per_second:.1f}"),
            ("goodput tok/s", f"{self.goodput_tokens_per_second:.1f}"),
            ("mean batch", f"{self.mean_batch_size:.2f}"),
            (
                "ttft p50/p95/p99",
                f"{self.ttft_p50_ms:.1f}/{self.ttft_p95_ms:.1f}/{self.ttft_p99_ms:.1f}ms",
            ),
            ("tpot p50/p95", f"{self.tpot_p50_ms:.2f}/{self.tpot_p95_ms:.2f}ms"),
            ("queue wait p50/p95", f"{self.queue_wait_p50_ms:.1f}/{self.queue_wait_p95_ms:.1f}ms"),
            ("latency p50/p95", f"{self.latency_p50_ms:.1f}/{self.latency_p95_ms:.1f}ms"),
            ("preemptions", str(self.preemptions)),
        ]
