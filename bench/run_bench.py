from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass

from bench.workload import WorkloadSpec, generate

from keel.clock import ManualClock
from keel.config import SchedulerConfig
from keel.scheduler.metrics import ServingMetrics
from keel.scheduler.policies import build_policy
from keel.scheduler.request import InferenceRequest, RequestState
from keel.scheduler.scheduler import Scheduler
from keel.sim_engine import DeviceProfile, SimModel, Tokenizer
from keel.sim_engine.engine import InferenceEngine
from keel.sim_engine.kv_cache import KVCacheManager

CONTEXT = 8192
BLOCK_SIZE = 16
TTFT_SLO_MS = 2_000.0
TPOT_SLO_MS = 250.0


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    mode: str = "batched"
    policy: str = "fcfs"
    max_num_seqs: int = 64
    max_num_batched_tokens: int = 8_192
    chunked_prefill_tokens: int = 2_048
    enable_preemption: bool = True
    num_blocks: int = 8_192


def _engine(clock: ManualClock, num_blocks: int) -> InferenceEngine:
    return InferenceEngine(
        tokenizer=Tokenizer(vocab_size=32_000),
        model=SimModel(vocab_size=32_000),
        device=DeviceProfile(),
        kv=KVCacheManager(num_blocks=num_blocks, block_size=BLOCK_SIZE),
        context_length=CONTEXT,
        clock=clock,
    )


def run_batched(spec: WorkloadSpec, scenario: Scenario) -> ServingMetrics:
    clock = ManualClock()
    engine = _engine(clock, scenario.num_blocks)
    config = SchedulerConfig(
        max_num_seqs=scenario.max_num_seqs,
        max_num_batched_tokens=scenario.max_num_batched_tokens,
        chunked_prefill_tokens=scenario.chunked_prefill_tokens,
        enable_preemption=scenario.enable_preemption,
    )
    scheduler = Scheduler(engine, config, policy=build_policy(scenario.policy), clock=clock)
    for request in generate(spec):
        scheduler.submit(request)
    scheduler.run(max_iterations=10_000_000)
    return scheduler.metrics(ttft_slo_ms=TTFT_SLO_MS, tpot_slo_ms=TPOT_SLO_MS)


def run_sequential(spec: WorkloadSpec) -> ServingMetrics:
    """One request at a time. The baseline every batching claim is measured against."""
    clock = ManualClock()
    engine = _engine(clock, num_blocks=8_192)
    results: list[InferenceRequest] = []

    for request in generate(spec):
        now_ms = clock.now() * 1000.0
        if request.arrival_ms > now_ms:
            clock.advance((request.arrival_ms - now_ms) / 1000.0)

        started_ms = clock.now() * 1000.0
        generation = engine.run(request.prompt_tokens, request.max_tokens)
        request.state = RequestState.RUNNING
        request.started_at_ms = started_ms
        request.first_token_at_ms = started_ms + generation.ttft_ms
        request.finished_at_ms = started_ms + generation.duration_ms
        request.enqueued_at_ms = started_ms
        request.output = list(generation.token_ids)
        request.state = RequestState.FINISHED
        results.append(request)

    return ServingMetrics.from_results(
        results,
        makespan_ms=clock.now() * 1000.0,
        mean_batch_size=1.0,
        ttft_slo_ms=TTFT_SLO_MS,
        tpot_slo_ms=TPOT_SLO_MS,
    )


def scenarios() -> list[Scenario]:
    return [
        Scenario(name="sequential", mode="sequential"),
        Scenario(name="batched max_num_seqs=1", max_num_seqs=1),
        Scenario(name="batched max_num_seqs=8", max_num_seqs=8),
        Scenario(name="batched max_num_seqs=32", max_num_seqs=32),
        Scenario(name="batched max_num_seqs=128", max_num_seqs=128),
        Scenario(name="policy sjf", max_num_seqs=32, policy="sjf"),
        Scenario(name="policy deadline", max_num_seqs=32, policy="deadline"),
        # Chunked prefill only bites once the batch budget is small enough to
        # stop a whole prompt being prefilled in one pass. With the default
        # budget these two rows are identical, which is the finding.
        Scenario(
            name="budget=1024 chunk=1024",
            max_num_seqs=32,
            max_num_batched_tokens=1024,
            chunked_prefill_tokens=1024,
        ),
        Scenario(
            name="budget=1024 chunk=128",
            max_num_seqs=32,
            max_num_batched_tokens=1024,
            chunked_prefill_tokens=128,
        ),
        Scenario(name="kv=256 preemption on", max_num_seqs=32, num_blocks=256),
        Scenario(
            name="kv=256 preemption off", max_num_seqs=32, num_blocks=256, enable_preemption=False
        ),
    ]


def _fmt(value: float, width: int, decimals: int) -> str:
    return f"{value:>{width}.{decimals}f}"


def header() -> str:
    return "  ".join(
        [
            "scenario".ljust(24),
            "makespan",
            "ttft p50",
            "ttft p95",
            "tpot p95",
            "wait p95",
            "out tok/s",
            "goodput",
            "batch",
            "preempt",
        ]
    )


def row(name: str, m: ServingMetrics) -> str:
    return "  ".join(
        [
            name.ljust(24),
            _fmt(m.makespan_ms / 1000, 8, 2),
            _fmt(m.ttft_p50_ms, 8, 1),
            _fmt(m.ttft_p95_ms, 8, 1),
            _fmt(m.tpot_p95_ms, 8, 2),
            _fmt(m.queue_wait_p95_ms, 8, 1),
            _fmt(m.output_tokens_per_second, 8, 1),
            _fmt(m.goodput_tokens_per_second, 8, 1),
            _fmt(m.mean_batch_size, 5, 2),
            f"{m.preemptions:>7d}",
        ]
    )


def as_json(rows: list[tuple[str, ServingMetrics]]) -> list[dict[str, float | str]]:
    return [
        {
            "scenario": name,
            "makespan_s": m.makespan_ms / 1000,
            "ttft_p50_ms": m.ttft_p50_ms,
            "ttft_p95_ms": m.ttft_p95_ms,
            "tpot_p95_ms": m.tpot_p95_ms,
            "queue_wait_p95_ms": m.queue_wait_p95_ms,
            "output_tokens_per_second": m.output_tokens_per_second,
            "goodput_tokens_per_second": m.goodput_tokens_per_second,
            "mean_batch_size": m.mean_batch_size,
            "preemptions": m.preemptions,
        }
        for name, m in rows
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="keel scheduler benchmark")
    parser.add_argument("--requests", type=int, default=150)
    parser.add_argument("--shared-prefix", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    spec = WorkloadSpec(
        num_requests=args.requests,
        shared_prefix_tokens=args.shared_prefix,
        seed=args.seed,
    )

    rows: list[tuple[str, ServingMetrics]] = [
        (
            scenario.name,
            run_sequential(spec) if scenario.mode == "sequential" else run_batched(spec, scenario),
        )
        for scenario in scenarios()
    ]

    if args.json:
        json.dump(as_json(rows), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    print(
        f"requests={spec.num_requests} seed={spec.seed} "
        f"shared_prefix={spec.shared_prefix_tokens} "
        f"slo: ttft<={TTFT_SLO_MS:.0f}ms tpot<={TPOT_SLO_MS:.0f}ms"
    )
    print(f"device={DeviceProfile().name} block_size={BLOCK_SIZE} context={CONTEXT}")
    print()
    print(header())
    print("-" * len(header()))
    for name, metrics in rows:
        print(row(name, metrics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
