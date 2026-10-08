from __future__ import annotations

import pytest

from keel.clock import ManualClock
from keel.config import SchedulerConfig
from keel.errors import OutOfBlocks
from keel.scheduler.metrics import ServingMetrics, percentile
from keel.scheduler.request import InferenceRequest, RequestState
from keel.scheduler.scheduler import Scheduler
from keel.sim_engine import DeviceProfile, SimModel, Tokenizer
from keel.sim_engine.engine import InferenceEngine
from keel.sim_engine.kv_cache import KVCacheManager


def build(
    *,
    num_blocks: int = 4096,
    block_size: int = 16,
    max_num_seqs: int = 32,
    enable_preemption: bool = True,
) -> tuple[InferenceEngine, Scheduler]:
    clock = ManualClock()
    engine = InferenceEngine(
        tokenizer=Tokenizer(vocab_size=4096),
        model=SimModel(vocab_size=4096),
        device=DeviceProfile(),
        kv=KVCacheManager(num_blocks=num_blocks, block_size=block_size),
        context_length=8192,
        clock=clock,
    )
    config = SchedulerConfig(max_num_seqs=max_num_seqs, enable_preemption=enable_preemption)
    return engine, Scheduler(engine, config, clock=clock)


def request(
    prompt_len: int = 64, max_tokens: int = 16, *, offset: int = 0, **kwargs: object
) -> InferenceRequest:
    # Distinct prompts by default. Identical ones would share prefix blocks and
    # silently shrink each request's real memory footprint.
    return InferenceRequest(
        prompt_tokens=list(range(offset, offset + prompt_len)),
        max_tokens=max_tokens,
        **kwargs,  # type: ignore[arg-type]
    )


def big_requests(
    scheduler: Scheduler,
    count: int,
    *,
    prompt_len: int = 600,
    max_tokens: int = 20,
) -> None:
    """Long prompts, short completions.

    The prompt blocks are resident from admission, so these hold most of the
    pool for their whole lifetime. Short completions stop the model from
    finishing early and releasing memory before the queue can build pressure.
    """
    for i in range(count):
        scheduler.submit(
            request(
                prompt_len=prompt_len,
                max_tokens=max_tokens,
                offset=i * 2000,
                request_id=f"big{i}",
            )
        )


def small_requests(scheduler: Scheduler, count: int) -> None:
    for i in range(count):
        scheduler.submit(
            request(prompt_len=16, max_tokens=16, offset=900_000 + i * 100, request_id=f"sm{i}")
        )


# --- preemption -------------------------------------------------------------------


def test_preemption_frees_memory_for_a_new_request() -> None:
    """A 600-token prompt occupies 38 of 40 blocks the moment it is admitted, so a
    small newcomer can only be served by displacing it."""
    engine, scheduler = build(num_blocks=40, block_size=16, max_num_seqs=4)
    big_requests(scheduler, 6)
    small_requests(scheduler, 6)
    scheduler.run(max_iterations=200_000)

    assert len(scheduler.results) == 12
    assert scheduler.preemptions > 0
    assert engine.kv.num_used_blocks == 0


def test_preempted_request_still_produces_full_output() -> None:
    _, scheduler = build(num_blocks=40, block_size=16, max_num_seqs=4)
    big_requests(scheduler, 6)
    small_requests(scheduler, 6)
    scheduler.run(max_iterations=200_000)

    assert scheduler.preemptions > 0
    for result in scheduler.results:
        assert result.output, "a preempted request must not come back empty"
        assert len(result.output) <= result.max_tokens


def test_disabling_preemption_avoids_eviction() -> None:
    _, scheduler = build(num_blocks=40, block_size=16, max_num_seqs=4, enable_preemption=False)
    big_requests(scheduler, 6)
    small_requests(scheduler, 6)
    scheduler.run(max_iterations=200_000)

    assert scheduler.preemptions == 0
    assert len(scheduler.results) == 12


def test_preemption_does_not_thrash() -> None:
    """Equal-sized requests must not displace each other in a loop. The victim
    guard refuses to evict a sequence that is not larger than the newcomer."""
    _, scheduler = build(num_blocks=40, block_size=16, max_num_seqs=4)
    for i in range(6):
        scheduler.submit(
            request(prompt_len=150, max_tokens=150, offset=i * 1000, request_id=f"r{i}")
        )
    scheduler.run(max_iterations=200_000)

    assert len(scheduler.results) == 6
    assert scheduler.preemptions == 0


def test_preemption_clears_partial_output() -> None:
    _, scheduler = build(num_blocks=40, block_size=16, max_num_seqs=4)
    big_requests(scheduler, 6)
    small_requests(scheduler, 6)
    scheduler.run(max_iterations=200_000)

    assert scheduler.preemptions > 0
    assert all(r.preemptions == 0 or r.finish_reason is not None for r in scheduler.results)


def test_unservable_request_is_reported_not_spun_on() -> None:
    _, scheduler = build(num_blocks=8, block_size=16, max_num_seqs=4)
    scheduler.submit(request(prompt_len=4000, max_tokens=4000))
    with pytest.raises(OutOfBlocks, match="pool holds"):
        scheduler.run(max_iterations=1000)


def test_preempted_request_is_counted_in_metrics() -> None:
    _, scheduler = build(num_blocks=40, block_size=16, max_num_seqs=4)
    big_requests(scheduler, 6)
    small_requests(scheduler, 6)
    scheduler.run(max_iterations=200_000)

    metrics = scheduler.metrics()
    assert metrics.preemptions == scheduler.preemptions
    assert metrics.num_finished == 12


# --- metrics ----------------------------------------------------------------------


def test_percentile_of_empty_is_zero() -> None:
    assert percentile([], 0.5) == 0.0


def test_percentile_of_single_value() -> None:
    assert percentile([7.0], 0.99) == 7.0


def test_percentile_interpolates() -> None:
    assert percentile([0.0, 10.0], 0.5) == pytest.approx(5.0)
    assert percentile([0.0, 10.0], 0.0) == 0.0
    assert percentile([0.0, 10.0], 1.0) == 10.0


def test_percentile_handles_unsorted_input() -> None:
    assert percentile([10.0, 0.0, 5.0], 0.5) == pytest.approx(5.0)


def test_metrics_report_throughput() -> None:
    _, scheduler = build(max_num_seqs=32)
    for _ in range(20):
        scheduler.submit(request(prompt_len=64, max_tokens=32))
    scheduler.run()

    metrics = scheduler.metrics()
    assert metrics.num_finished == 20
    assert metrics.completion_tokens > 0
    assert metrics.output_tokens_per_second > 0
    assert metrics.mean_batch_size > 0
    assert metrics.makespan_ms > 0


def test_goodput_never_exceeds_raw_throughput() -> None:
    _, scheduler = build(max_num_seqs=32)
    for _ in range(15):
        scheduler.submit(request(prompt_len=64, max_tokens=32))
    scheduler.run()

    generous = scheduler.metrics(ttft_slo_ms=10_000.0, tpot_slo_ms=10_000.0)
    strict = scheduler.metrics(ttft_slo_ms=0.001, tpot_slo_ms=0.001)
    assert strict.goodput_tokens_per_second == 0.0
    assert strict.goodput_tokens_per_second <= generous.goodput_tokens_per_second


def test_metrics_handle_an_empty_run() -> None:
    _, scheduler = build()
    metrics = scheduler.metrics()
    assert metrics.num_requests == 0
    assert metrics.output_tokens_per_second == 0.0
    assert metrics.makespan_ms == 0.0


def test_tail_percentiles_exceed_the_median() -> None:
    _, scheduler = build(max_num_seqs=32)
    for _ in range(40):
        scheduler.submit(request(prompt_len=200, max_tokens=40))
    scheduler.run()

    metrics = scheduler.metrics()
    assert metrics.ttft_p95_ms >= metrics.ttft_p50_ms
    assert metrics.latency_p95_ms >= metrics.latency_p50_ms


def test_metrics_render_as_rows() -> None:
    _, scheduler = build(max_num_seqs=8)
    scheduler.submit(request())
    scheduler.run()
    rows = dict(scheduler.metrics().as_rows())
    assert "ttft p50/p95/p99" in rows
    assert rows["requests"] == "1/1"


def test_state_after_a_full_run_is_terminal() -> None:
    _, scheduler = build()
    for _ in range(5):
        scheduler.submit(request())
    scheduler.run()
    assert all(r.state is RequestState.FINISHED for r in scheduler.results)


def test_serving_metrics_from_results_directly() -> None:
    requests = [
        InferenceRequest(
            request_id=str(i),
            prompt_tokens=[1, 2, 3],
            max_tokens=4,
            output=[1, 2, 3],
            state=RequestState.FINISHED,
            arrival_ms=0.0,
            started_at_ms=1.0,
            first_token_at_ms=5.0,
            finished_at_ms=20.0,
            enqueued_at_ms=0.5,
        )
        for i in range(4)
    ]
    metrics = ServingMetrics.from_results(requests, makespan_ms=1000.0, mean_batch_size=2.0)
    assert metrics.num_finished == 4
    assert metrics.ttft_p50_ms == pytest.approx(4.0)
    assert metrics.queue_wait_p50_ms == pytest.approx(0.5)
    # Decode window is 15ms spread across 2 gaps between 3 output tokens.
    assert metrics.tpot_p50_ms == pytest.approx(7.5)
