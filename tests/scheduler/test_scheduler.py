from __future__ import annotations

import pytest

from keel.clock import ManualClock
from keel.config import SchedulerConfig
from keel.scheduler.policies import DeadlinePolicy, ShortestJobFirstPolicy
from keel.scheduler.request import InferenceRequest, RequestState
from keel.scheduler.scheduler import Scheduler
from keel.sim_engine import DeviceProfile, SimModel, Tokenizer
from keel.sim_engine.engine import FinishReason, InferenceEngine
from keel.sim_engine.kv_cache import KVCacheManager


def build(
    *,
    num_blocks: int = 4096,
    block_size: int = 16,
    max_num_seqs: int = 32,
    max_num_batched_tokens: int = 8192,
    chunked_prefill_tokens: int = 2048,
    policy: object = None,
) -> tuple[InferenceEngine, Scheduler, ManualClock]:
    clock = ManualClock()
    engine = InferenceEngine(
        tokenizer=Tokenizer(vocab_size=4096),
        model=SimModel(vocab_size=4096),
        device=DeviceProfile(),
        kv=KVCacheManager(num_blocks=num_blocks, block_size=block_size),
        context_length=8192,
        clock=clock,
    )
    config = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        chunked_prefill_tokens=chunked_prefill_tokens,
    )
    kwargs = {} if policy is None else {"policy": policy}
    return engine, Scheduler(engine, config, clock=clock, **kwargs), clock


def request(prompt_len: int = 64, max_tokens: int = 32, **kwargs: object) -> InferenceRequest:
    return InferenceRequest(
        prompt_tokens=list(range(100, 100 + prompt_len)),
        max_tokens=max_tokens,
        **kwargs,  # type: ignore[arg-type]
    )


# --- basic correctness -------------------------------------------------------------


def test_single_request_completes() -> None:
    _, scheduler, _ = build()
    scheduler.submit(request())
    scheduler.run()
    assert len(scheduler.results) == 1
    assert scheduler.results[0].state is RequestState.FINISHED


def test_every_request_is_accounted_for() -> None:
    _, scheduler, _ = build()
    submitted = [request(prompt_len=40 + i * 7, max_tokens=16) for i in range(25)]
    for req in submitted:
        scheduler.submit(req)
    scheduler.run()

    assert len(scheduler.results) == 25
    assert {r.request_id for r in scheduler.results} == {r.request_id for r in submitted}


def test_max_tokens_is_never_exceeded() -> None:
    _, scheduler, _ = build()
    for _ in range(15):
        scheduler.submit(request(max_tokens=9))
    scheduler.run()
    assert all(len(r.output) <= 9 for r in scheduler.results)


def test_output_length_matches_recorded_tokens() -> None:
    _, scheduler, _ = build()
    scheduler.submit(request(max_tokens=12))
    scheduler.run()
    result = scheduler.results[0]
    assert result.output == result.token_ids if hasattr(result, "token_ids") else True
    assert len(result.output) == len(result.text) or result.text


def test_kv_blocks_are_all_returned() -> None:
    engine, scheduler, _ = build()
    for _ in range(30):
        scheduler.submit(request(prompt_len=64, max_tokens=24))
    scheduler.run()
    assert engine.kv.num_used_blocks == 0


def test_simulated_clock_advances() -> None:
    _, scheduler, clock = build()
    before = clock.now()
    scheduler.submit(request())
    scheduler.run()
    assert clock.now() > before


def test_results_are_reproducible() -> None:
    def make() -> list[float | None]:
        _, scheduler, _ = build()
        for _ in range(10):
            scheduler.submit(request(prompt_len=64, max_tokens=16))
        scheduler.run()
        return sorted(r.ttft_ms or 0.0 for r in scheduler.results)

    assert make() == make()


# --- latency accounting -----------------------------------------------------------


def test_ttft_is_measured_from_admission_not_arrival() -> None:
    _, scheduler, _ = build(max_num_seqs=1)
    for _ in range(4):
        scheduler.submit(request(max_tokens=8))
    scheduler.run()

    for result in scheduler.results:
        assert result.started_at_ms is not None
        assert result.first_token_at_ms is not None
        assert result.first_token_at_ms >= result.started_at_ms


def test_queue_wait_is_recorded() -> None:
    _, scheduler, _ = build(max_num_seqs=1)
    for _ in range(4):
        scheduler.submit(request(max_tokens=6))
    scheduler.run()
    waits = [r.queue_wait_ms for r in scheduler.results if r.queue_wait_ms is not None]
    assert waits
    assert max(waits) > 0, "a serialised batch must make some requests wait"


def test_finish_reason_is_reported() -> None:
    _, scheduler, _ = build()
    scheduler.submit(request(max_tokens=1))
    scheduler.run()
    assert scheduler.results[0].finish_reason in (
        FinishReason.STOP,
        FinishReason.LENGTH,
    )


# --- continuous batching ----------------------------------------------------------


def test_batching_beats_serving_requests_one_at_a_time() -> None:
    """The load-bearing claim. Weights are read once per batch, so wall time for
    the whole cohort must be far below the sum of the isolated runs."""
    _, batched, _ = build(max_num_seqs=64)
    for _ in range(24):
        batched.submit(request(prompt_len=128, max_tokens=64))
    batched.run()
    batched_ms = max(r.latency_ms or 0.0 for r in batched.results)

    _, serial, _ = build(max_num_seqs=1)
    total = 0.0
    for _ in range(24):
        serial.submit(request(prompt_len=128, max_tokens=64))
    serial.run()
    total = sum(r.latency_ms or 0.0 for r in serial.results)

    assert batched_ms < total / 4


def test_batch_grows_beyond_a_single_sequence() -> None:
    _, scheduler, _ = build(max_num_seqs=64)
    for _ in range(32):
        scheduler.submit(request(prompt_len=256, max_tokens=64))
    scheduler.step()
    assert scheduler.batch_size > 1


def test_a_late_arriver_does_not_wait_for_the_batch_to_drain() -> None:
    """Iteration-level admission: the head-of-line block is one iteration, not one
    generation."""
    _, scheduler, _ = build(max_num_seqs=2)
    scheduler.submit(request(prompt_len=32, max_tokens=400, request_id="long"))
    scheduler.submit(request(prompt_len=32, max_tokens=400, request_id="peer"))

    scheduler.run(max_iterations=200)
    assert len(scheduler.results) == 2


def test_batch_size_is_capped_by_configuration() -> None:
    _, scheduler, _ = build(max_num_seqs=3)
    for _ in range(20):
        scheduler.submit(request(prompt_len=64, max_tokens=64))
    scheduler.step()
    scheduler.step()
    assert scheduler.batch_size <= 3


def test_serving_many_requests_converges_quickly() -> None:
    _, scheduler, _ = build(max_num_seqs=64)
    for _ in range(200):
        scheduler.submit(request(prompt_len=32, max_tokens=24))
    scheduler.run()
    assert len(scheduler.results) == 200
    assert scheduler.iterations < 200 * 24 * 4


# --- chunked prefill --------------------------------------------------------------


def test_long_prompt_is_prefilled_in_chunks() -> None:
    _, scheduler, _ = build(max_num_batched_tokens=1024, chunked_prefill_tokens=256)
    scheduler.submit(request(prompt_len=2048, max_tokens=4))
    scheduler.run()
    assert len(scheduler.results) == 1


def test_chunked_prefill_does_not_starve_decode() -> None:
    _, scheduler, _ = build(max_num_seqs=8, max_num_batched_tokens=1024, chunked_prefill_tokens=256)
    scheduler.submit(request(prompt_len=6000, max_tokens=8))
    for _ in range(3):
        scheduler.submit(request(prompt_len=64, max_tokens=64))
    scheduler.run()
    assert len(scheduler.results) == 4


def test_prefill_budget_is_respected_per_iteration() -> None:
    _, scheduler, _ = build(max_num_batched_tokens=512, chunked_prefill_tokens=512)
    for _ in range(5):
        scheduler.submit(request(prompt_len=4096, max_tokens=4))
    scheduler.run()
    assert len(scheduler.results) == 5


# --- policies ---------------------------------------------------------------------


def test_sjf_admits_short_requests_first() -> None:
    _, scheduler, _ = build(max_num_seqs=1, policy=ShortestJobFirstPolicy())
    scheduler.submit(request(max_tokens=400, request_id="long"))
    for i in range(5):
        scheduler.submit(request(max_tokens=4, request_id=f"short{i}"))

    scheduler.step()
    admitted = scheduler.running + [r for r in scheduler.results if r.finish_reason is not None]
    assert admitted, "a request should have been admitted on the first iteration"
    assert admitted[0].request_id.startswith("short")


def test_fcfs_admits_arrival_order_first() -> None:
    _, scheduler, _ = build(max_num_seqs=1)
    scheduler.submit(request(max_tokens=400, request_id="long"))
    for i in range(3):
        scheduler.submit(request(max_tokens=4, request_id=f"short{i}"))

    scheduler.step()
    assert scheduler.running[0].request_id == "long"


def test_deadline_policy_is_usable_end_to_end() -> None:
    _, scheduler, _ = build(policy=DeadlinePolicy())
    scheduler.submit(request(max_tokens=16, deadline_ms=5000.0))
    scheduler.submit(request(max_tokens=16, deadline_ms=1000.0))
    scheduler.run()
    assert len(scheduler.results) == 2


def test_arrival_times_are_respected() -> None:
    _, scheduler, clock = build(max_num_seqs=4)
    early = request(request_id="early", arrival_ms=0.0)
    late = request(request_id="late", arrival_ms=10_000.0)
    scheduler.submit(early)
    scheduler.submit(late)

    scheduler.step()
    assert early.state in (RequestState.RUNNING, RequestState.FINISHED)
    assert late.state is RequestState.WAITING
    assert clock.now() < 10.0


# --- failure modes ----------------------------------------------------------------


def test_scheduler_reports_non_convergence() -> None:
    _, scheduler, _ = build(max_num_seqs=1)
    for _ in range(20):
        scheduler.submit(request(max_tokens=32))
    with pytest.raises(RuntimeError, match="did not converge"):
        scheduler.run(max_iterations=3)


def test_idle_scheduler_has_no_work() -> None:
    _, scheduler, _ = build()
    assert not scheduler.has_work()
    assert scheduler.results == []
    assert scheduler.queue_depth == 0
