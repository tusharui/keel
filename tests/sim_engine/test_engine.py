from __future__ import annotations

import pytest

from keel.clock import ManualClock
from keel.errors import ContextLengthExceeded
from keel.sim_engine import DeviceProfile, SimModel, Tokenizer
from keel.sim_engine.engine import FinishReason, Generation, InferenceEngine
from keel.sim_engine.kv_cache import KVCacheManager


@pytest.fixture
def clock() -> ManualClock:
    return ManualClock()


@pytest.fixture
def engine(clock: ManualClock) -> InferenceEngine:
    return InferenceEngine(
        tokenizer=Tokenizer(vocab_size=4096),
        model=SimModel(vocab_size=4096),
        device=DeviceProfile(),
        kv=KVCacheManager(num_blocks=512, block_size=16),
        context_length=4096,
        clock=clock,
    )


@pytest.fixture
def prompt() -> list[int]:
    return list(range(16, 48))


def test_generation_produces_text(engine: InferenceEngine, prompt: list[int]) -> None:
    result = engine.run(prompt, max_tokens=32)
    assert result.text
    assert result.prompt_tokens == len(prompt)
    assert result.completion_tokens == len(result.token_ids)


def test_max_tokens_is_respected(engine: InferenceEngine, prompt: list[int]) -> None:
    result = engine.run(prompt, max_tokens=5)
    assert result.completion_tokens <= 5
    assert result.finish_reason in (FinishReason.STOP, FinishReason.LENGTH)


def test_zero_max_tokens_yields_nothing(engine: InferenceEngine, prompt: list[int]) -> None:
    result = engine.run(prompt, max_tokens=0)
    assert result.token_ids == []
    assert result.text == ""
    assert result.finish_reason is FinishReason.LENGTH


def test_eos_terminates_early(engine: InferenceEngine, prompt: list[int]) -> None:
    generous = engine.run(prompt, max_tokens=2048)
    assert generous.finish_reason is FinishReason.STOP
    assert generous.completion_tokens < 2048


def test_runs_are_reproducible(engine: InferenceEngine, prompt: list[int]) -> None:
    assert (
        engine.run(prompt, max_tokens=40).token_ids == engine.run(prompt, max_tokens=40).token_ids
    )


def test_simulated_time_advances(engine: InferenceEngine, clock: ManualClock) -> None:
    before = clock.now()
    engine.run(list(range(16, 48)), max_tokens=32)
    assert clock.now() > before


def test_ttft_is_only_the_prefill(engine: InferenceEngine, prompt: list[int]) -> None:
    result = engine.run(prompt, max_tokens=200)
    assert 0 < result.ttft_ms < result.duration_ms


def test_tpot_excludes_prefill(engine: InferenceEngine) -> None:
    result = Generation(
        text="",
        token_ids=[1, 2, 3, 4],
        finish_reason=FinishReason.LENGTH,
        prompt_tokens=10,
        completion_tokens=4,
        ttft_ms=100.0,
        duration_ms=200.0,
    )
    assert result.tpot_ms == pytest.approx((200.0 - 100.0) / 3)


def test_tpot_is_undefined_for_a_single_token() -> None:
    result = Generation(
        text="",
        token_ids=[1],
        finish_reason=FinishReason.STOP,
        prompt_tokens=10,
        completion_tokens=1,
        ttft_ms=50.0,
        duration_ms=50.0,
    )
    assert result.tpot_ms is None


def test_stop_sequence_truncates(engine: InferenceEngine, prompt: list[int]) -> None:
    baseline = engine.run(prompt, max_tokens=200)
    assert len(baseline.token_ids) > 4

    stop = baseline.text[:3]
    truncated = engine.run(prompt, max_tokens=200, stop=(stop,))
    assert truncated.text == ""
    assert truncated.finish_reason is FinishReason.STOP


def test_stop_sequence_at_a_later_offset(engine: InferenceEngine, prompt: list[int]) -> None:
    baseline = engine.run(prompt, max_tokens=200)
    stop = baseline.text[5:8]
    truncated = engine.run(prompt, max_tokens=200, stop=(stop,))
    assert truncated.text == baseline.text[:5]
    assert truncated.completion_tokens < baseline.completion_tokens


def test_context_overflow_is_rejected(engine: InferenceEngine) -> None:
    with pytest.raises(ContextLengthExceeded):
        engine.run(list(range(5000)), max_tokens=10)


def test_blocks_are_returned_after_a_run(engine: InferenceEngine, prompt: list[int]) -> None:
    engine.run(prompt, max_tokens=64)
    assert engine.kv.num_used_blocks == 0


def test_blocks_are_returned_even_when_overflowing(engine: InferenceEngine) -> None:
    with pytest.raises(ContextLengthExceeded):
        engine.run(list(range(5000)), max_tokens=10)
    assert engine.kv.num_used_blocks == 0


def test_repeated_prompts_reuse_the_prefix(engine: InferenceEngine, prompt: list[int]) -> None:
    engine.run(prompt, max_tokens=200)
    second = engine.run(prompt, max_tokens=200)
    assert second.cached_prefix_blocks > 0


def test_decode_step_advances_every_sequence(engine: InferenceEngine) -> None:
    kv = engine.kv
    a = kv.allocate(list(range(16, 48)), max_tokens=16)
    b = kv.allocate(list(range(100, 132)), max_tokens=16)
    engine.prefill(a, 32)
    engine.prefill(b, 32)
    a.pending_token, b.pending_token = 100, 200

    before = a.num_tokens + b.num_tokens
    engine.decode_step([a, b])
    assert a.num_tokens + b.num_tokens == before + 2
    kv.release(a)
    kv.release(b)


def test_empty_decode_step_is_free(engine: InferenceEngine, clock: ManualClock) -> None:
    before = clock.now()
    assert engine.decode_step([]) == 0.0
    assert clock.now() == before


def test_batched_decode_costs_less_than_sequential(engine: InferenceEngine) -> None:
    """The claim the whole scheduler rests on."""
    kv = engine.kv
    seqs = [kv.allocate(list(range(16 + i, 48 + i)), max_tokens=16) for i in range(8)]
    for seq in seqs:
        engine.prefill(seq, 32)
        seq.pending_token = 100

    batched = sum(engine.decode_step(seqs) for _ in range(4))
    one_at_a_time = sum(engine.decode_step([seq]) for seq in seqs for _ in range(4))
    for seq in seqs:
        kv.release(seq)

    assert batched < one_at_a_time / 4
