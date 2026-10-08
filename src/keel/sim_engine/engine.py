from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from keel.clock import Clock, default_clock
from keel.errors import ContextLengthExceeded
from keel.sim_engine.kv_cache import KVCacheManager, SequenceState
from keel.sim_engine.model import DeviceProfile, SimModel
from keel.sim_engine.tokenizer import EOS, Tokenizer


class FinishReason(StrEnum):
    STOP = "stop"
    LENGTH = "length"


@dataclass(frozen=True, slots=True)
class Generation:
    text: str
    token_ids: list[int]
    finish_reason: FinishReason
    prompt_tokens: int
    completion_tokens: int
    ttft_ms: float
    duration_ms: float
    cached_prefix_blocks: int = 0

    @property
    def tpot_ms(self) -> float | None:
        """Inter-token latency, excluding the prefill that produced token one.

        Averaging total latency over token count conflates two very different
        costs. Anything inspecting streaming smoothness wants this number, not
        duration divided by tokens.
        """
        if self.completion_tokens < 2:
            return None
        return (self.duration_ms - self.ttft_ms) / (self.completion_tokens - 1)


def longest_stop(stops: tuple[str, ...]) -> int:
    return max((len(s) for s in stops), default=0)


def find_stop(text: str, stops: tuple[str, ...], search_from: int) -> tuple[int, int] | None:
    """Earliest stop sequence at or after ``search_from``, as (index, length).

    The caller is responsible for a window that reaches back past the boundary
    where the newest token was appended. A stop sequence can straddle that
    boundary, and a window starting at the new tail will never find it.
    """
    earliest: tuple[int, int] | None = None
    for stop in stops:
        index = text.find(stop, search_from)
        if index != -1 and (earliest is None or index < earliest[0]):
            earliest = (index, len(stop))
    return earliest


class InferenceEngine:
    """Prefill and decode for one sequence at a time.

    This is the reference implementation: correct, obvious, and roughly an order
    of magnitude slower than the batched scheduler, because it reads the model
    weights once per token per request. It exists so the scheduler has something
    honest to be compared against.
    """

    __slots__ = ("_clock", "_device", "_kv", "_model", "_tokenizer", "context_length")

    def __init__(
        self,
        *,
        tokenizer: Tokenizer,
        model: SimModel,
        device: DeviceProfile,
        kv: KVCacheManager,
        context_length: int,
        clock: Clock | None = None,
    ) -> None:
        self._tokenizer = tokenizer
        self._model = model
        self._device = device
        self._kv = kv
        self._clock = clock or default_clock()
        self.context_length = context_length

    @property
    def kv(self) -> KVCacheManager:
        return self._kv

    @property
    def tokenizer(self) -> Tokenizer:
        return self._tokenizer

    @property
    def device(self) -> DeviceProfile:
        return self._device

    def sample(self, seq: SequenceState) -> int:
        """Draw the next token for a sequence whose prefill has completed."""
        return self._model.next_token(seq.tokens[-1], seq.num_tokens - 1)

    def prefill(self, seq: SequenceState, num_tokens: int) -> float:
        """Charge the prefill pass. Time advances even though nothing is emitted."""
        if seq.prompt_len + seq.max_tokens > self.context_length:
            raise ContextLengthExceeded(
                f"{seq.prompt_len}+{seq.max_tokens} exceeds context window {self.context_length}"
            )
        cost = self._device.prefill_seconds(num_tokens)
        self._clock.advance(cost)
        return cost

    def decode_step(self, seqs: list[SequenceState]) -> float:
        """One decode iteration across a batch.

        Every sequence advances exactly one token and the cost is charged once
        for the whole batch. That single accounting decision is what makes
        batching worth doing.
        """
        if not seqs:
            return 0.0
        cost = self._device.decode_seconds([s.num_tokens for s in seqs])
        for seq in seqs:
            self._kv.append(seq, seq.pending_token)
            seq.pending_token = self._model.next_token(seq.pending_token, seq.num_tokens)
        self._clock.advance(cost)
        return cost

    def run(
        self,
        prompt: list[int],
        max_tokens: int,
        *,
        stop: tuple[str, ...] = (),
    ) -> Generation:
        seq = self._kv.allocate(prompt, max_tokens)
        admitted_at = self._clock.now()
        try:
            self.prefill(seq, len(prompt))
            seq.prefill_completed = True

            token = self._model.next_token(seq.tokens[-1], seq.num_tokens - 1)
            seq.pending_token = token
            ttft_ms = (self._clock.now() - admitted_at) * 1000.0

            generated: list[int] = []
            text = ""
            finish = FinishReason.LENGTH

            while True:
                if token == EOS:
                    finish = FinishReason.STOP
                    break
                if len(generated) >= max_tokens:
                    finish = FinishReason.LENGTH
                    break

                generated.append(token)
                previous_len = len(text)
                text += self._tokenizer.decode([token])

                # Only the tail can hold a match that just appeared, but the
                # window has to reach back past the boundary: a stop sequence
                # can straddle the point where the last token was appended.
                found = find_stop(text, stop, max(0, previous_len - longest_stop(stop) + 1))
                if found is not None:
                    text = text[: found[0]]
                    finish = FinishReason.STOP
                    break

                self.decode_step([seq])
                token = seq.pending_token

            duration_ms = (self._clock.now() - admitted_at) * 1000.0
            return Generation(
                text=text,
                token_ids=generated,
                finish_reason=finish,
                prompt_tokens=len(prompt),
                completion_tokens=len(generated),
                ttft_ms=ttft_ms,
                duration_ms=duration_ms,
                cached_prefix_blocks=seq.shared_prefix_blocks,
            )
        finally:
            self._kv.release(seq)
