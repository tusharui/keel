from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

from keel.sim_engine.tokenizer import EOS, RESERVED


def stable_digest(*parts: object) -> int:
    payload = ":".join(str(p) for p in parts).encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


class SimModel:
    """Stand-in for a transformer that is deterministic and free.

    Produces the same token for the same (previous token, position) pair every
    time, which is what makes the scheduler's batching decisions reproducible.
    A real model would be sampled; here the sampling seed is derived from the
    sequence itself, so replaying a run reproduces it exactly.

    EOS arrives on roughly one token in _EOS_RATE, giving completions a
    geometric length distribution rather than always running to max_tokens.
    The shape matters: a scheduler benchmark where every sequence finishes at
    once measures nothing, because no request ever arrives mid-batch.
    """

    __slots__ = ("_body_vocab", "_seed")

    _EOS_RATE = 24

    def __init__(self, vocab_size: int, *, seed: int = 0xC0FFEE) -> None:
        if vocab_size <= RESERVED:
            raise ValueError(f"vocab_size must exceed {RESERVED}")
        self._seed = seed
        self._body_vocab = vocab_size - RESERVED

    def next_token(self, last_token: int, position: int) -> int:
        h = stable_digest(self._seed, last_token, position)
        # EOS is sticky: once a sequence has ended it must not emit EOS again,
        # or a resumed or batched sequence would look finished twice over.
        if h % self._EOS_RATE == 0 and last_token != EOS:
            return EOS
        # Avoid emitting the same id twice in a row; real models rarely do and a
        # run of identical tokens makes cache-hit rates look better than they are.
        candidate = RESERVED + (h >> 12) % self._body_vocab
        if candidate == last_token:
            candidate = RESERVED + (candidate + 1 - RESERVED) % self._body_vocab
        return candidate


@dataclass(frozen=True, slots=True)
class DeviceProfile:
    """Roofline cost model: compute-bound prefill, memory-bound decode.

    Prefill reads every weight once per token, so its cost scales with the
    token count. Decode reads the same weights for the entire batch no matter
    how many sequences it contains, so its cost is nearly flat in batch size and
    only the attention term grows. That asymmetry is the entire reason batching
    pays for itself, and it falls out of the two equations rather than being
    asserted.
    """

    name: str = "a100-80gb"
    parameters: int = 70_000_000_000
    flops_per_s: float = 312e12
    memory_bandwidth_bytes_per_s: float = 2_039e9
    kv_bytes_per_token_per_seq: int = 160 * 1024

    def prefill_seconds(self, num_tokens: int) -> float:
        if num_tokens <= 0:
            return 0.0
        return 2.0 * self.parameters * num_tokens / self.flops_per_s

    def decode_seconds(self, context_lengths: Sequence[int]) -> float:
        if not context_lengths:
            return 0.0
        weight_traffic = self.parameters * 2.0 / self.memory_bandwidth_bytes_per_s
        attention = sum(context_lengths) * self.kv_bytes_per_token_per_seq * 2.0 / self.flops_per_s
        return weight_traffic + attention

    @property
    def kv_bytes_per_token(self) -> int:
        return self.kv_bytes_per_token_per_seq
