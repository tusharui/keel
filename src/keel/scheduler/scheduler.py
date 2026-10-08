from __future__ import annotations

from keel.clock import Clock, default_clock
from keel.config import SchedulerConfig
from keel.errors import OutOfBlocks
from keel.scheduler.policies import AdmissionPolicy, FCFSPolicy
from keel.scheduler.request import InferenceRequest, RequestState
from keel.sim_engine.engine import FinishReason, InferenceEngine, find_stop, longest_stop
from keel.sim_engine.kv_cache import SequenceState
from keel.sim_engine.tokenizer import EOS


def _seq(request: InferenceRequest) -> SequenceState:
    """Narrow the optional sequence, loudly if the invariant is broken.

    An assert would disappear under ``python -O`` and turn a scheduler bug into a
    silently malformed batch.
    """
    seq = request.seq
    if seq is None:
        raise RuntimeError(f"request {request.request_id} has no live sequence")
    return seq


class Scheduler:
    """Continuous batching with iteration-level admission.

    A static batch is decided once and executed to completion. Here the batch is
    reassembled every iteration: finished sequences leave and waiting requests
    take their slots immediately. That is the whole difference, and it is why a
    request arriving one millisecond into a long generation waits one iteration
    rather than one entire generation.
    """

    __slots__ = (
        "_clock",
        "_config",
        "_decoding",
        "_engine",
        "_finished",
        "_inbox",
        "_iterations",
        "_policy",
        "_prefill",
        "_tokenizer",
        "_waiting",
    )

    def __init__(
        self,
        engine: InferenceEngine,
        config: SchedulerConfig,
        *,
        policy: AdmissionPolicy | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._engine = engine
        self._config = config
        self._policy = policy or FCFSPolicy()
        self._clock = clock or default_clock()
        self._tokenizer = engine.tokenizer
        self._inbox: list[InferenceRequest] = []
        self._waiting: list[InferenceRequest] = []
        self._prefill: list[InferenceRequest] = []
        self._decoding: list[InferenceRequest] = []
        self._finished: list[InferenceRequest] = []
        self._iterations = 0

    # --- properties ---------------------------------------------------------------

    @property
    def policy(self) -> AdmissionPolicy:
        return self._policy

    @property
    def iterations(self) -> int:
        return self._iterations

    @property
    def results(self) -> list[InferenceRequest]:
        return list(self._finished)

    @property
    def running(self) -> list[InferenceRequest]:
        return list(self._decoding)

    @property
    def waiting(self) -> list[InferenceRequest]:
        return list(self._waiting)

    @property
    def queue_depth(self) -> int:
        return len(self._inbox) + len(self._waiting) + len(self._prefill)

    @property
    def batch_size(self) -> int:
        return len(self._decoding)

    def has_work(self) -> bool:
        return bool(self._inbox or self._waiting or self._prefill or self._decoding)

    # --- lifecycle ----------------------------------------------------------------

    def submit(self, request: InferenceRequest) -> None:
        """Queue a request. It becomes eligible once simulated time reaches its
        arrival time, so an arrival stream can be modelled without interleaving
        submissions into the loop."""
        self._inbox.append(request)

    def _drain_inbox(self) -> None:
        if not self._inbox:
            return
        now_ms = self._clock.now() * 1000.0
        arrived = [r for r in self._inbox if r.arrival_ms <= now_ms]
        if not arrived:
            return
        for request in arrived:
            self._inbox.remove(request)
            request.enqueued_at_ms = now_ms
            self._waiting.append(request)

    # --- admission ----------------------------------------------------------------

    def _slots(self) -> int:
        return self._config.max_num_seqs - len(self._prefill) - len(self._decoding)

    def _admit(self) -> int:
        if not self._waiting or self._slots() <= 0:
            return 0

        slots = self._slots()
        admitted: list[InferenceRequest] = []
        for request in self._policy.order(self._waiting):
            if len(admitted) >= slots:
                break
            if not self._engine.kv.can_fit(request.prompt_tokens, request.max_tokens):
                break
            try:
                request.seq = self._engine.kv.allocate(request.prompt_tokens, request.max_tokens)
            except OutOfBlocks:
                break
            request.state = RequestState.RUNNING
            request.started_at_ms = self._clock.now() * 1000.0
            self._prefill.append(request)
            admitted.append(request)

        for request in admitted:
            self._waiting.remove(request)
        return len(admitted)

    # --- prefill ------------------------------------------------------------------

    def _run_prefill(self) -> bool:
        if not self._prefill:
            return False

        budget = self._config.max_num_batched_tokens
        did_work = False

        while self._prefill and budget > 0:
            request = self._prefill[0]
            seq = _seq(request)

            remaining = seq.prompt_len - seq.prefilled
            chunk = min(remaining, self._config.chunked_prefill_tokens, budget)
            self._engine.prefill(seq, chunk)
            seq.prefilled += chunk
            budget -= chunk
            did_work = True

            if seq.prefilled >= seq.prompt_len:
                self._prefill.remove(request)
                self._begin_decode(request)

        return did_work

    def _begin_decode(self, request: InferenceRequest) -> None:
        """First token falls out of the prefill pass, which is where TTFT ends."""
        seq = _seq(request)
        seq.prefill_completed = True
        seq.pending_token = self._engine.sample(seq)
        request.output.append(seq.pending_token)
        request.first_token_at_ms = self._clock.now() * 1000.0
        self._decoding.append(request)

    # --- decode -------------------------------------------------------------------

    def _run_decode(self) -> bool:
        if not self._decoding:
            return False

        self._engine.decode_step([r.seq for r in self._decoding if r.seq is not None])

        completed: list[tuple[InferenceRequest, FinishReason]] = []
        for request in self._decoding:
            seq = _seq(request)
            token = seq.pending_token

            previous_len = len(request.text)
            request.output.append(token)
            request.text += self._tokenizer.decode([token])

            if token == EOS:
                completed.append((request, FinishReason.STOP))
            elif len(request.output) >= request.max_tokens:
                completed.append((request, FinishReason.LENGTH))
            elif request.stop and find_stop(
                request.text, request.stop, max(0, previous_len - longest_stop(request.stop) + 1)
            ):
                completed.append((request, FinishReason.STOP))

        for request, reason in completed:
            self._finish(request, reason)
        return True

    def _finish(self, request: InferenceRequest, reason: FinishReason) -> None:
        request.state = RequestState.FINISHED
        request.finish_reason = reason
        request.finished_at_ms = self._clock.now() * 1000.0
        if request.seq is not None:
            self._engine.kv.release(request.seq)
            request.seq = None
        self._decoding.remove(request)
        self._finished.append(request)

    # --- main loop ----------------------------------------------------------------

    def step(self) -> bool:
        self._iterations += 1
        self._drain_inbox()
        self._admit()
        self._run_prefill()
        self._run_decode()
        return self.has_work()

    def run(self, *, max_iterations: int = 5_000_000) -> None:
        guard = 0
        while self.has_work():
            self.step()
            guard += 1
            if guard > max_iterations:
                raise RuntimeError(
                    f"scheduler did not converge after {max_iterations} iterations; "
                    f"waiting={len(self._waiting)} prefill={len(self._prefill)} "
                    f"decoding={len(self._decoding)}"
                )
